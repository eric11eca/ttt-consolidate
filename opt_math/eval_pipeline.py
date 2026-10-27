"""Orchestrator for the evaluation pipeline with test-time training.

For each test example the inner loop adapts the model's LoRA weights on the
example's context (a set of reasoning trajectories), then the adapted model
generates the answer. This is handled end to end by
``MetaLearningModule.validation_step``.

The pipeline:

1. builds the config the meta-learning module expects from the eval config;
2. loads the Lightning checkpoint (``MetaLearningModule.load_from_checkpoint``);
3. runs ``trainer.validate(model)``, which triggers per-example adaptation and
   generation (the ``generate`` stage);
4. scores the generations and writes them by case (subset x model x metric)
   into the run directory (the ``score`` stage).

The stages can run separately (``stages=[generate]`` on GPUs, then
``stages=[score]`` anywhere): generations persist in the resume store, which
is also what makes interrupted runs resumable.
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, List, Optional

from opt_math.eval_config import EvalConfig
from opt_math.eval_io import (
    answer_pairs_path,
    ensure_dir,
    ensure_case_dirs,
    eval_resume_file_template,
    generation_path,
    metrics_path,
    read_jsonl,
    summary_path,
    write_json,
    write_jsonl,
)
from opt_math.scorer import MathScorer, EvalCase
from opt_math.data_reader import EVAL_SOURCES

LOGGER = logging.getLogger("opt_math.eval_pipeline")


def _is_rank_zero() -> bool:
    try:
        import torch.distributed as dist
    except ImportError:
        dist = None
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    for env_name in ("RANK", "LOCAL_RANK"):
        env_value = os.environ.get(env_name)
        if env_value is None:
            continue
        try:
            return int(env_value) == 0
        except ValueError:
            continue
    return True


def _barrier_if_distributed() -> None:
    try:
        import torch.distributed as dist
    except ImportError:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def validate_config(config: EvalConfig) -> None:
    for metric in config.eval_metrics:
        if not metric.startswith("pass@"):
            raise ValueError(f"Invalid metric format: {metric!r}. Expected 'pass@<k>'.")
        try:
            int(metric.split("@")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid metric format: {metric!r}. Expected 'pass@<k>'.")
    unknown = [s for s in config.eval_subsets if s.lower() != "all" and s not in EVAL_SOURCES]
    if unknown:
        raise ValueError(
            f"Unknown eval_subsets {unknown}; expected 'all' or a subset of {list(EVAL_SOURCES)}"
        )


def resolve_subsets(config: EvalConfig) -> List[str]:
    """Resolve the subset list; ``["all"]`` expands to every benchmark source."""
    if len(config.eval_subsets) == 1 and config.eval_subsets[0].lower() == "all":
        return list(EVAL_SOURCES)
    return list(config.eval_subsets)


def build_perk_config(
    config: EvalConfig, subset: str, eval_model: str, metric: str
) -> Any:
    """Build the config ``MetaLearningModule`` expects, for one case."""
    from omegaconf import OmegaConf

    num_samples = config.num_samples_for_metric(metric)
    temperature = config.temperature_for_metric(metric)
    top_p = config.top_p_for_metric(metric)
    do_sample = not config.is_greedy(metric)
    resume_template = eval_resume_file_template(config, subset, eval_model, metric)

    return OmegaConf.create({
        # Model
        "model_name_or_path": config.model_name_or_path,
        "inner_start": config.inner_start,
        "inner_end": config.inner_end,
        "peft_lora": config.peft_lora,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": 0.0,
        "use_rslora": config.use_rslora,
        "lora_target_modules": list(config.lora_target_modules),
        "mem_depth": config.mem_depth,
        "bf16": config.bf16,
        "attn_implementation": config.attn_implementation,

        # Inner loop
        "inner_loop_type": config.inner_loop_type,
        "outer_loss_type": config.outer_loss_type,
        "n_inner_iter": config.n_inner_iter,
        "inner_lr": config.inner_lr,
        "inner_beta1": config.inner_beta1,
        "inner_beta2": config.inner_beta2,
        "inner_eps": config.inner_eps,
        "inner_grad_accum": config.inner_grad_accum,
        "unroll_start": config.unroll_start,
        "inner_opt": config.inner_opt,
        "c_kl": config.c_kl,
        "outer_reg_lambda": config.outer_reg_lambda,
        "weight_decay": config.weight_decay,

        # Weighting
        "weighted_loss": config.weighted_loss,
        "weight_normalize": config.weight_normalize,
        "weight_non_linearity": config.weight_non_linearity,
        "weighting_model": config.weighting_model,

        # Data
        "dataset_subset": subset,
        "eval_repo_prefix": config.eval_repo_prefix,
        "eval_revision": config.eval_revision,
        "conditioned": config.conditioned,
        "max_context_length": config.max_context_length,
        "packing": config.packing,
        "max_eval_data": config.max_eval_data,
        "source_model": config.source_model,
        "num_thinking": config.num_thinking,
        "strip_generated_boxed_answers": config.strip_generated_boxed_answers,
        "seed": config.seed,
        "dataset_cache_version": "v1",

        # Lightning bookkeeping (not used for evaluation)
        "do_train": False,
        "do_eval": True,
        "skip_builtin_validation_metrics": True,
        "test_only_dataset": True,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "num_train_epochs": 1,
        "callback_monitor": "val_f1",
        "learning_rate": 5e-5,
        "max_grad_norm": 5.0,
        "warmup_proportion": 0.03,
        "adam_epsilon": 1e-8,
        "val_check_interval": 0.0,
        "gradient_accumulation_steps": 1,
        "accelerator": config.accelerator,
        "n_gpu": config.n_gpu,
        "n_nodes": 1,
        "num_workers": 4,

        # Generation
        "inference_max_new_tokens": config.max_new_tokens,
        "eval_resume": config.resume,
        "eval_resume_file": resume_template,
        "eval_subset": subset,
        "eval_model": eval_model,
        "eval_metric": metric,
        "eval_num_samples": num_samples,
        "eval_temperature": temperature,
        "eval_top_p": top_p,
        "eval_do_sample": do_sample,

        # Output dirs
        "output_dir": config.output_dir,
        "run_dir": config.run_dir,
        "gen_dir": config.gen_dir,
        "eval_dir": config.eval_dir,

        # Logging
        "use_wandb": False,
        "wandb_project": "",

        # Checkpoint
        "checkpoint": config.checkpoint,
        "load_checkpoint": True,
        "resume_from_checkpoint": False,
    })


def collect_records_from_resume_files(
    config: EvalConfig,
    scorer: MathScorer,
    subset: str,
    eval_model: str,
    metric: str,
) -> List[Dict]:
    """Collect completed generation rows from the per-rank resume files.

    Resume files can hold duplicate ``record_uid`` rows after an interruption
    and restart. Duplicates are resolved by keeping a correct row when one
    exists (via the scorer's cheap correctness probe), so a stale incorrect row
    does not mask a later correct generation.
    """
    template = eval_resume_file_template(config, subset, eval_model, metric)
    paths = sorted(glob.glob(template.replace("{rank}", "*")))
    records_by_uid: Dict[str, Dict] = {}
    duplicate_rows = 0
    duplicate_uids = set()
    corrected_duplicates = 0
    for path in paths:
        for row_idx, row in enumerate(read_jsonl(path)):
            if row.get("output") is None:
                continue
            record_uid = row.get("record_uid") or f"{path}:{row_idx}"
            current = records_by_uid.get(record_uid)
            if current is None:
                records_by_uid[record_uid] = row
                continue

            duplicate_rows += 1
            duplicate_uids.add(record_uid)
            current_matches = scorer.quick_correctness(current)
            candidate_matches = scorer.quick_correctness(row)
            if candidate_matches and not current_matches:
                records_by_uid[record_uid] = row
                corrected_duplicates += 1
            elif current_matches and not candidate_matches:
                continue
            else:
                records_by_uid[record_uid] = row

    if duplicate_rows:
        LOGGER.info(
            "Deduplicated %d duplicate resume rows across %d record_uids for "
            "%s/%s/%s; selected a correct replacement for %d stale rows.",
            duplicate_rows,
            len(duplicate_uids),
            subset,
            eval_model,
            metric,
            corrected_duplicates,
        )
    return list(records_by_uid.values())


def _run_lightning_validate(
    config: EvalConfig,
    scorer: MathScorer,
    subset: str,
    eval_model: str,
    metric: str,
) -> List[Dict]:
    """Load the checkpoint, run Lightning validation, return generation records."""
    from perk.modules import MetaLearningModule
    from perk.train import setup_trainer

    perk_cfg = build_perk_config(config, subset, eval_model, metric)

    checkpoint_path = config.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(config.output_dir, checkpoint_path)

    LOGGER.info("Loading MetaLearningModule from checkpoint: %s", checkpoint_path)
    model = MetaLearningModule.load_from_checkpoint(
        checkpoint_path,
        config=perk_cfg,
        map_location="cpu",
    )
    # With test_only_dataset=True the module loads only the test split and
    # aliases train/val to it. Lightning validate() runs the adaptation and
    # generation because that is where MetaLearningModule implements them.
    trainer = setup_trainer(perk_cfg)

    LOGGER.info(
        "Running Lightning validate over test split for subset=%s, eval_model=%s, metric=%s",
        subset, eval_model, metric,
    )
    trainer.validate(model)
    _barrier_if_distributed()

    if config.resume:
        return [{
            "records": collect_records_from_resume_files(
                config, scorer, subset, eval_model, metric
            )
        }]

    # The module retains validation outputs at epoch end before clearing them.
    return getattr(model, "last_validation_step_outputs", model.validation_step_outputs)


def collect_records_from_outputs(outputs: List[Dict]) -> List[Dict]:
    """Extract generation records from Lightning validation step outputs."""
    all_records: List[Dict] = []
    for output in outputs:
        records = output.get("records", [])
        for record in records:
            if record.get("output") is not None:
                all_records.append(record)
    return all_records


def _score_and_write_case(
    config: EvalConfig,
    scorer: MathScorer,
    subset: str,
    eval_model: str,
    metric: str,
    records: List[Dict],
) -> Optional[Dict[str, Any]]:
    """Score records for one eval case and write outputs/metrics artifacts."""
    if not records:
        LOGGER.warning(
            "No records found for %s/%s/%s - skipping.",
            subset,
            eval_model,
            metric,
        )
        return None

    ctx = EvalCase(subset=subset, eval_model=eval_model, metric=metric, config=config)
    scored_rows = scorer.score_records(records, ctx)

    num_samples = config.num_samples_for_metric(metric)
    ensure_case_dirs(config.run_dir, subset, eval_model, metric)

    write_jsonl(scored_rows, generation_path(config.run_dir, subset, eval_model, metric))
    write_jsonl(
        [scorer.answer_pair_row(row) for row in scored_rows],
        answer_pairs_path(config.run_dir, subset, eval_model, metric),
    )

    completed_map: Dict[str, Dict[int, Dict]] = {}
    for row in scored_rows:
        qid = row["question_id"]
        sample_idx = row.get("sample_idx", 0)
        completed_map.setdefault(qid, {})[sample_idx] = row
    expected_sample_indices = set(range(num_samples))
    completed_map = {
        qid: sample_map
        for qid, sample_map in completed_map.items()
        if set(sample_map.keys()) >= expected_sample_indices
    }

    metrics = scorer.aggregate(completed_map, ctx, num_samples)
    by_source = scorer.aggregate_by_source(completed_map, ctx, num_samples)

    payload: Dict[str, Any] = {
        "subset": subset,
        "eval_model": eval_model,
        "source_model": config.source_model,
        "metric": metric,
        "num_thinking": config.num_thinking,
        "generation_num_samples": num_samples,
        "generation_temperature": config.temperature_for_metric(metric),
        "generation_top_p": config.top_p_for_metric(metric),
        "generation_do_sample": not config.is_greedy(metric),
        "checkpoint": config.checkpoint,
        **metrics,
    }
    if by_source:
        payload["by_source"] = by_source

    write_json(payload, metrics_path(config.run_dir, subset, eval_model, metric))

    metric_name, metric_value = scorer.primary_metric(metrics, ctx)
    LOGGER.info(
        "Metrics: subset=%s eval_model=%s source_model=%s "
        "num_thinking=%d metric=%s %s=%.4f (%d questions)",
        subset, eval_model, config.source_model,
        config.num_thinking, metric, metric_name, metric_value,
        metrics.get("num_questions", 0),
    )
    return payload


def _write_summary(config: EvalConfig, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {"source_model": config.source_model, "results": all_results}
    write_json(summary, summary_path(config.run_dir))
    LOGGER.info("Summary written to %s", summary_path(config.run_dir))

    LOGGER.info("=== Evaluation Summary ===")
    for entry in all_results:
        metric = entry.get("metric", "?")
        k = int(metric.split("@")[1]) if "@" in metric else 1
        LOGGER.info(
            "  %s / %s / source_model=%s / num_thinking=%s / %s : "
            "pass@%d = %.4f  (%d questions)",
            entry.get("subset", "?"),
            entry.get("eval_model", "?"),
            entry.get("source_model", "?"),
            entry.get("num_thinking", "?"),
            metric, k,
            entry.get(f"pass@{k}", 0.0),
            entry.get("num_questions", 0),
        )
    return summary


def _validate_subsets(config: EvalConfig) -> List[str]:
    subsets = resolve_subsets(config)
    if not subsets:
        raise ValueError("No subsets resolved. Check eval_subsets.")
    if not config.eval_models:
        raise ValueError("No models specified in eval_models.")
    if not config.eval_metrics:
        raise ValueError("No metrics specified in eval_metrics.")
    return subsets


def score_resume(config: EvalConfig) -> Dict[str, Any]:
    """Score existing generations from the resume store, without running a model."""
    validate_config(config)
    scorer = MathScorer()

    ensure_dir(config.run_dir)
    ensure_dir(config.gen_dir)
    ensure_dir(config.eval_dir)

    subsets = _validate_subsets(config)

    LOGGER.info(
        "Resume-only eval scoring: subsets=%s, eval_models=%s, metrics=%s, "
        "source_model=%s, run_dir=%s",
        subsets, config.eval_models, config.eval_metrics,
        config.source_model, config.run_dir,
    )

    write_json(config.to_dict(), os.path.join(config.run_dir, "config.json"))

    all_results: List[Dict[str, Any]] = []
    for eval_model in config.eval_models:
        for subset in subsets:
            for metric in config.eval_metrics:
                LOGGER.info(
                    "=== Resume Score: subset=%s, eval_model=%s, metric=%s ===",
                    subset, eval_model, metric,
                )
                records = collect_records_from_resume_files(
                    config, scorer, subset, eval_model, metric
                )
                LOGGER.info(
                    "Collected %d resume generation records for %s/%s/%s",
                    len(records), subset, eval_model, metric,
                )
                payload = _score_and_write_case(
                    config, scorer, subset, eval_model, metric, records
                )
                if payload is not None:
                    all_results.append(payload)

    return _write_summary(config, all_results)


def run_eval(config: EvalConfig) -> Dict[str, Any]:
    """Run the evaluation pipeline with test-time training.

    For each (subset, eval_model, metric) case:
    1. load the Lightning checkpoint;
    2. adapt on each example's context and generate (the ``generate`` stage);
    3. score the generations and write the case metrics (the ``score`` stage);
    4. write the cross-case summary.
    """
    validate_config(config)
    scorer = MathScorer()

    do_generate = "generate" in config.stages
    do_score = "score" in config.stages
    if not do_generate and not do_score:
        raise ValueError("stages must include 'generate' and/or 'score'.")
    if not do_generate:
        return score_resume(config)
    if do_generate and not do_score and not config.resume:
        raise ValueError(
            "stages=['generate'] persists generations via the resume store; "
            "set resume=true (or include the 'score' stage)."
        )

    ensure_dir(config.run_dir)
    ensure_dir(config.gen_dir)
    ensure_dir(config.eval_dir)

    subsets = _validate_subsets(config)

    LOGGER.info(
        "Eval pipeline: subsets=%s, eval_models=%s, metrics=%s, source_model=%s, "
        "stages=%s, run_dir=%s, checkpoint=%s",
        subsets, config.eval_models, config.eval_metrics, config.source_model,
        config.stages, config.run_dir, config.checkpoint,
    )

    if _is_rank_zero():
        write_json(config.to_dict(), os.path.join(config.run_dir, "config.json"))
    _barrier_if_distributed()

    all_results: List[Dict[str, Any]] = []

    for eval_model in config.eval_models:
        for subset in subsets:
            for metric in config.eval_metrics:
                LOGGER.info(
                    "=== Eval: subset=%s, eval_model=%s, metric=%s ===",
                    subset, eval_model, metric,
                )

                outputs = _run_lightning_validate(
                    config, scorer, subset, eval_model, metric,
                )
                records = collect_records_from_outputs(outputs)

                LOGGER.info(
                    "Collected %d generation records for %s/%s/%s",
                    len(records), subset, eval_model, metric,
                )
                if not _is_rank_zero():
                    _barrier_if_distributed()
                    continue

                if do_score:
                    payload = _score_and_write_case(
                        config, scorer, subset, eval_model, metric, records
                    )
                    if payload is not None:
                        all_results.append(payload)
                _barrier_if_distributed()

    if do_score and _is_rank_zero():
        return _write_summary(config, all_results)
    return {"source_model": config.source_model, "results": all_results}
