"""Meta-learning module: inner-loop adaptation (RGU) with a cross-entropy outer loss.

The module composes an inner-loop strategy and an outer-loop strategy
(``perk.modules.inner_loop`` / ``perk.modules.outer_loop``) selected through
``inner_loop_type`` and ``outer_loss_type`` in the config.
"""

import gc
import glob
import json
import os
import pickle
import torch
import functorch
import logging
import warnings
import torch.distributed as dist

from typing import Any, Dict, List, Optional, Tuple
from torch.optim import AdamW

from perk.model import CausalLM
from perk.modules.base_module import BaseModule
from perk.utils.data_utils import get_features
from perk.modules.optimizers import LRParameters
from perk.modules.inner_loop import create_inner_loop, InnerLoopStrategy
from perk.modules.outer_loop import create_outer_loop, OuterLoopStrategy
from perk.inference import HFGenerator

from transformers import logging as hf_logging
hf_logging.set_verbosity_info()


logger = logging.getLogger("perk.meta_learner")


def _read_jsonl(path):
    """Rows of a JSONL file; a missing file gives no rows and malformed lines are skipped."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(rows, path):
    """Append ``rows`` to a JSONL file, creating its directory if needed."""
    if not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


class MetaLearningModule(BaseModule):
    """Meta-learning module with pluggable inner/outer loop strategies.

    Configuration options:
        inner_loop_type: str - type of inner loop (``rgu``)
        outer_loss_type: str - type of outer loss (``ce``)
    """

    def __init__(self, config):
        super().__init__(config, logger)

        self.model = CausalLM.from_config(config)
        self.tokenizer = self.model.tokenizer
        self.load_dataset()
        self.set_inner_params()
        self.inner_lr_scheduler_config(config.n_inner_iter, config.inner_lr)
        self.model_logger.info(
            f"PL module loaded | global_epoch={self.global_epoch_counter}")

        self.verbose_inner = False
        self.batch_skip = torch.tensor([0], device=self.device, dtype=torch.int32)

        self.inner_loop_type = getattr(config, "inner_loop_type", None) or "rgu"
        self.outer_loss_type = getattr(config, "outer_loss_type", "ce")
        self._inference_max_new_tokens = int(
            getattr(config, "inference_max_new_tokens", 16384)
        )
        self._inference_generator: Optional[HFGenerator] = None

        self.eval_resume = bool(getattr(config, "eval_resume", False))
        self.eval_resume_file = getattr(config, "eval_resume_file", None)
        self.eval_num_samples = max(1, int(getattr(config, "eval_num_samples", 1)))
        self.eval_metric = str(getattr(config, "eval_metric", "pass@1"))
        self.eval_subset = str(getattr(config, "eval_subset", getattr(config, "dataset_subset", "")))
        self.eval_model_name = str(getattr(config, "eval_model", getattr(config, "model_name_or_path", "")))
        self.eval_temperature = getattr(config, "eval_temperature", None)
        self.eval_top_p = getattr(config, "eval_top_p", None)
        self.eval_do_sample = bool(getattr(config, "eval_do_sample", False))
        self._eval_resume_initialized = False
        self._eval_resume_path: Optional[str] = None
        self._completed_eval_records: Dict[str, Dict[str, Any]] = {}

        if self.eval_resume:
            self.model_logger.info("Resumable validation inference is enabled.")

        self._inner_loop_strategy: Optional[InnerLoopStrategy] = None
        self._outer_loop_strategy: Optional[OuterLoopStrategy] = None

        self.model_logger.info(
            f"Meta module configured: inner_loop={self.inner_loop_type}, "
            f"outer_loop={self.outer_loss_type}"
        )

    def _get_inference_generator(self) -> HFGenerator:
        if self._inference_generator is None:
            self._inference_generator = HFGenerator(
                model=self.model.lm_model,
                tokenizer=self.model.tokenizer,
                device=self.device,
            )
        return self._inference_generator

    def _runtime_rank(self) -> int:
        if dist.is_initialized():
            return int(dist.get_rank())
        # Lightning's public ``trainer`` property raises when the module is
        # intentionally exercised before attachment (for example in unit
        # tests). Inspect the backing reference without triggering it.
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and getattr(trainer, "local_rank", None) is not None:
            return int(trainer.local_rank)
        return int(getattr(self, "local_rank", 0))

    def _resolve_eval_resume_path(self) -> str:
        rank = self._runtime_rank()
        if self.eval_resume_file:
            if "{rank}" in self.eval_resume_file:
                return self.eval_resume_file.format(rank=rank)
            root, ext = os.path.splitext(self.eval_resume_file)
            ext = ext or ".jsonl"
            return f"{root}.rank{rank}{ext}"
        run_dir = getattr(
            self.hparams,
            "run_dir",
            getattr(self.hparams, "output_dir", "."),
        )
        return os.path.join(run_dir, "resume_generations", f"generations_rank{rank}.jsonl")

    def _resolve_eval_resume_read_paths(self) -> List[str]:
        """Load completed records from every rank file for the active eval case."""
        write_path = self._resolve_eval_resume_path()
        if self.eval_resume_file:
            if "{rank}" in self.eval_resume_file:
                glob_path = self.eval_resume_file.replace("{rank}", "*")
            else:
                root, ext = os.path.splitext(self.eval_resume_file)
                glob_path = f"{root}.rank*{ext or '.jsonl'}"
        else:
            glob_path = os.path.join(os.path.dirname(write_path), "generations_rank*.jsonl")

        paths = sorted(glob.glob(glob_path))
        if write_path not in paths:
            paths.append(write_path)
        return paths

    def _init_eval_resume_store(self) -> None:
        if not self.eval_resume or self._eval_resume_initialized:
            return
        self._eval_resume_path = self._resolve_eval_resume_path()
        loaded = 0
        loaded_paths = 0
        for resume_path in self._resolve_eval_resume_read_paths():
            rows = _read_jsonl(resume_path)
            if rows:
                loaded_paths += 1
            for row in rows:
                record_uid = row.get("record_uid")
                if not record_uid:
                    continue
                if not row.get("complete", False):
                    continue
                if row.get("output") is None:
                    continue
                if record_uid not in self._completed_eval_records:
                    loaded += 1
                self._completed_eval_records[record_uid] = row
        self._eval_resume_initialized = True
        self.model_logger.info(
            "Loaded %d completed evaluation records from %d resume files; "
            "current rank will append to %s",
            loaded,
            loaded_paths,
            self._eval_resume_path,
        )

    def _build_eval_records(self, print_out: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompts = list(print_out.get("prompt", []))
        responses = list(print_out.get("response", []))
        guid = print_out.get("guid", "")
        task = print_out.get("task", "")
        gold_answers = print_out.get("gold_answer", None)
        records = []
        rank = self._runtime_rank()
        world_size = int(dist.get_world_size()) if dist.is_initialized() else 1
        num_samples = max(1, int(getattr(self, "eval_num_samples", 1)))
        eval_subset = str(getattr(self, "eval_subset", getattr(self.hparams, "dataset_subset", "")))
        eval_model = str(getattr(self, "eval_model_name", getattr(self.hparams, "model_name_or_path", "")))
        eval_metric = str(getattr(self, "eval_metric", "pass@1"))
        source_model = str(getattr(self.hparams, "source_model", ""))
        num_thinking = getattr(self.hparams, "num_thinking", None)

        for query_idx, (prompt, response) in enumerate(zip(prompts, responses)):
            guid_i = guid[query_idx] if isinstance(guid, (list, tuple)) else guid
            task_i = task[query_idx] if isinstance(task, (list, tuple)) else task
            for sample_idx in range(num_samples):
                record_uid = "::".join([
                    eval_subset,
                    eval_model,
                    eval_metric,
                    source_model,
                    str(num_thinking),
                    str(guid_i),
                    str(query_idx),
                    str(sample_idx),
                ])

                record = {
                    "guid": guid_i,
                    "task": task_i,
                    "prompt": prompt,
                    "answer": response,
                    "query_idx": query_idx,
                    "sample_idx": sample_idx,
                    "generation_idx": sample_idx,
                    "record_uid": record_uid,
                    "complete": False,
                    "subset": eval_subset,
                    "eval_model": eval_model,
                    "eval_metric": eval_metric,
                    "source_model": source_model,
                    "num_thinking": num_thinking,
                    "num_thinking_trajectories": num_thinking,
                    "inference_backend": "hf",
                    "generation_num_samples": num_samples,
                    "generation_temperature": getattr(self, "eval_temperature", None),
                    "generation_top_p": getattr(self, "eval_top_p", None),
                    "generation_do_sample": bool(getattr(self, "eval_do_sample", False)),
                    "rank": rank,
                    "world_size": world_size,
                }
                if gold_answers is not None:
                    record["gold_answer"] = (
                        gold_answers[query_idx]
                        if isinstance(gold_answers, (list, tuple))
                        else gold_answers
                    )

                cached = self._completed_eval_records.get(record_uid)
                if cached:
                    merged = dict(record)
                    merged.update(cached)
                    merged["complete"] = bool(cached.get("complete", False))
                    record = merged
                records.append(record)
        return records

    @staticmethod
    def _split_records(
        records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        completed = []
        pending = []
        for record in records:
            has_output = record.get("output") is not None
            if bool(record.get("complete", False)) and has_output:
                completed.append(record)
            else:
                pending.append(record)
        return completed, pending

    def _append_completed_eval_records(self, records: List[Dict[str, Any]]) -> None:
        if not self.eval_resume:
            return
        if self._eval_resume_path is None:
            return
        new_rows = []
        for record in records:
            record_uid = record.get("record_uid")
            if not record_uid:
                continue
            if not record.get("complete", False):
                continue
            if record.get("output") is None:
                continue
            if record_uid in self._completed_eval_records:
                continue
            row = dict(record)
            self._completed_eval_records[record_uid] = row
            new_rows.append(row)
        _append_jsonl(new_rows, self._eval_resume_path)

    def set_inner_params(self):
        self.inner_params = []
        self.inner_params_idx = []

        inner_layers = [f"layers.{i}." for i in range(
            self.hparams.inner_start,
            self.hparams.inner_end
        )]
        print("Layers adapted in the inner loop:")
        print(inner_layers)

        for idx, (n, p) in enumerate(self.model.named_parameters()):
            if "lora_" in n and any(l in n for l in inner_layers):
                p.requires_grad = True
                self.inner_params.append(n)
                self.inner_params_idx.append(idx)

            if "prompt_encoder" in n:
                self.inner_params.append(n)
                self.inner_params_idx.append(idx)

    def inner_lr_scheduler_config(self, n_inner_iter, inner_lr):
        _, params, _ = functorch.make_functional_with_buffers(self.model)
        params_with_grad = [
            p if p.requires_grad else torch.tensor([], requires_grad=True, device=self.device)
            for p in params
        ]
        self.lrs = LRParameters(
            num_inner_iter=n_inner_iter,
            init_lr=inner_lr,
            params=params_with_grad)

    def configure_optimizers(self):
        """Setup the main optimizer

        :rtype: torch.optim
        :returns: the main optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        params, params_no_decay = [], []

        for n, p in self.model.named_parameters():
            if not any(nd in n for nd in no_decay):
                params.append(p)
            else:
                params_no_decay.append(p)

        optimizer_grouped_parameters = [
            {"params": params, "weight_decay": self.hparams.weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
            {"params": self.lrs.parameters(), "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            betas=self._outer_adamw_betas(),
        )

        self.opt = optimizer
        scheduler = self.get_lr_scheduler()
        return [optimizer], [scheduler]

    def _outer_adamw_betas(self):
        """Outer AdamW betas from ``outer_beta1``/``outer_beta2``.

        The checkpoint contract records these config values, so the optimizer
        must consume the same ones (defaults 0.9/0.95 match the former
        hardcoded pair).
        """
        beta1 = float(getattr(self.hparams, "outer_beta1", 0.9))
        beta2 = float(getattr(self.hparams, "outer_beta2", 0.95))
        for name, value in (("outer_beta1", beta1), ("outer_beta2", beta2)):
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must lie in [0, 1); got {value}")
        return (beta1, beta2)

    @property
    def inner_loop_strategy(self) -> InnerLoopStrategy:
        """Lazily create inner loop strategy."""
        if self._inner_loop_strategy is None:
            self._inner_loop_strategy = create_inner_loop(
                inner_loop_type=self.inner_loop_type,
                model=self.model,
                inner_params_idx=self.inner_params_idx,
                lrs=self.lrs,
                hparams=self.hparams,
                device=self.device,
                verbose=self.verbose_inner,
                local_rank=self.local_rank,
            )
        return self._inner_loop_strategy

    @property
    def outer_loop_strategy(self) -> OuterLoopStrategy:
        """Lazily create outer loop strategy."""
        if self._outer_loop_strategy is None:
            self._outer_loop_strategy = create_outer_loop(
                outer_loop_type=self.outer_loss_type,
                model=self.model,
                hparams=self.hparams,
                device=self.device,
                local_rank=self.local_rank,
            )
        return self._outer_loop_strategy

    def _cleanup_after_batch_skip(self) -> None:
        """Clear gradients and free memory after a skipped batch."""
        self.model.zero_grad(set_to_none=True)
        if hasattr(self, "lrs"):
            self.lrs.zero_grad(set_to_none=True)
        if hasattr(self, "opt"):
            self.opt.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _is_cuda_oom_error(exc: BaseException) -> bool:
        """Return True for CUDA OOM exceptions raised as either typed or RuntimeError."""
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _skip_batch_output(self) -> Dict[str, Any]:
        """Return a zero-loss output dict for synchronized batch skipping."""
        self._manual_backward_done = False
        return {
            "loss": torch.tensor(0.0, device=self.device),
            "inner_loss": torch.tensor(0.0, device=self.device),
            "outer_loss": torch.tensor(0.0, device=self.device),
            "outer_gain": torch.tensor(0.0, device=self.device),
            "kl_div": torch.tensor(0.0, device=self.device),
            "params": [],
        }

    def _to_cpu_for_pickle(self, value):
        """Recursively move tensors to CPU so skipped batches can be serialized."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._to_cpu_for_pickle(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_cpu_for_pickle(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_cpu_for_pickle(v) for v in value)
        return value

    def _oom_skip_log_path(self) -> str:
        """Resolve per-rank path for skipped OOM batch records."""
        rank = dist.get_rank() if dist.is_initialized() else self.trainer.local_rank
        log_dir = (
            getattr(self.hparams, "run_dir", None)
            or getattr(self.hparams, "output_dir", ".")
        )
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"oom_skipped_batches_rank{rank}.pkl")

    def _record_skipped_oom_batch(self, batch, context_length: int) -> None:
        """
        Append one skipped batch record to disk.

        Record format is a tuple: (batch_on_cpu, context_length).
        The file is an append-only pickle stream for efficient growth.
        """
        record = (self._to_cpu_for_pickle(batch), int(context_length))
        log_path = self._oom_skip_log_path()
        try:
            with open(log_path, "ab") as fout:
                pickle.dump(record, fout, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.warning(
                f"Rank {self.trainer.local_rank} - Failed to write skipped OOM batch "
                f"to {log_path}: {exc}"
            )

    # ========================================================================
    # Main step function with strategy selection
    # ========================================================================

    def step(self, batch, return_parameters=False):
        inner_batches, outer_batches, print_out = get_features(
            batch,
            packing=getattr(self.hparams, "packing", "flash"),
            accumulation_steps=self.hparams.inner_grad_accum,
        )

        context_length = sum(b["input_ids"].shape[1] for b in inner_batches)

        if self.trainer.local_rank == 0 and self.verbose_inner:
            print(f"Number of inner batches: {len(inner_batches)}")
            print(f"Context length: {context_length}")

        self.batch_skip = torch.tensor([0], device=self.device, dtype=torch.int32)
        if not return_parameters and context_length > self.hparams.max_context_length:
            logger.info(f"Rank {self.trainer.local_rank} - Skipping batch due to context length: {context_length}")
            self.batch_skip[0] = 1

        if dist.is_initialized() and not return_parameters:
            dist.all_reduce(self.batch_skip, op=dist.ReduceOp.SUM)

        if self.batch_skip.item() > 0:
            self._cleanup_after_batch_skip()
            return self._skip_batch_output()

        # Allocate before risky ops so we can all_reduce even after OOM.
        oom_flag = torch.zeros(1, device=self.device, dtype=torch.int32)
        output_dict = None

        try:
            self.model.train()

            inner_result = self.inner_loop_strategy.run(
                inner_batches=inner_batches,
                outer_batches=outer_batches,
                return_parameters=return_parameters,
            )

            adapted_params = []
            if return_parameters:
                raw_params = inner_result.get("params", [])
                expected_params = len(list(self.model.parameters()))
                if not isinstance(raw_params, (list, tuple)):
                    logger.warning(
                        "Rank %s - Adapted params has invalid type %s; "
                        "falling back to current model parameters.",
                        self.trainer.local_rank,
                        type(raw_params).__name__,
                    )
                    adapted_params = list(self.model.parameters())
                elif len(raw_params) != expected_params:
                    logger.warning(
                        "Rank %s - Adapted params count mismatch "
                        "(expected=%d, got=%d); falling back to current model parameters.",
                        self.trainer.local_rank,
                        expected_params,
                        len(raw_params),
                    )
                    adapted_params = list(self.model.parameters())
                else:
                    adapted_params = list(raw_params)

            outer_result = self.outer_loop_strategy.compute_loss(
                inner_result=inner_result,
                outer_batches=outer_batches,
                print_out=print_out,
                log_fn=self.log,
            )

            output_dict = {
                "loss": outer_result["loss"],
                "inner_loss": inner_result["inner_loss"].detach(),
                "outer_loss": outer_result["loss"].detach() if isinstance(outer_result["loss"], torch.Tensor) else torch.tensor(outer_result["loss"], device=self.device),
                "params": adapted_params if return_parameters else [],
            }

            for k, v in outer_result.get("metrics", {}).items():
                output_dict[k] = v

            # Keep metric keys deterministic across ranks, including skip/OOM cases.
            output_dict.setdefault("outer_gain", torch.tensor(0.0, device=self.device))
            output_dict.setdefault("kl_div", torch.tensor(0.0, device=self.device))

            self._manual_backward_done = outer_result.get("manual_backward", False)

        except torch.cuda.OutOfMemoryError as oom:
            oom_flag[0] = 1
            oom_message = str(oom).splitlines()[0]
            logger.warning(
                f"Rank {self.trainer.local_rank} - CUDA OOM during step. "
                f"Local context length={context_length}. Error: {oom_message}"
            )
            self._cleanup_after_batch_skip()
        except RuntimeError as exc:
            if not self._is_cuda_oom_error(exc):
                raise
            oom_flag[0] = 1
            oom_message = str(exc).splitlines()[0]
            logger.warning(
                f"Rank {self.trainer.local_rank} - CUDA OOM during step. "
                f"Local context length={context_length}. Error: {oom_message}"
            )
            self._cleanup_after_batch_skip()

        if dist.is_initialized():
            dist.all_reduce(oom_flag, op=dist.ReduceOp.SUM)

        if oom_flag.item() > 0:
            self.batch_skip[0] = 1
            if output_dict is not None:
                del output_dict
            self._record_skipped_oom_batch(batch, context_length)
            logger.warning(
                f"Rank {self.trainer.local_rank} - Skipping batch due to synchronized CUDA OOM. "
                f"Logged skipped batch with context length={context_length}."
            )
            self._cleanup_after_batch_skip()

            if return_parameters:
                output_dict = self._skip_batch_output()
                output_dict["skip_loss"] = True
                output_dict["skip_inference"] = True
                output_dict["oom_skip"] = True
                return output_dict
            return self._skip_batch_output()

        torch.cuda.empty_cache()
        return output_dict

    def training_step(self, batch, batch_idx) -> Dict:
        if not self.hparams.do_train:
            return {"loss": torch.tensor(0.0, device=self.device)}

        output_dict = self.step(batch)

        for mkey in output_dict.keys():
            if mkey in ["inner_loss", "outer_loss", "outer_ce_loss", "outer_gain", "kl_div", "outer_grad_norm"]:
                val = output_dict[mkey]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, device=self.device)
                self.log(
                    f"{mkey}",
                    val,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=(mkey in ["inner_loss", "outer_loss"]),
                    sync_dist=True,
                    rank_zero_only=False
                )
        self.global_training_step += 1
        return output_dict

    def validation_step(self, batch, batch_idx) -> Dict:
        """Runs a single validation step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        def apply_inner_params(model, params):
            model_params = list(model.parameters())
            if not isinstance(params, (list, tuple)):
                logger.warning(
                    "Rank %s - Adapted params has invalid type %s in validation. "
                    "Using base model parameters for this batch.",
                    self.trainer.local_rank,
                    type(params).__name__,
                )
                return None
            if len(params) != len(model_params):
                logger.warning(
                    "Rank %s - Adapted params count mismatch in validation "
                    "(expected=%d, got=%d). Using base model parameters for this batch.",
                    self.trainer.local_rank,
                    len(model_params),
                    len(params),
                )
                return None

            backups = {}
            for idx in self.inner_params_idx:
                if idx >= len(model_params):
                    logger.warning(
                        "Rank %s - Inner param index %d out of range for %d model params.",
                        self.trainer.local_rank,
                        idx,
                        len(model_params),
                    )
                    with torch.no_grad():
                        for restore_idx, restore_tensor in backups.items():
                            model_params[restore_idx].copy_(restore_tensor)
                    return None

                src = params[idx]
                dst = model_params[idx]
                if src is None:
                    logger.warning(
                        "Rank %s - Adapted param at index %d is None. "
                        "Using base model parameters for this batch.",
                        self.trainer.local_rank,
                        idx,
                    )
                    with torch.no_grad():
                        for restore_idx, restore_tensor in backups.items():
                            model_params[restore_idx].copy_(restore_tensor)
                    return None
                if src.shape != dst.shape:
                    logger.warning(
                        "Rank %s - Adapted param shape mismatch at index %d "
                        "(expected=%s, got=%s). Using base model parameters for this batch.",
                        self.trainer.local_rank,
                        idx,
                        tuple(dst.shape),
                        tuple(src.shape),
                    )
                    with torch.no_grad():
                        for restore_idx, restore_tensor in backups.items():
                            model_params[restore_idx].copy_(restore_tensor)
                    return None

                with torch.no_grad():
                    backups[idx] = dst.detach().clone()
                    dst.copy_(src.detach())
            return backups

        def restore_inner_params(model, backups):
            if not backups:
                return
            model_params = list(model.parameters())
            with torch.no_grad():
                for idx, tensor in backups.items():
                    if idx < len(model_params) and model_params[idx].shape == tensor.shape:
                        model_params[idx].copy_(tensor)

        self._init_eval_resume_store()
        _, outer_batches, print_out = get_features(
            batch,
            packing=getattr(self.hparams, "packing", "flash"),
            accumulation_steps=self.hparams.inner_grad_accum,
        )
        records = self._build_eval_records(print_out)
        completed_records, pending_records = self._split_records(records)

        pending_work = torch.tensor(
            [1 if len(pending_records) > 0 else 0],
            device=self.device,
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.all_reduce(pending_work, op=dist.ReduceOp.SUM)

        if self.eval_resume and pending_work.item() == 0:
            output_dict = {
                "loss": torch.tensor(0.0),
                "inner_loss": torch.tensor(0.0),
                "outer_loss": torch.tensor(0.0),
                "records": records,
                "skip_loss": True,
            }
            logger.info(
                "Validation sample already complete (guid=%s). Skipping inference.",
                str(records[0]["guid"]) if records else "unknown",
            )
            self.validation_step_outputs.append(output_dict)
            return output_dict

        torch.set_grad_enabled(True)
        output_dict = self.step(batch, return_parameters=True)
        assert isinstance(output_dict, dict)
        output_dict["inner_loss"] = output_dict["inner_loss"].cpu()
        output_dict["outer_loss"] = output_dict["outer_loss"].cpu()
        output_dict["loss"] = output_dict["loss"].detach().cpu()
        output_dict["skip_loss"] = bool(output_dict.get("skip_loss", False))
        self.model.zero_grad()

        if output_dict.get("skip_inference", False):
            output_dict["records"] = completed_records
            self.validation_step_outputs.append(output_dict)
            return output_dict

        inner_param_backups = None
        try:
            params = output_dict["params"]
            inner_param_backups = apply_inner_params(self.model, params)
            if inner_param_backups is None:
                output_dict["skip_loss"] = True

            with torch.no_grad():
                self.model.eval()
                if not output_dict.get("skip_loss", False):
                    outer_loss = self.model(outer_batches)["loss"]
                    output_dict["outer_loss"] = outer_loss.detach().cpu()
                else:
                    output_dict["outer_loss"] = torch.tensor(0.0)

                if len(pending_records) > 0:
                    self.validation_inference(pending_records, self.model)
                    for record in pending_records:
                        record["complete"] = record.get("output") is not None
                    self._append_completed_eval_records(pending_records)
                elif len(completed_records) > 0:
                    logger.info(
                        "All records were restored from resume checkpoint for guid=%s.",
                        str(records[0]["guid"]) if records else "unknown",
                    )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not self._is_cuda_oom_error(exc):
                raise
            logger.warning(
                "Rank %s - CUDA OOM during validation inference/finalization. "
                "Skipping local pending records for guid=%s. Error: %s",
                self.trainer.local_rank,
                str(records[0]["guid"]) if records else "unknown",
                str(exc).splitlines()[0],
            )
            self._cleanup_after_batch_skip()
            output_dict["skip_loss"] = True
            output_dict["skip_inference"] = True
            output_dict["oom_skip"] = True
            output_dict["records"] = completed_records
            output_dict.pop("params", None)
            self.validation_step_outputs.append(output_dict)
            return output_dict
        finally:
            restore_inner_params(self.model, inner_param_backups)
            self.model.train()

        torch.cuda.empty_cache()
        output_dict["records"] = records

        del inner_param_backups
        del output_dict["params"]

        self.validation_step_outputs.append(output_dict)

        return output_dict

    def validation_inference(self, records, model):
        generator = self._get_inference_generator()
        do_sample = bool(getattr(self, "eval_do_sample", False))
        temperature = getattr(self, "eval_temperature", None) if do_sample else None
        top_p = getattr(self, "eval_top_p", None) if do_sample else None
        pad_token_id = getattr(
            getattr(model, "tokenizer", None), "pad_token_id", None
        )
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            generator.generate(
                records,
                pad_token_id=pad_token_id,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                max_new_tokens=self._inference_max_new_tokens,
                use_cache=True,
            )

            if self.local_rank == 0:
                logger.info(f"<guid>\n{records[0]['guid']}\n</guid>")
                logger.info(f"<prompt>\n{records[0]['prompt']}\n</prompt>")
                logger.info(f"<answer>\n{records[0].get('gold_answer')}\n</answer>")
                logger.info(f"<completion>\n{records[0]['output']}\n</completion>")

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your own
        implementation if you need to.

        Args:
            loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
                holds the normalized value (scaled by 1 / accumulation steps).
        """
        if self.hparams.do_train:
            if self.verbose_inner:
                print(f"Backward propagation starts:")
            if self.batch_skip.item() > 0:
                self._cleanup_after_batch_skip()
            elif getattr(self, '_manual_backward_done', False):
                self._manual_backward_done = False
                return
            else:
                local_oom = False
                oom_message = ""
                try:
                    super().backward(loss, *args, **kwargs)
                except torch.cuda.OutOfMemoryError as oom:
                    local_oom = True
                    oom_message = str(oom).splitlines()[0]
                except RuntimeError as exc:
                    # Some OOMs surface as RuntimeError in checkpoint recomputation.
                    msg = str(exc).lower()
                    if "out of memory" in msg and "cuda" in msg:
                        local_oom = True
                        oom_message = str(exc).splitlines()[0]
                    else:
                        raise

                oom_flag = torch.tensor(
                    [1 if local_oom else 0],
                    device=self.device,
                    dtype=torch.int32,
                )
                if dist.is_initialized():
                    dist.all_reduce(oom_flag, op=dist.ReduceOp.SUM)

                if oom_flag.item() > 0:
                    if local_oom:
                        logger.warning(
                            f"Rank {self.trainer.local_rank} - CUDA OOM during backward. "
                            f"Error: {oom_message}"
                        )
                    # Keep all ranks in sync by dropping partial grads and skipping this batch.
                    self.batch_skip[0] = 1
                    self._cleanup_after_batch_skip()
                    return
        else:
            pass
