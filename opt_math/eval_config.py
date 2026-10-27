"""Evaluation config.

``to_dict()`` key order feeds ``config.json`` in the run directory. The
config is flat: every key of ``configs/eval.yaml`` maps to one field.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _to_string_list(value: Any, default: Optional[List[str]] = None) -> List[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        parsed = [t.strip() for t in value.split(",") if t.strip()]
        return parsed if parsed else list(default or [])
    if isinstance(value, (list, tuple)):
        parsed = [str(item).strip() for item in value if str(item).strip()]
        return parsed if parsed else list(default or [])
    raise ValueError("Expected a list or comma-separated string")


def _pick(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    value = cfg.get(key)
    if value is not None:
        return value
    return default


@dataclass
class EvalConfig:
    # --- Cases: subset x model x metric ---
    eval_subsets: List[str]
    eval_models: List[str]
    eval_metrics: List[str]

    # --- Trajectory source (filters rows by their `model` column) ---
    source_model: str

    # --- Checkpoint (Lightning .ckpt path) ---
    checkpoint: str

    # --- Base model and LoRA architecture (must match the training run) ---
    model_name_or_path: str
    peft_lora: bool
    lora_rank: int
    lora_alpha: int
    use_rslora: bool
    lora_target_modules: List[str]
    mem_depth: int
    inner_start: int
    inner_end: int
    attn_implementation: str

    # --- Inner loop (test-time training) ---
    inner_loop_type: str
    outer_loss_type: str
    n_inner_iter: int
    inner_lr: float
    inner_grad_accum: int
    unroll_start: int
    inner_opt: str
    inner_beta1: float
    inner_beta2: float
    inner_eps: float
    weight_decay: float
    c_kl: float
    outer_reg_lambda: float

    # --- Weighting ---
    weighted_loss: bool
    weight_normalize: bool
    weight_non_linearity: str
    weighting_model: str

    # --- Generation ---
    max_new_tokens: int
    pass1_temperature: float
    pass1_num_samples: int
    passk_temperature: float
    passk_num_samples: int
    passk_top_p: float

    # --- Dataset ---
    eval_repo_prefix: str
    eval_revision: Optional[str]
    num_thinking: int
    conditioned: bool
    strip_generated_boxed_answers: bool
    max_context_length: int
    packing: bool
    max_eval_data: int

    # --- Output ---
    output_dir: str
    run_name: str
    run_dir: str

    # --- Hardware ---
    accelerator: str
    n_gpu: int
    bf16: bool

    # --- Misc ---
    resume: bool
    seed: int
    stages: List[str] = field(default_factory=lambda: ["generate", "score"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def num_samples_for_metric(self, metric: str) -> int:
        if metric == "pass@1":
            return self.pass1_num_samples
        return self.passk_num_samples

    def temperature_for_metric(self, metric: str) -> float:
        if metric == "pass@1":
            return self.pass1_temperature
        return self.passk_temperature

    def top_p_for_metric(self, metric: str) -> float:
        if metric == "pass@1":
            return 1.0
        return self.passk_top_p

    def is_greedy(self, metric: str) -> bool:
        return metric == "pass@1"

    def k_value(self, metric: str) -> int:
        """Extract k from a ``pass@k`` string."""
        if not metric.startswith("pass@"):
            raise ValueError(f"Invalid metric format: {metric!r}. Expected 'pass@<k>'.")
        return int(metric.split("@")[1])

    @property
    def gen_dir(self) -> str:
        return os.path.join(self.run_dir, "outputs")

    @property
    def eval_dir(self) -> str:
        return os.path.join(self.run_dir, "evals")


def build_eval_config(args: "DictConfig") -> EvalConfig:
    """Build an :class:`EvalConfig` from the flat hydra config."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(args, resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("Hydra config must resolve to a mapping")

    output_dir = os.path.abspath(str(_pick(cfg, "output_dir", "./outputs")))
    run_name = str(_pick(cfg, "run_name", "eval_run"))
    run_dir = os.path.join(output_dir, run_name)

    stages = _to_string_list(_pick(cfg, "stages", ["generate", "score"]))
    for stage in stages:
        if stage not in {"generate", "score"}:
            raise ValueError(f"Unknown stage {stage!r}; expected 'generate' and/or 'score'.")

    return EvalConfig(
        eval_subsets=_to_string_list(_pick(cfg, "eval_subsets", ["all"])),
        eval_models=_to_string_list(_pick(cfg, "eval_models", [])),
        eval_metrics=_to_string_list(_pick(cfg, "eval_metrics", ["pass@1"])),
        source_model=str(_pick(cfg, "source_model", "mixed-low-quality")),

        checkpoint=str(_pick(cfg, "checkpoint", "")),

        model_name_or_path=str(_pick(cfg, "model_name_or_path", "Qwen/Qwen3-4B-Instruct-2507")),
        peft_lora=bool(_pick(cfg, "peft_lora", True)),
        lora_rank=int(_pick(cfg, "lora_rank", 256)),
        lora_alpha=int(_pick(cfg, "lora_alpha", 128)),
        use_rslora=bool(_pick(cfg, "use_rslora", True)),
        lora_target_modules=_to_string_list(_pick(cfg, "lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ])),
        mem_depth=int(_pick(cfg, "mem_depth", 4)),
        inner_start=int(_pick(cfg, "inner_start", 32)),
        inner_end=int(_pick(cfg, "inner_end", 36)),
        attn_implementation=str(_pick(cfg, "attn_implementation", "flash_attention_2")),

        inner_loop_type=str(_pick(cfg, "inner_loop_type", "rgu")),
        outer_loss_type=str(_pick(cfg, "outer_loss_type", "ce")),
        n_inner_iter=int(_pick(cfg, "n_inner_iter", 4)),
        inner_lr=float(_pick(cfg, "inner_lr", 1e-4)),
        inner_grad_accum=int(_pick(cfg, "inner_grad_accum", 1)),
        unroll_start=int(_pick(cfg, "unroll_start", 2)),
        inner_opt=str(_pick(cfg, "inner_opt", "adamw")),
        inner_beta1=float(_pick(cfg, "inner_beta1", 0.9)),
        inner_beta2=float(_pick(cfg, "inner_beta2", 0.95)),
        inner_eps=float(_pick(cfg, "inner_eps", 1e-10)),
        weight_decay=float(_pick(cfg, "weight_decay", 0.01)),
        c_kl=float(_pick(cfg, "c_kl", 0.1)),
        outer_reg_lambda=float(_pick(cfg, "outer_reg_lambda", 0.0)),

        weighted_loss=bool(_pick(cfg, "weighted_loss", True)),
        weight_normalize=bool(_pick(cfg, "weight_normalize", False)),
        weight_non_linearity=str(_pick(cfg, "weight_non_linearity", "softplus")),
        weighting_model=str(_pick(cfg, "weighting_model", "linear")),

        max_new_tokens=int(_pick(cfg, "max_new_tokens", 16384)),
        pass1_temperature=float(_pick(cfg, "pass1_temperature", 0.0)),
        pass1_num_samples=int(_pick(cfg, "pass1_num_samples", 1)),
        passk_temperature=float(_pick(cfg, "passk_temperature", 0.7)),
        passk_num_samples=int(_pick(cfg, "passk_num_samples", 8)),
        passk_top_p=float(_pick(cfg, "passk_top_p", 0.95)),

        eval_repo_prefix=str(_pick(cfg, "eval_repo_prefix", "<org>/math-eval-sources")),
        eval_revision=cfg.get("eval_revision"),
        num_thinking=int(_pick(cfg, "num_thinking", 8)),
        conditioned=bool(_pick(cfg, "conditioned", True)),
        strip_generated_boxed_answers=bool(_pick(cfg, "strip_generated_boxed_answers", True)),
        max_context_length=int(_pick(cfg, "max_context_length", 20480)),
        packing=bool(_pick(cfg, "packing", True)),
        max_eval_data=int(_pick(cfg, "max_eval_data", 10000)),

        output_dir=output_dir,
        run_name=run_name,
        run_dir=run_dir,

        accelerator=str(_pick(cfg, "accelerator", "gpu")),
        n_gpu=int(_pick(cfg, "n_gpu", 1)),
        bf16=bool(_pick(cfg, "bf16", True)),

        resume=bool(_pick(cfg, "resume", True)),
        seed=int(_pick(cfg, "seed", 42)),
        stages=stages,
    )
