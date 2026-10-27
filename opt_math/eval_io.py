"""Path conventions and atomic I/O for the evaluation pipeline.

Case layout::

    run_dir/
      outputs/{subset}/{model_slug}/{metric}/   <- generations
      evals/{subset}/{model_slug}/{metric}/     <- metrics
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List


def model_slug(model_name: str) -> str:
    """Convert a model name to a filesystem-safe slug.

    ``Qwen/Qwen3-4B-Instruct-2507`` -> ``Qwen__Qwen3-4B-Instruct-2507``
    """
    cleaned = model_name.replace("/", "__")
    cleaned = re.sub(r"[^a-zA-Z0-9_.\-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def case_gen_dir(run_dir: str, subset: str, eval_model: str, metric: str) -> str:
    return os.path.join(run_dir, "outputs", subset, model_slug(eval_model), metric)


def case_eval_dir(run_dir: str, subset: str, eval_model: str, metric: str) -> str:
    return os.path.join(run_dir, "evals", subset, model_slug(eval_model), metric)


def generation_path(run_dir: str, subset: str, eval_model: str, metric: str) -> str:
    return os.path.join(case_gen_dir(run_dir, subset, eval_model, metric), "generations.jsonl")


def answer_pairs_path(run_dir: str, subset: str, eval_model: str, metric: str) -> str:
    return os.path.join(case_gen_dir(run_dir, subset, eval_model, metric), "answer_pairs.jsonl")


def metrics_path(run_dir: str, subset: str, eval_model: str, metric: str) -> str:
    return os.path.join(case_eval_dir(run_dir, subset, eval_model, metric), "metrics.json")


def summary_path(run_dir: str) -> str:
    return os.path.join(run_dir, "evals", "summary.json")


def eval_resume_file_template(config: Any, subset: str, eval_model: str, metric: str) -> str:
    """Per-case resume path template; ``MetaLearningModule`` fills ``{rank}``."""
    return os.path.join(
        config.run_dir,
        "resume_generations",
        subset,
        model_slug(eval_model),
        metric,
        "generations_rank{rank}.jsonl",
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_case_dirs(run_dir: str, subset: str, eval_model: str, metric: str) -> None:
    ensure_dir(case_gen_dir(run_dir, subset, eval_model, metric))
    ensure_dir(case_eval_dir(run_dir, subset, eval_model, metric))


def write_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def write_jsonl(rows: Iterable[Dict], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
