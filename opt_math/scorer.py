"""Scoring of math generations: answer matching plus pass@k aggregation."""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from opt_math.grading.match import extract_non_thinking_text, match_math_response

LOGGER = logging.getLogger("opt_math.scorer")


@dataclass(frozen=True)
class EvalCase:
    """One (subset, eval_model, metric) evaluation case."""

    subset: str
    eval_model: str
    metric: str
    config: Any  # EvalConfig


def _pass_at_k_one(c: int, n: int, k: int) -> float:
    """Unbiased estimator of pass@k for a single problem.

    ``1 - C(n-c, k) / C(n, k)`` where *c* = correct, *n* = total samples.
    """
    if n <= 0 or c <= 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def compute_pass_at_k(
    completed_map: Dict[str, Dict[int, Dict]],
    k: int,
    num_samples: int,
) -> Dict[str, Any]:
    """Compute pass@k metrics over completed generations.

    Args:
        completed_map: ``{question_id: {sample_idx: row_dict}}``.
        k: The k in pass@k.
        num_samples: Expected number of samples per question.

    Returns:
        Metrics dict with ``pass@k``, ``num_questions``, ``num_samples``, etc.
    """
    if not completed_map:
        return {
            f"pass@{k}": 0.0,
            "num_questions": 0,
            "num_samples": num_samples,
        }

    total_pass = 0.0
    total_first_gen_correct = 0
    num_questions = len(completed_map)

    for qid, sample_map in completed_map.items():
        c = sum(1 for row in sample_map.values() if bool(row.get("is_match", False)))
        n = len(sample_map)
        total_pass += _pass_at_k_one(c=c, n=n, k=k)
        if 0 in sample_map and bool(sample_map[0].get("is_match", False)):
            total_first_gen_correct += 1

    return {
        f"pass@{k}": total_pass / num_questions,
        "first_generation_accuracy": total_first_gen_correct / num_questions,
        "num_questions": num_questions,
        "num_samples": num_samples,
    }


def compute_pass_at_k_by_subset(
    completed_map: Dict[str, Dict[int, Dict]],
    k: int,
    num_samples: int,
) -> Dict[str, Dict[str, Any]]:
    """Compute pass@k grouped by the ``source`` field in generation rows."""
    by_source: Dict[str, Dict[str, Dict[int, Dict]]] = defaultdict(dict)
    for qid, sample_map in completed_map.items():
        first_row = next(iter(sample_map.values()), {})
        source = str(first_row.get("source", "unknown"))
        by_source[source][qid] = sample_map

    results: Dict[str, Dict[str, Any]] = {}
    for source in sorted(by_source.keys()):
        results[source] = compute_pass_at_k(by_source[source], k=k, num_samples=num_samples)
    return results


def record_gold_answer(record: Dict) -> str:
    gold_answer = record.get("gold_answer", None)
    if gold_answer is None or gold_answer == "":
        gold_answer = record.get("answer", "")
    return str(gold_answer or "")


class MathScorer:
    """Scores math generations: the MathArena grader with its source policies
    (letter matching for single-letter golds such as GPQA) and pass@k aggregation."""

    def aggregate(self, completed_map, ctx, num_samples):
        k = ctx.config.k_value(ctx.metric)
        return compute_pass_at_k(completed_map, k=k, num_samples=num_samples)

    def aggregate_by_source(self, completed_map, ctx, num_samples):
        k = ctx.config.k_value(ctx.metric)
        return compute_pass_at_k_by_subset(completed_map, k=k, num_samples=num_samples)

    def primary_metric(self, metrics, ctx):
        k = ctx.config.k_value(ctx.metric)
        return (f"pass@{k}", metrics.get(f"pass@{k}", 0.0))

    def quick_correctness(self, record: Dict) -> Optional[bool]:
        raw_response = record.get("output", "")
        if raw_response is None:
            return False
        return match_math_response(
            str(raw_response), record_gold_answer(record), source=record.get("task", "")
        ).is_match

    def score_records(self, records: List[Dict], ctx: "EvalCase") -> List[Dict]:
        """Score raw generation records from Lightning validation into case rows."""
        config = ctx.config
        subset, eval_model, metric = ctx.subset, ctx.eval_model, ctx.metric
        scored_rows: List[Dict] = []
        num_samples = config.num_samples_for_metric(metric)
        temperature = config.temperature_for_metric(metric)
        top_p = config.top_p_for_metric(metric)
        do_sample = not config.is_greedy(metric)
        total_records = len(records)

        for record_idx, record in enumerate(records, start=1):
            raw_response = record.get("output", "")
            if raw_response is None:
                raw_response = ""

            answer = record_gold_answer(record)
            guid = record.get("guid", "")
            task = record.get("task", subset)
            sample_idx = int(record.get("sample_idx", record.get("query_idx", 0)) or 0)

            non_thinking = extract_non_thinking_text(raw_response)
            match = match_math_response(raw_response, answer, task)
            is_match, match_type = match.is_match, match.match_type
            pred_candidates, gold_candidates = match.pred_candidates, match.gold_candidates

            if record_idx == 1 or record_idx % 50 == 0 or record_idx == total_records:
                LOGGER.info(
                    "Scored %d/%d records for %s/%s/%s (guid=%s, match=%s:%s)",
                    record_idx,
                    total_records,
                    subset,
                    eval_model,
                    metric,
                    guid,
                    bool(is_match),
                    match_type,
                )

            scored_rows.append({
                "question_id": guid,
                "source": task,
                "subset": subset,
                "eval_model": eval_model,
                "source_model": config.source_model,
                "eval_metric": metric,
                "sample_idx": sample_idx,
                "generation_idx": sample_idx,
                "query_idx": record.get("query_idx", 0),
                "record_uid": record.get("record_uid", ""),
                "num_thinking": config.num_thinking,
                "num_thinking_trajectories": config.num_thinking,
                "inference_backend": "hf",
                "checkpoint": config.checkpoint,
                "generation_num_samples": num_samples,
                "generation_temperature": temperature,
                "generation_top_p": top_p,
                "generation_do_sample": do_sample,
                "rank": record.get("rank"),
                "world_size": record.get("world_size"),
                "prompt": record.get("prompt", ""),
                "gold_answer_raw": answer,
                "gold_answer_normalized": match.gold_answer_normalized,
                "gold_solution_candidates": gold_candidates,
                "model_response_raw": raw_response,
                "model_response_non_thinking": non_thinking,
                "model_solution_extracted": pred_candidates[0] if pred_candidates else "",
                "model_solution_candidates": pred_candidates,
                "extracted_answer_raw": match.extracted_answer_raw,
                "extracted_answer_normalized": match.extracted_answer_normalized,
                "is_match": bool(is_match),
                "match_type": match_type,
                "matched_model_candidate": match.matched_pred,
                "matched_gold_candidate": match.matched_gold,
            })
        return scored_rows

    def answer_pair_row(self, row: Dict) -> Dict:
        return {
            "question_id": row["question_id"],
            "subset": row["subset"],
            "eval_model": row["eval_model"],
            "source_model": row["source_model"],
            "eval_metric": row["eval_metric"],
            "sample_idx": row["sample_idx"],
            "match_type": row["match_type"],
            "is_match": row["is_match"],
            "extracted_answer_raw": row.get("extracted_answer_raw", ""),
            "extracted_answer_normalized": row.get("extracted_answer_normalized", ""),
            "gold_answer_raw": row.get("gold_answer_raw", ""),
            "gold_answer_normalized": row.get("gold_answer_normalized", ""),
            "matched_model_candidate": row.get("matched_model_candidate"),
            "matched_gold_candidate": row.get("matched_gold_candidate"),
        }
