"""Per-generation result record and the validation metrics."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from opt_math.grading.match import MathMatchResult, match_math_response

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """One generation with its prompt, extracted answer, gold answer and metrics."""
    guid: str
    prompt: str
    response: str
    answer: str
    gold_answer: Any
    task: str
    metrics: Dict
    metadata: Dict

    @classmethod
    def from_output(cls, output: Dict):
        """Loads from raw outputs

        :param outputs: the outputs produced by the model
        :rtype: GenerationOutput
        """
        guid = output["guid"]
        prompt = output["prompt"]
        response = output["output"]
        answer = output["answer"]
        gold_answer = output.get("gold_answer")
        task = output["task"]
        metrics = {}
        metadata = {}

        return cls(
            guid=guid,
            prompt=prompt,
            response=response,
            answer=answer,
            gold_answer=gold_answer,
            task=task,
            metrics=metrics,
            metadata=metadata,
        )

    @property
    def answer_pair(self):
        pair = {
            "guid": self.guid,
            "task": self.task,
            "prompt": self.prompt,
            "response": self.response,
            "gold_answer_raw": self.gold_answer if self.gold_answer is not None else self.answer,
            "em": self.metrics.get("em"),
            "f1": self.metrics.get("f1"),
        }
        pair.update(self.metadata.get("answer_pair", {}))
        return pair

    def compute_metrics(self):
        response = self.response or ""
        eval_answer = self.gold_answer
        if isinstance(eval_answer, list):
            answers = [str(item) for item in eval_answer if str(item).strip()]
        elif eval_answer is not None and str(eval_answer).strip():
            answers = [str(eval_answer)]
        else:
            answers = []
        if not answers:
            if isinstance(self.answer, list):
                answers = [str(item) for item in self.answer]
            else:
                answers = [str(self.answer)]
        best_result: Optional[MathMatchResult] = None
        answers = answers or [""]
        for answer in answers:
            result = match_math_response(response, answer)
            if best_result is None:
                best_result = result
            if result.is_match:
                best_result = result
                break

        em = int(best_result.is_match)
        self.metrics["em"] = em
        self.metrics["f1"] = em
        self.metadata["answer_pair"] = best_result.as_answer_pair


def evaluate(records):
    generations = [GenerationOutput.from_output(record) for record in records]
    for generation in generations:
        generation.compute_metrics()
    if not generations:
        # Zero records (e.g. every batch of a validation epoch was skipped)
        # score 0.0 instead of crashing the run. The warning tells this case
        # apart from a real all-wrong 0.0.
        logger.warning(
            "evaluate() received 0 scoreable generations; returning 0.0 "
            "metrics. If unexpected, check the validation generator."
        )
        metrics = {"accuracy": 0.0, "f1": 0.0}
        return metrics, generations
    accuracy = sum([gen.metrics["em"] for gen in generations]) / len(generations)
    f1 = sum([gen.metrics["f1"] for gen in generations]) / len(generations)
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
    }
    return metrics, generations
