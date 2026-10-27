"""Score a math response against a gold answer.

The grading itself (MathArena checker plus the source policies in
opt_math.grading.patches) runs in a worker process; this module extracts
the final answer text from a response and builds the result record.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_FRAC_CMD_RE = re.compile(r"\\(?:dfrac|tfrac|frac)")
SIMPLE_MATH_ATOM_RE = re.compile(r"(?:\d+(?:\.\d+)?|[A-Za-z]+|\\[A-Za-z]+)")
_LATEX_LEFT_RIGHT_RE = re.compile(r"\\left(?![a-zA-Z])|\\right(?![a-zA-Z])")
_LATEX_TEXT_COMMAND_RE = re.compile(r"\\text(?:bf|it|rm|sf)?\s*\{([^}]*)\}")
_LATEX_SINGLE_CHAR_BRACE_RE = re.compile(r"([_^])\{(\w)\}")
_LATEX_SPACING_RE = re.compile(r"\\(?:quad|qquad|[;,!:])")
_LATEX_DEGREE_RE = re.compile(r"\^\{?\\circ\}?")
_THOUSANDS_COMMA_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_SCIENTIFIC_NOTATION_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*\\times\s*10\^\{([^}]+)\}"
)
_LATEX_COMPARISON_LONG: dict[str, str] = {
    r"\geq": r"\ge",
    r"\leq": r"\le",
    r"\neq": r"\ne",
}
_BOOLEAN_NORMALISATION: dict[str, str] = {
    "true": "True",
    "yes": "True",
    "false": "False",
    "no": "False",
}


def extract_non_thinking_text(response: str) -> str:
    text = (response or "").strip()
    if not text:
        return ""
    if re.search(r"</think>", text, flags=re.IGNORECASE):
        tail = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
        if tail:
            return tail
    stripped = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if stripped:
        return stripped
    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()


def _is_simple_math_atom(text: str) -> bool:
    return bool(text) and SIMPLE_MATH_ATOM_RE.fullmatch(text) is not None


def _format_fraction_operand(text: str, *, is_denominator: bool) -> str:
    normalized = "".join(_canonicalize_fraction_forms(text).split())
    if not normalized or _is_simple_math_atom(normalized):
        return normalized
    if is_denominator or any(operator in normalized for operator in "+-*/^"):
        return f"({normalized})"
    return normalized


def _extract_brace_group(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return None


def _canonicalize_fraction_forms(text: str) -> str:
    result = text
    safety = 50
    while safety > 0:
        safety -= 1
        matched = _FRAC_CMD_RE.search(result)
        if matched is None:
            break
        pos = matched.end()
        while pos < len(result) and result[pos] in " \t":
            pos += 1
        numerator_result = _extract_brace_group(result, pos)
        if numerator_result is None:
            break
        numerator_content, pos = numerator_result
        while pos < len(result) and result[pos] in " \t":
            pos += 1
        denominator_result = _extract_brace_group(result, pos)
        if denominator_result is None:
            break
        denominator_content, end_pos = denominator_result
        numerator = _format_fraction_operand(numerator_content.strip(), is_denominator=False)
        denominator = _format_fraction_operand(denominator_content.strip(), is_denominator=True)
        result = result[: matched.start()] + f"{numerator}/{denominator}" + result[end_pos:]
    return result


def _normalize_scientific_notation(text: str) -> str:
    def _to_e_notation(matched: re.Match[str]) -> str:
        coefficient = matched.group(1)
        exponent = matched.group(2).strip().lstrip("+")
        return f"{coefficient}e{exponent}"

    return _SCIENTIFIC_NOTATION_RE.sub(_to_e_notation, text)


def _strip_trailing_decimal_zeros(text: str) -> str:
    return re.sub(
        r"(?<![a-zA-Z])(\d+\.\d*?)0+(?=\b|[^0-9]|$)",
        lambda matched: matched.group(1).rstrip("."),
        text,
    )


# The normalizer below only fills the diagnostic *_normalized fields; it never decides the verdict.
def _normalize_match_text(value: Any) -> str:
    text = str(value)
    text = _LATEX_SINGLE_CHAR_BRACE_RE.sub(r"\1\2", text)
    text = _canonicalize_fraction_forms(text)
    text = _normalize_scientific_notation(text)
    text = text.replace("\\$", "")
    text = text.replace("$", "")
    text = text.replace("\\%", "%")
    text = _LATEX_LEFT_RIGHT_RE.sub("", text)
    text = _LATEX_TEXT_COMMAND_RE.sub(r"\1", text)
    text = _LATEX_SPACING_RE.sub("", text)
    text = _LATEX_DEGREE_RE.sub("", text)
    text = text.replace("°", "")
    for long_form, short_form in _LATEX_COMPARISON_LONG.items():
        text = text.replace(long_form, short_form)
    text = text.rstrip(".")
    text = text.replace("\\ ", " ")
    text = "".join(text.split())
    if _THOUSANDS_COMMA_RE.fullmatch(text):
        text = text.replace(",", "")
    text = _strip_trailing_decimal_zeros(text)
    text = re.sub(r"([eE])([+-]?)0+(\d)", r"\1\2\3", text)
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    canonical_bool = _BOOLEAN_NORMALISATION.get(text.lower())
    if canonical_bool is not None:
        text = canonical_bool
    if re.fullmatch(r"\([A-Za-z]\)", text):
        text = text[1:-1].lower()
    elif re.fullmatch(r"[A-Za-z]", text):
        text = text.lower()
    elif re.fullmatch(r"[A-Ea-e]\..+", text):
        text = text[0].lower()
    return text


@dataclass
class MathMatchResult:
    is_match: bool
    match_type: str
    extracted_answer_raw: str
    extracted_answer_normalized: str
    gold_answer_raw: str
    gold_answer_normalized: str
    matched_pred: Optional[str]
    matched_gold: Optional[str]
    pred_candidates: List[str]
    gold_candidates: List[str]

    @property
    def as_answer_pair(self) -> Dict[str, Any]:
        return {
            "extracted_answer_raw": self.extracted_answer_raw,
            "extracted_answer_normalized": self.extracted_answer_normalized,
            "gold_answer_raw": self.gold_answer_raw,
            "gold_answer_normalized": self.gold_answer_normalized,
            "matched_pred": self.matched_pred,
            "matched_gold": self.matched_gold,
            "match_type": self.match_type,
            "is_match": self.is_match,
            "pred_candidates": self.pred_candidates,
            "gold_candidates": self.gold_candidates,
        }


def _matharena_grade(non_thinking_text: str, gold: str, source: Optional[str] = None) -> tuple[bool, str, str]:
    """Grade in the worker process.

    Returns (is_correct, extracted_answer, status) with status "ok", "timeout"
    or "error". Parser errors and timeouts count as wrong; a LaTeX backend that
    cannot load raises.
    """
    from .pool import grade_in_worker

    return grade_in_worker(non_thinking_text, gold, source)


def match_math_response(
    prediction: str,
    gold_answer: str,
    source: Optional[str] = None,
) -> MathMatchResult:
    """Score a math response with the MathArena checker and its source policies.

    Only the last \\boxed answer of the non-thinking part of the response is
    compared with the gold answer; there is no last-integer fallback.
    """
    prediction = prediction or ""
    gold_answer = str(gold_answer if gold_answer is not None else "")
    # Single-letter golds (A-E, e.g. GPQA) use letter matching, not the MathArena grader.
    from .choice import match_letter_choice

    letter_fields = match_letter_choice(prediction, gold_answer)
    if letter_fields is not None:
        return MathMatchResult(**letter_fields)
    non_thinking = extract_non_thinking_text(prediction)
    is_match, extracted, status = _matharena_grade(non_thinking, gold_answer, source)
    gold_normalized = _normalize_match_text(gold_answer)
    extracted_normalized = _normalize_match_text(extracted) if extracted else ""
    return MathMatchResult(
        is_match=is_match,
        match_type="matharena" if is_match else status if status in ("timeout", "error") else "none",
        extracted_answer_raw=extracted,
        extracted_answer_normalized=extracted_normalized,
        gold_answer_raw=gold_answer,
        gold_answer_normalized=gold_normalized,
        matched_pred=extracted if is_match else None,
        matched_gold=gold_answer if is_match else None,
        pred_candidates=[extracted] if extracted else [],
        gold_candidates=[gold_answer] if gold_answer.strip() else [],
    )
