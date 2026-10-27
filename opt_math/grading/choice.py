"""Letter-choice scoring (multiple-choice golds A-E, e.g. GPQA).

A gold that is a single letter A-E is matched against the answer the response
commits to: the last boxed or "Answer:" text if there is one, otherwise any
answer-like span of the response. Unboxed answers ("The answer is B",
"Answer: (B)") count. Every other gold goes to the MathArena grader.
"""

import math
import re
import string
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .match import _normalize_match_text, extract_non_thinking_text  # noqa: F401

_NUMERIC_CONSTANTS: dict[str, float] = {"e": math.e, "pi": math.pi, "\\pi": math.pi}


_NUMERIC_REL_TOL = 5e-3


_NUMERIC_ABS_TOL = 1e-9


_UNIT_SUFFIX_RE = re.compile(
    r"^(.+?)\s*(?:eV|MeV|keV|GeV|Hz|kHz|MHz|GHz|THz|m/s|km/s|m|cm|mm|km|kg|"
    r"g|mg|s|ms|ns|ps|J|kJ|W|kW|MW|N|Pa|kPa|MPa|V|mV|kV|A|mA|ohm|Ohm|"
    r"\\Omega|F|H|T|C|K|mol|L|mL|diopters?|D)$"
)


def normalize_text(text):
    """
    Removing articles and punctuation, and
    standardizing whitespace are all typical
    text processing steps.

    :param text: text to normalize
    """
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def _strip_math_delimiters(text: str) -> str:
    if text is None:
        return ""
    value = text.strip()
    changed = True
    while changed:
        changed = False
        for start, end in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
            if value.startswith(start) and value.endswith(end) and len(value) >= len(start) + len(end):
                value = value[len(start):-len(end)].strip()
                changed = True
    return value


def _normalize_math_candidate(text: str) -> str:
    if text is None:
        return ""
    return _normalize_match_text(text)


def _normalize_for_substring(text: str) -> str:
    if text is None:
        return ""
    value = str(text).replace("\u2212", "-")
    value = _strip_math_delimiters(value)
    value = value.replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\\(?:mathrm|mathbf|text)\{~?([^}]*)\}", r"\1", value)
    value = re.sub(r"\s+", " ", value.lower()).strip()
    return value


def _is_numeric_like(text: str) -> bool:
    if not text:
        return False
    return re.fullmatch(r"[-+]?\d*\.?\d+(?:/\d+)?%?", text.strip()) is not None


def _split_top_level_commas(expr: str) -> List[str]:
    if not expr:
        return []
    items = []
    depth = 0
    start = 0
    for idx, char in enumerate(expr):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            item = expr[start:idx].strip()
            if item:
                items.append(item)
            start = idx + 1
    last = expr[start:].strip()
    if last:
        items.append(last)
    return items


def _expand_pm(expr: str) -> List[str]:
    if "\\pm" in expr:
        return [expr.replace("\\pm", "+"), expr.replace("\\pm", "-")]
    return [expr]


def _extract_boxed_contents(text: str) -> List[str]:
    if not text:
        return []
    results = []
    for command in ("\\boxed{", "\\fbox{"):
        search_pos = 0
        while True:
            start = text.find(command, search_pos)
            if start == -1:
                break
            brace_start = start + len(command)
            depth = 1
            idx = brace_start
            while idx < len(text) and depth > 0:
                if text[idx] == "{":
                    depth += 1
                elif text[idx] == "}":
                    depth -= 1
                idx += 1
            if depth == 0:
                results.append(text[brace_start:idx - 1].strip())
                search_pos = idx
            else:
                search_pos = brace_start
    return results


def _extract_tagged_answers(text: str) -> List[str]:
    if not text:
        return []
    candidates = []
    for tag in ("answer", "output", "final", "result"):
        pattern = rf"<{tag}>(.*?)</{tag}>"
        for match in re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            candidates.append(match.strip())
    return candidates


def _extract_gsm8k_answers(text: str) -> List[str]:
    if not text:
        return []
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)", text)
    return [matches[-1]] if matches else []


def _extract_answer_lines(text: str) -> List[str]:
    if not text:
        return []
    candidates = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        patterns = [
            r"^(?:final\s+answer|answer)\s*[:\-]\s*(.+)$",
            r"^(?:final\s+answer|answer)\s*=\s*(.+)$",
            r"^(?:answer|ans|result|output)\s*[:\-]\s*(.+)$",
            r"^answer\s*is\s*[:\-]?\s*(.+)$",
            r"^the answer is\s*[:\-]?\s*(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                candidates.append(match.group(1).strip())
    return candidates


def _extract_latex_spans(text: str) -> List[str]:
    if not text:
        return []
    candidates = []
    for pattern in (r"\$(.+?)\$", r"\\\((.+?)\\\)", r"\\\[(.+?)\\\]"):
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            candidates.append(matches[-1].strip())
    return candidates


def _extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?%?", text)
    return matches[-1] if matches else None


def _split_additive(text: str) -> list[str] | None:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth -= 1
        elif ch in "+-" and depth == 0 and i > 0:
            if text[i - 1] in "eE" and i >= 2 and text[i - 2].isdigit():
                continue
            parts.append(text[start:i])
            start = i
    parts.append(text[start:])
    parts = [part for part in parts if part]
    return parts if len(parts) >= 2 else None


def _try_parse_numeric(text: str) -> float | None:
    if not text:
        return None
    normalized = _normalize_match_text(text)
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError:
        pass
    if normalized in _NUMERIC_CONSTANTS:
        return _NUMERIC_CONSTANTS[normalized]
    if normalized.startswith("+") and len(normalized) > 1:
        return _try_parse_numeric(normalized[1:])
    if normalized.startswith("(") and normalized.endswith(")"):
        depth = 0
        balanced = True
        for i, ch in enumerate(normalized):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and i < len(normalized) - 1:
                balanced = False
                break
        if balanced:
            inner_value = _try_parse_numeric(normalized[1:-1])
            if inner_value is not None:
                return inner_value
    if normalized.endswith("%"):
        try:
            return float(normalized[:-1]) / 100.0
        except ValueError:
            pass
    matched = re.fullmatch(r"\\sqrt\{([^}]+)\}", normalized)
    if matched:
        inner_value = _try_parse_numeric(matched.group(1))
        if inner_value is not None and inner_value >= 0:
            return math.sqrt(inner_value)
    matched = re.fullmatch(r"(-?\d+(?:\.\d+)?)\\pi", normalized)
    if matched:
        return float(matched.group(1)) * math.pi
    matched = re.fullmatch(r"\\pi/(-?\d+(?:\.\d+)?)", normalized)
    if matched:
        return math.pi / float(matched.group(1))
    matched = re.fullmatch(r"e\^\{([^}]+)\}", normalized)
    if matched:
        exponent_value = _try_parse_numeric(matched.group(1))
        if exponent_value is not None:
            return math.exp(exponent_value)
    matched = re.fullmatch(r"e\^(-?\d+(?:\.\d+)?)", normalized)
    if matched:
        return math.exp(float(matched.group(1)))
    matched = re.fullmatch(r"(-?\d+(?:\.\d+)?)e\^\{([^}]+)\}", normalized)
    if matched:
        coefficient = float(matched.group(1))
        exponent_value = _try_parse_numeric(matched.group(2))
        if exponent_value is not None:
            return coefficient * math.exp(exponent_value)
    matched = re.fullmatch(r"\\binom\{(\d+)\}\{(\d+)\}", normalized)
    if matched:
        try:
            return float(math.comb(int(matched.group(1)), int(matched.group(2))))
        except OverflowError:
            return None
    matched = re.fullmatch(r"\\ln\{?([^}]+)\}?", normalized)
    if matched:
        inner_value = _try_parse_numeric(matched.group(1))
        if inner_value is not None and inner_value > 0:
            return math.log(inner_value)
    for log_pattern in (r"\\log_\{(\d+)\}\{?([^}]+)\}?", r"\\log_(\d)(.+)"):
        matched = re.fullmatch(log_pattern, normalized)
        if matched:
            base = int(matched.group(1))
            inner_value = _try_parse_numeric(matched.group(2))
            if inner_value is not None and inner_value > 0 and base > 1:
                return math.log(inner_value) / math.log(base)
    matched = re.fullmatch(r"(-?\d+(?:\.\d+)?)\\ln\{?([^}]+)\}?", normalized)
    if matched:
        coefficient = float(matched.group(1))
        inner_value = _try_parse_numeric(matched.group(2))
        if inner_value is not None and inner_value > 0:
            return coefficient * math.log(inner_value)
    for log_pattern in (
        r"(-?\d+(?:\.\d+)?)\\log_\{(\d+)\}\{?([^}]+)\}?",
        r"(-?\d+(?:\.\d+)?)\\log_(\d)(.+)",
    ):
        matched = re.fullmatch(log_pattern, normalized)
        if matched:
            coefficient = float(matched.group(1))
            base = int(matched.group(2))
            inner_value = _try_parse_numeric(matched.group(3))
            if inner_value is not None and inner_value > 0 and base > 1:
                return coefficient * math.log(inner_value) / math.log(base)
    matched = re.fullmatch(r"\\log_(\d+)\\binom\{(\d+)\}\{(\d+)\}", normalized)
    if matched:
        base = int(matched.group(1))
        if base > 1:
            try:
                return math.log(math.comb(int(matched.group(2)), int(matched.group(3)))) / math.log(base)
            except OverflowError:
                return None
    additive_terms = _split_additive(normalized)
    if additive_terms is not None:
        total = 0.0
        for term in additive_terms:
            value = _try_parse_numeric(term)
            if value is None:
                break
            total += value
        else:
            return total
    matched = re.fullmatch(r"\(?([^()/]+)\)?/\(?([^()/]+)\)?", normalized)
    if matched:
        numerator = _try_parse_numeric(matched.group(1))
        denominator = _try_parse_numeric(matched.group(2))
        if numerator is not None and denominator is not None and denominator != 0:
            return numerator / denominator
    if normalized.startswith("-") and len(normalized) > 1:
        inner_value = _try_parse_numeric(normalized[1:])
        if inner_value is not None:
            return -inner_value
    matched = _UNIT_SUFFIX_RE.match(normalized)
    if matched:
        return _try_parse_numeric(matched.group(1))
    return None


def _values_approx_equal(a: float, b: float) -> bool:
    if a == b:
        return True
    if a == 0 or b == 0:
        return abs(a - b) <= _NUMERIC_ABS_TOL
    return abs(a - b) / max(abs(a), abs(b)) <= _NUMERIC_REL_TOL


def _parse_numeric_value(text: str) -> Optional[Tuple[float, bool]]:
    numeric = _try_parse_numeric(text)
    if numeric is None:
        return None
    return numeric, "%" in str(text)


def _numeric_equiv(pred: str, target: str, tol: float = 1e-6) -> bool:
    pred_value = _parse_numeric_value(pred)
    target_value = _parse_numeric_value(target)
    if not pred_value or not target_value:
        return False
    pred_num, _ = pred_value
    target_num, _ = target_value
    if _values_approx_equal(pred_num, target_num):
        return True
    if math.isclose(pred_num * 100, target_num, rel_tol=tol, abs_tol=tol):
        return True
    if math.isclose(pred_num, target_num * 100, rel_tol=tol, abs_tol=tol):
        return True
    return False


def _math_equiv_simple(pred: str, target: str) -> bool:
    pred_norm = _normalize_math_candidate(pred)
    target_norm = _normalize_math_candidate(target)
    if not pred_norm or not target_norm:
        return False
    if pred_norm == target_norm:
        return True
    return _numeric_equiv(pred_norm, target_norm)


def _math_equiv(pred: str, target: str) -> bool:
    pred_norm = _normalize_math_candidate(pred)
    target_norm = _normalize_math_candidate(target)
    if not pred_norm or not target_norm:
        return False
    if pred_norm == target_norm:
        return True
    pred_items = _split_top_level_commas(pred_norm)
    target_items = _split_top_level_commas(target_norm)
    if len(pred_items) > 1 or len(target_items) > 1:
        if len(pred_items) != len(target_items):
            return False
        remaining = target_items[:]
        for item in pred_items:
            matched = False
            for candidate in list(remaining):
                if _math_equiv_simple(item, candidate):
                    remaining.remove(candidate)
                    matched = True
                    break
            if not matched:
                return False
        return not remaining
    return _math_equiv_simple(pred_norm, target_norm)


def _substring_match_non_numeric(container_norm: str, target_norm: str) -> bool:
    """Treat ``container_norm`` as containing ``target_norm``.

    GPQA-style MC labels normalize to a single letter; naive ``target in container``
    would label almost any English rationale correct because ``a``–``d`` occur inside words.
    """
    if not container_norm or not target_norm:
        return False
    if len(target_norm) == 1 and target_norm.isalpha():
        return (
            re.search(
                rf"(?<![a-z0-9]){re.escape(target_norm)}(?![a-z0-9])",
                container_norm,
            )
            is not None
        )
    return target_norm in container_norm


def _substring_exact_match(response: str, target: str) -> bool:
    response_norm = _normalize_for_substring(response)
    target_norm = _normalize_for_substring(target)
    if not response_norm or not target_norm:
        return False
    if _is_numeric_like(target_norm):
        response_norm = response_norm.replace(",", "")
        target_norm = target_norm.replace(",", "")
        pattern = rf"(?<![\d\.]){re.escape(target_norm)}(?![\d\.])"
        return re.search(pattern, response_norm) is not None
    return _substring_match_non_numeric(response_norm, target_norm)


def _extract_math_candidates(text: str, include_original: bool = False) -> List[str]:
    if text is None:
        return []
    content = str(text)
    candidates: List[str] = []
    candidates.extend(_extract_boxed_contents(content))
    candidates.extend(_extract_tagged_answers(content))
    candidates.extend(_extract_gsm8k_answers(content))
    candidates.extend(_extract_answer_lines(content))
    candidates.extend(_extract_latex_spans(content))

    brace_match = re.search(r"\{([^{}]+)\}\s*$", content.strip())
    if brace_match:
        candidates.append(brace_match.group(1).strip())
    eq_match = re.search(r"=\s*([^\n\.\,\s]+)\s*$", content.strip())
    if eq_match:
        candidates.append(eq_match.group(1).strip())
    answer_is_match = re.search(r"(?i)answer is[:\s]*([^\n\.\,]+)", content)
    if answer_is_match:
        candidates.append(answer_is_match.group(1).strip())

    last_number = _extract_last_number(content)
    if last_number:
        candidates.append(last_number)
    if include_original:
        candidates.append(content)

    expanded: List[str] = []
    for candidate in candidates:
        expanded.extend(_expand_pm(candidate))

    normalized: List[str] = []
    seen = set()
    for candidate in expanded:
        normalized_candidate = _normalize_math_candidate(candidate)
        if normalized_candidate and normalized_candidate not in seen:
            seen.add(normalized_candidate)
            normalized.append(normalized_candidate)
    return normalized


def _build_shared_gold_candidates(answer_raw: str, solution_raw: str = "") -> List[str]:
    candidates: List[str] = []
    if answer_raw:
        candidates.extend(_extract_math_candidates(answer_raw, include_original=True))
    if solution_raw:
        candidates.extend(_extract_math_candidates(solution_raw, include_original=True))
    if not candidates:
        if answer_raw:
            candidates.append(_normalize_math_candidate(answer_raw))
        if solution_raw:
            candidates.append(_normalize_math_candidate(solution_raw))
    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = str(candidate).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _dedupe_non_empty(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for v in values:
        v = (v or "").strip()
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


def _looks_like_standalone_gold_answer(answer_raw: str) -> bool:
    text = (answer_raw or "").strip()
    if not text or len(text) > 120:
        return False
    if "\\boxed" in text:
        return False
    if re.search(r"(?i)\b(answer|solution|therefore|final|correct|ends?\s+with)\b", text):
        return False
    return True


def _is_single_letter_choice(value: str) -> bool:
    value = (value or "").strip()
    return len(value) == 1 and value.lower() in {"a", "b", "c", "d", "e"}


def _extract_explicit_final_answer_text(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(
        r"(?ims)(?:^|\n)\s*(?:\*\*)?\s*(?:final\s+answer|answer)\s*(?:\*\*)?\s*"
        r"(?::|is)?\s*(.+?)(?=\n\s*\n|\Z)",
        text,
    )
    if not matches:
        return None
    return matches[-1].strip()


def _candidate_variants_from_final_text(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    variants = [text]
    option_match = re.match(r"(?is)^(?:option\s*)?\(?([A-E])\)?(?:[.\):]|$)", text)
    if option_match:
        variants.append(option_match.group(1))
    assignment_match = re.match(r"(?is)^[A-Za-z][\w\s{}\\]*\s*=\s*(.+)$", text)
    if assignment_match:
        variants.append(assignment_match.group(1).strip())
    return _dedupe_non_empty(_normalize_math_candidate(item) for item in variants)


def _build_final_pred_candidates(non_thinking_text: str) -> List[str]:
    boxed_items = _extract_boxed_contents(non_thinking_text)
    if boxed_items:
        return _candidate_variants_from_final_text(boxed_items[-1])
    final_answer_text = _extract_explicit_final_answer_text(non_thinking_text)
    if final_answer_text:
        boxed_in_final = _extract_boxed_contents(final_answer_text)
        if boxed_in_final:
            return _candidate_variants_from_final_text(boxed_in_final[-1])
        return _candidate_variants_from_final_text(final_answer_text)
    return []


def _decimal_places(value: str) -> Optional[int]:
    matched = re.search(r"(?<![\w.])[-+]?\d+\.(\d+)(?![\w.])", str(value or ""))
    if not matched:
        return None
    return len(matched.group(1))


def _rounded_numeric_equiv(pred: str, gold: str) -> bool:
    places = _decimal_places(gold)
    if places is None or places <= 0:
        return False
    try:
        pred_value = _try_parse_numeric(pred)
        gold_value = _try_parse_numeric(gold)
    except (OverflowError, ValueError):
        return False
    if pred_value is None or gold_value is None:
        return False
    return round(pred_value, places) == round(gold_value, places)


def _match_candidate_lists(
    pred_candidates: Sequence[str],
    gold_candidates: Sequence[str],
    *,
    allow_substring: bool,
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    for pred in pred_candidates:
        for gold in gold_candidates:
            try:
                if _math_equiv(pred, gold):
                    return True, "math_equiv", pred, gold
            except OverflowError:
                continue
            if _rounded_numeric_equiv(pred, gold):
                return True, "rounded_numeric", pred, gold
    for pred in pred_candidates:
        for gold in gold_candidates:
            if normalize_text(pred) == normalize_text(gold):
                return True, "normalized_exact", pred, gold
    if allow_substring:
        for pred in pred_candidates:
            if not (pred or "").strip():
                continue
            for gold in gold_candidates:
                if _is_single_letter_choice(gold):
                    continue
                if _substring_exact_match(pred, gold):
                    return True, "substring_extracted", pred, gold
    return False, "none", None, None


def build_pred_candidates(non_thinking: str, raw_response: str = "") -> List[str]:
    candidates = _extract_math_candidates(non_thinking, include_original=False)
    if not candidates and raw_response and raw_response != non_thinking:
        candidates = _extract_math_candidates(raw_response, include_original=False)
    return _dedupe_non_empty(candidates)


def match_prediction(
    pred_candidates: List[str],
    gold_candidates: List[str],
    non_thinking_text: str,
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    final_candidates = _build_final_pred_candidates(non_thinking_text)
    if final_candidates:
        return _match_candidate_lists(
            final_candidates,
            gold_candidates,
            allow_substring=False,
        )

    is_match, match_type, matched_pred, matched_gold = _match_candidate_lists(
        pred_candidates,
        gold_candidates,
        allow_substring=True,
    )
    if is_match:
        return is_match, match_type, matched_pred, matched_gold

    return False, "none", None, None


def build_gold_candidates(answer_raw: str, solution_raw: str = "") -> List[str]:
    if answer_raw and not solution_raw and _looks_like_standalone_gold_answer(answer_raw):
        return _dedupe_non_empty([_normalize_math_candidate(answer_raw)])
    return _dedupe_non_empty(_build_shared_gold_candidates(answer_raw, solution_raw))


def match_letter_choice(prediction: str, gold_answer: str) -> Optional[Dict[str, Any]]:
    """Score a response against a single-letter gold; None if the gold is not a letter.

    Returns the fields of a MathMatchResult.
    """
    gold_candidates = build_gold_candidates(gold_answer, "")
    if not any(_is_single_letter_choice(gold) for gold in gold_candidates):
        return None
    non_thinking = extract_non_thinking_text(prediction)
    pred_candidates = build_pred_candidates(non_thinking, prediction)
    is_match, match_type, matched_pred, matched_gold = match_prediction(
        pred_candidates=pred_candidates,
        gold_candidates=gold_candidates,
        non_thinking_text=non_thinking,
    )
    extracted_answer_raw = pred_candidates[0] if pred_candidates else ""
    return dict(
        is_match=bool(is_match),
        match_type=match_type,
        extracted_answer_raw=extracted_answer_raw,
        extracted_answer_normalized=matched_pred or extracted_answer_raw,
        gold_answer_raw=gold_answer,
        gold_answer_normalized=gold_candidates[0] if gold_candidates else str(gold_answer or ""),
        matched_pred=matched_pred,
        matched_gold=matched_gold,
        pred_candidates=pred_candidates,
        gold_candidates=gold_candidates,
    )
