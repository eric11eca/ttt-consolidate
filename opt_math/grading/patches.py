"""Math answer grading: MathArena checker plus family-specific patches.

This module runs inside the grading worker process (opt_math/grading/pool.py),
which has the vendored antlr4 4.11 runtime first on sys.path. It is not meant
to be imported into a process that uses hydra/omegaconf.

Grading rules, all of which fail closed (unparseable or ambiguous means wrong):

* the answer is the last ``\\boxed{...}``; there is no last-integer fallback
* a gold that contains ``\\boxed{...}`` is reduced to the boxed content
* boolean golds, tuple golds and rounded decimal golds have their own patches
* everything else goes to MathArena's ``check_answers`` with strict parsing

See opt_math/grading/third_party/matharena/README.md for the list of deviations.
"""

import re
from typing import Optional

_BOOL_WORDS = {"true": True, "yes": True, "false": False, "no": False}
_BOXED_RE = r"(?:boxed|fbox)\{((?:[^{}]|\{(?1)\})*)\}"
_Result = tuple  # (is_correct: bool, extracted: str)


def _strip_wrappers(text: str) -> str:
    text = str(text).strip()
    text = text.replace("$", "")
    text = re.sub(r"\\[dt]frac", r"\\frac", text)
    for _ in range(3):
        text = re.sub(r"\\(?:text|textbf|mathrm|mathbf|textit|mbox)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\*\*|__", "", text)
    # a leading backslash is a LaTeX command (\left, \frac), never stray punctuation
    return text.strip().rstrip(".,;:!?\\ ").lstrip(".,;:!? ").strip()


def _last_boxed(text: str) -> Optional[str]:
    import regex

    matches = list(regex.finditer(_BOXED_RE, text))
    return matches[-1].group(1) if matches else None


def _split_top_level(s: str, delim: str = ",") -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == delim and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts]


# ---------------------------------------------------------------- normalized exact accept
_WRAPPER_RE = re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf|operatorname|mbox)\s*\{([^{}]*)\}")


def _normalize_exact(text: str) -> str:
    """Layout-insensitive form of an answer, used only to accept literal matches.

    Text wrappers are unwrapped (never deleted). Only a leading sentence
    capital on the first word is folded ("Odd n" == "odd n"); every other letter
    keeps its case (A vs a, N vs n, aB vs ab). Spaces are dropped only next to
    punctuation or operators, so "5 6" never becomes "56".
    """
    s = str(text)
    for _ in range(4):
        s = _WRAPPER_RE.sub(r"\1", s)
    s = s.replace("$", "")
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\q?quad|\\[,;:! ]|~", " ", s)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"([\^_])\{([A-Za-z0-9])\}", r"\1\2", s)
    s = re.sub(r"^\s*([A-Z])(?=[a-z]+\b)", lambda m: m.group(1).lower(), s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?<![A-Za-z0-9]) | (?![A-Za-z0-9])", "", s)
    s = re.sub(r"(?<=\d) (?=[A-Za-z])", "", s)  # "2 n" and "2n" are the same product
    return s.rstrip(".")


def grade_normalized_exact(text: str, gold: str) -> Optional[_Result]:
    """Accept when the last box equals the gold up to LaTeX layout. Never rejects."""
    boxed = _last_boxed(text)
    if boxed is None:
        return None
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", gold.replace("$", "").strip()) and re.search(r",\s", boxed):
        return None  # "12, 345" is two numbers, not 12,345
    norm_gold = _normalize_exact(gold)
    if norm_gold and _normalize_exact(boxed) == norm_gold:
        return True, boxed
    return None


# ---------------------------------------------------------------- parenthesised choice letters
_CHOICE_GOLD_RE = re.compile(r"^\(([A-Ea-e])\)$")


def grade_choice_letter(text: str, gold: str) -> Optional[_Result]:
    """Gold "(d)": the last box must be that single letter, in any common form."""
    m = _CHOICE_GOLD_RE.match(_strip_wrappers(gold))
    if not m:
        return None
    boxed = _last_boxed(text)
    if boxed is None:
        return False, ""
    b = re.fullmatch(r"\(?([A-Za-z])\)?", _strip_wrappers(boxed).strip())
    if not b:
        return False, boxed
    return b.group(1).lower() == m.group(1).lower(), boxed


# ---------------------------------------------------------------- boolean golds
def _as_bool(text: str) -> Optional[bool]:
    return _BOOL_WORDS.get(_strip_wrappers(text).lower())


def grade_boolean(text: str, gold: str) -> Optional[_Result]:
    gold_bool = _as_bool(gold)
    if gold_bool is None:
        return None
    boxed = _last_boxed(text)
    if boxed is None:
        return False, ""
    s = _strip_wrappers(boxed).lower()
    words = re.findall(r"[a-z]+", s)
    opposite = {w for w, v in _BOOL_WORDS.items() if v != gold_bool}
    if any(w in opposite for w in words):
        return False, boxed
    if s in _BOOL_WORDS:
        return _BOOL_WORDS[s] == gold_bool, boxed
    m = re.fullmatch(r"(yes|no|true|false)\s*,\s*([^,]*)", s)
    if m:
        # Qualifiers always reject. Words of the opposite polarity reject too, so
        # "No, ... not ..." agrees with a False gold but "Yes, ... not ..." does not.
        lead_true = _BOOL_WORDS[m.group(1)]
        clause = m.group(2)
        rejected = r"except|unless|but|only|otherwise|if|when|although|though|fails?"
        # Negative words fit a No/False lead only; positive words fit a Yes/True lead only.
        rejected += r"|not|no|never|cannot|impossible|false|none|nothing|neither|nor|\w+n't" if lead_true else r"|yes|true"
        double_negative = re.search(r"\b(not|never|cannot|can't)\s+(\w+\s+)?(true|false|yes|no)\b", clause)
        if not re.search(rf"\b({rejected})\b", clause) and not double_negative:
            return lead_true == gold_bool, boxed
    return False, boxed


# ---------------------------------------------------------------- tuple golds
def _parse_tuples(s: str) -> Optional[list[list[str]]]:
    """Parse '(a,b),(c,d)' (optionally in braces / separated by 'and') into component lists."""
    s = _strip_wrappers(s)
    s = re.sub(r"\\left|\\right|\\q?quad|\\[,;! ]", " ", s)
    s = s.replace("\n", " ")
    s = re.sub(r"\\\{|\\\}", "", s)
    s = s.strip().strip("{}[]").strip()
    s = re.sub(r"\band\b|\\text\{\s*and\s*\}", ",", s)
    items = [p for p in _split_top_level(s) if p]
    tuples = []
    for item in items:
        if not (item.startswith("(") and item.endswith(")")):
            return None
        inner = item[1:-1]
        depth = 0
        for pos, ch in enumerate(item):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and pos < len(item) - 1:
                return None  # the opening paren closes early: not a single tuple
        if depth != 0:
            return None
        comps = _split_top_level(inner)
        if len(comps) < 2 or any(not c for c in comps):
            return None
        tuples.append(comps)
    return tuples or None


def _component_equal(a: str, b: str) -> bool:
    from .third_party.matharena.parser import check_answers, parse_answer

    if a.replace(" ", "") == b.replace(" ", ""):
        return True
    pa, _ = parse_answer(a, list_answer=False, typed_delimiters=True)
    pb, _ = parse_answer(b, list_answer=False, typed_delimiters=True)
    return bool(check_answers(pa, pb))


_TUPLE_LABEL_RE = re.compile(
    r"^\(\s*([A-Za-z][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z][A-Za-z0-9_]*)+)\s*\)\s*=\s*(.+)$", flags=re.S
)
_CONDITION_RE = re.compile(r"\s*,?\s*\b(?:for|where|with|if|when)\b.*$", flags=re.S)


def grade_tuples(text: str, gold: str) -> Optional[_Result]:
    """Unordered comparison of tuple lists.

    A trailing condition ("for n >= 1") is dropped from the gold. The model's
    own trailing condition is dropped too, but only if the gold had one and no
    tuple follows the condition word (otherwise the answer is a case split or
    adds a restriction the gold lacks, and counts as wrong).
    Any other free text in the gold leaves the case to MathArena.
    """
    gold_stripped = _strip_wrappers(gold)
    gold_has_condition = _CONDITION_RE.search(gold_stripped) is not None
    g = _CONDITION_RE.sub("", gold_stripped)
    if not g.lstrip("{[ ").startswith(("(", "\\left(")):
        return None
    gold_tuples = _parse_tuples(g)
    if gold_tuples is None:
        return None
    boxed = _last_boxed(text)
    if boxed is None:
        return False, ""
    pred = _strip_wrappers(boxed)
    if re.search(r"\b(otherwise|else|none|except|unless)\b", pred):
        return False, boxed
    cond = _CONDITION_RE.search(pred)
    if cond:
        if not gold_has_condition or "(" in cond.group(0):
            return False, boxed
        pred = pred[: cond.start()]
    label = _TUPLE_LABEL_RE.match(pred)
    n_labels = 0
    if label:
        n_labels = len([n for n in label.group(1).split(",") if n.strip()])
        pred = label.group(2)
    pred_tuples = _parse_tuples(pred)
    if pred_tuples is None or len(pred_tuples) != len(gold_tuples):
        return False, boxed
    if n_labels and any(len(t) != n_labels for t in pred_tuples):
        return False, boxed
    remaining = list(gold_tuples)
    for pt in pred_tuples:
        for i, gt in enumerate(remaining):
            if len(gt) == len(pt) and all(_component_equal(x, y) for x, y in zip(pt, gt)):
                del remaining[i]
                break
        else:
            return False, boxed
    return not remaining, boxed


# ---------------------------------------------------------------- rounded decimal golds
_DECIMAL_RE = re.compile(r"^-?\d+\.(\d+)$")
_PLAIN_NUMBER_RE = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?$")


def grade_rounded_decimal(text: str, gold: str) -> Optional[_Result]:
    """Gold with d >= 1 nonzero decimals: accept within half a unit of the last place.

    Golds like "3.0" are integers and go to MathArena (exact equality).
    """
    from sympy import N, Rational

    from .third_party.matharena.parser import extract_answer

    g = _strip_wrappers(gold).replace(" ", "")
    m = _DECIMAL_RE.match(g)
    if not m or int(m.group(1)) == 0:
        return None
    d = len(m.group(1))
    boxed = _last_boxed(text)
    if boxed is None:
        return False, ""
    ans, _ = extract_answer(text, strict_parsing=True, parse=True, list_answer=False, typed_delimiters=True)
    return _rounded_decimal_ok(boxed, ans, g, d), str(ans)


def _rounded_decimal_ok(boxed: str, ans, g: str, d: int) -> bool:
    from sympy import N, Rational

    raw = _strip_wrappers(boxed).replace(" ", "").replace("\\!", "").replace("\\,", "").replace("{,}", ",")
    written = _PLAIN_NUMBER_RE.match(raw)
    if written is not None and len(written.group(1) or "") < d:
        return False
    if ans is None or isinstance(ans, (str, list)):
        return False
    if written is not None:
        value = Rational(raw.replace(",", ""))
    else:
        value = N(ans, 30)
        if not getattr(value, "is_number", False) or not getattr(value, "is_real", False):
            return False
    return bool(abs(value - Rational(g)) < Rational(5, 10 ** (d + 1)))


def _grade_decimal_bracket_list(box: str, gold_inner: str) -> Optional[bool]:
    """Bracket list of plain numbers with at least one decimal: per-item rounding rule.

    Returns None if the gold is not such a list. Items match in any order.
    """
    from .third_party.matharena.parser import check_answers, parse_answer

    gold_items = [i for i in _split_top_level(gold_inner) if i]
    if len(gold_items) < 2 or not all(re.fullmatch(r"-?\d+(?:\.\d+)?", i) for i in gold_items):
        return None
    if not any(_DECIMAL_RE.match(i) and int(_DECIMAL_RE.match(i).group(1)) != 0 for i in gold_items):
        return None
    ans_items = [i for i in _split_top_level(box) if i]
    if len(ans_items) != len(gold_items):
        return False

    def item_ok(a: str, g: str) -> bool:
        parsed, _ = parse_answer(a, list_answer=False, typed_delimiters=True)
        m = _DECIMAL_RE.match(g)
        if m and int(m.group(1)) != 0:
            return _rounded_decimal_ok(a, parsed, g, len(m.group(1)))
        parsed_gold, _ = parse_answer(g, list_answer=False, typed_delimiters=True)
        return bool(check_answers(parsed, parsed_gold))

    remaining = list(gold_items)
    for a in ans_items:
        for i, g in enumerate(remaining):
            if item_ok(a, g):
                del remaining[i]
                break
        else:
            return False
    return not remaining


# ---------------------------------------------------------------- percent
_PERCENT_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*(?:\\%|%)$")


def _percent_to_fraction(s: str) -> Optional[str]:
    from decimal import Decimal

    m = _PERCENT_RE.match(_strip_wrappers(s).replace("\\,", "").strip())
    if not m:
        return None
    return format(Decimal(m.group(1)) / 100, "f")


def _resolve_percent(text: str, gold: str) -> tuple:
    """Divide a percentage gold and/or last box by 100 so both are plain numbers."""
    boxed = _last_boxed(text)
    if boxed is None:
        return text, gold
    gold_pct, box_pct = _percent_to_fraction(gold), _percent_to_fraction(boxed)
    if box_pct is not None:
        text = "\\boxed{" + box_pct + "}"
        if gold_pct is None:
            # A bare gold may already be in percent units: "10.92" vs "10.92\\%".
            m = _PERCENT_RE.match(_strip_wrappers(boxed).replace("\\,", "").strip())
            g = _strip_wrappers(gold).replace(" ", "")
            if m and re.fullmatch(r"-?\d+(?:\.\d+)?", g):
                from decimal import Decimal

                if Decimal(m.group(1)) == Decimal(g) and 1 < abs(Decimal(g)) < 100:
                    text = "\\boxed{" + g + "}"
    return text, (gold_pct if gold_pct is not None else gold)


# ---------------------------------------------------------------- theoremqa numeric golds
_INTEGER_LOOKING_RE = re.compile(r"^-?\d+(?:\.0+)?$")
_DECIMAL_GOLD_RE = re.compile(r"^-?\d+\.\d+$")


def grade_theoremqa_tolerance(text: str, gold: str) -> Optional[_Result]:
    """theoremqa only: a nonzero numeric gold accepts the last box within a relative tolerance.

    Integer-looking golds allow 1%. Decimal golds allow 0.1%, since many theoremqa
    golds are truncated or off in the last digit (6.2831852 for 2*pi, 0.4295 for
    (e-1)/4). Accept-only: a miss returns None so the normal rules decide.
    """
    from sympy import N, Rational

    from .third_party.matharena.parser import extract_answer

    g = _strip_wrappers(gold).replace(" ", "")
    if _INTEGER_LOOKING_RE.match(g):
        tol = Rational(1, 100)
    elif _DECIMAL_GOLD_RE.match(g):
        tol = Rational(1, 1000)
    else:
        return None
    if Rational(g) == 0 or _last_boxed(text) is None:
        return None
    ans, _ = extract_answer(text, strict_parsing=True, parse=True, list_answer=False, typed_delimiters=True)
    if ans is None or isinstance(ans, (str, list)):
        return None
    value = N(ans, 30)
    if not getattr(value, "is_number", False) or not getattr(value, "is_real", False):
        return None
    if abs(value - Rational(g)) < tol * abs(Rational(g)):
        return True, str(ans)
    return None


# ---------------------------------------------------------------- answer less precise than gold
def grade_rounded_answer(text: str, gold: str) -> Optional[_Result]:
    """Accept a plain decimal that is the gold rounded half-up to the answer's decimals.

    The answer needs at least 3 significant digits (every leading zero is
    ignored, so 0.368 qualifies, 0.083, 0.1 and 0.000 do not), the same sign as the gold,
    and fewer decimals than the gold. Accept-only.
    """
    from decimal import ROUND_HALF_UP, Decimal

    g = _strip_wrappers(gold).replace(" ", "")
    boxed = _last_boxed(text)
    if boxed is None or not re.fullmatch(r"-?\d+\.\d+", g):
        return None
    b = _strip_wrappers(boxed).replace(" ", "").replace("\\!", "").replace("\\,", "")
    m = re.fullmatch(r"-?(\d+)\.(\d+)", b)
    if not m:
        return None
    d, dg = len(m.group(2)), len(g.split(".")[1])
    significant = (m.group(1) + m.group(2)).lstrip("0")  # leading zeros never count
    if d >= dg or len(significant) < 3 or b.startswith("-") != g.startswith("-"):
        return None
    rounded = Decimal(g).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    if rounded == Decimal(b):
        return True, boxed
    return None


# ---------------------------------------------------------------- MathArena (strict)
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_CHOICE_ITEM_RE = re.compile(r"^\(?([A-Za-z])\)?$")


def _list_len(parsed) -> int:
    from .third_party.matharena.parser import TYPED_DELIMITED_ANSWER_TYPES, AnswerList

    if isinstance(parsed, AnswerList):
        return len(parsed.answers)
    if isinstance(parsed, TYPED_DELIMITED_ANSWER_TYPES):
        return -1  # typed collections carry their own structure
    return 1


def _choice_letters(raw: str) -> Optional[frozenset]:
    items = [i for i in _split_top_level(_strip_wrappers(raw)) if i]
    letters = []
    for item in items:
        m = _CHOICE_ITEM_RE.match(item.strip())
        if not m:
            return None
        letters.append(m.group(1).upper())
    if not letters or len(set(letters)) != len(letters):
        return None
    return frozenset(letters)


# a digit right after "\\log_" is a base (\\log_2 3), not the end of a number
_SPLIT_NUMBER_RE = re.compile(
    r"(?<!\\log_)\d\s*(?:\\q?quad|\\thinspace|\\[;:! ]|~|\\text\{[^{}]*\}|\s)+\s*\d"
)
_REPEATED_THIN_RE = re.compile(r"\d\s*(?:\\,\s*){2,}\d")
# "\\," is fine as a thousands separator (12\\,345) but not between other digit groups
_SPLIT_THIN_RE = re.compile(r"(?<!\\log_)\d\s*(?:\\,|\{,\})\s*(?!\d{3}(?!\d))\d")


def _has_split_number(box: str) -> bool:
    return bool(_SPLIT_NUMBER_RE.search(box) or _REPEATED_THIN_RE.search(box) or _SPLIT_THIN_RE.search(box))


def _grade_matharena(text: str, gold: str) -> _Result:
    from .third_party.matharena.parser import check_answers, extract_answer, extract_boxed_answer, parse_answer

    g = gold.strip().replace("$", "").strip()
    text2 = text
    if _THOUSANDS_RE.match(g):
        gold = g.replace(",", "")
        list_answer = False
        box = _last_boxed(text)
        if box is None:
            return False, ""
        b = box.strip().replace("\\,", "").replace("{,}", ",")
        if _THOUSANDS_RE.match(b):
            b = b.replace(",", "")
        elif "," in b:
            return False, box
        text2 = "\\boxed{" + b + "}"
    elif re.fullmatch(r"\[.*\]", g, flags=re.S):
        gold = g[1:-1]
        list_answer = "," in gold
        # MathArena joins several boxes on the final line ("\\boxed{0}, \\boxed{1}")
        box, _ = extract_boxed_answer(text, list_answer)
        if box is None:
            return False, ""
        b = re.sub(r"\\left|\\right", "", box).strip()
        b = re.sub(r"\^\{?\\?(?:T|top|intercal|mathsf\{T\})\}?$", "", b).strip()
        if b.startswith("("):
            return False, box
        if b.startswith("[") and b.endswith("]"):
            b = b[1:-1]
        elif b.startswith("["):
            return False, box
        decimal_list = _grade_decimal_bracket_list(b, gold)
        if decimal_list is not None:
            return decimal_list, b
        text2 = "\\boxed{" + b + "}"
    else:
        list_answer = "," in gold

    model_answer, _ = extract_answer(
        text2, strict_parsing=True, parse=True, list_answer=list_answer, typed_delimiters=True
    )
    extracted = "" if model_answer is None else str(model_answer)
    typed_gold, _ = parse_answer(gold, list_answer=list_answer, typed_delimiters=True)

    if list_answer:
        # MathArena's AnswerList silently drops items without digits ("A, C"),
        # which lets "B, D" match "A, C". Require every item to survive parsing.
        raw_gold_n = len([p for p in _split_top_level(gold) if p])
        model_raw, _ = extract_boxed_answer(text2, list_answer)
        raw_model_n = len([p for p in _split_top_level(model_raw or "") if p])
        n_gold, n_model = _list_len(typed_gold), _list_len(model_answer)
        if n_gold == 0 or (n_gold > 0 and n_gold != raw_gold_n) or (n_model > 0 and n_model != raw_model_n):
            gold_letters = _choice_letters(gold)
            if gold_letters is not None and model_raw is not None:
                return gold_letters == _choice_letters(model_raw), extracted
            return False, extracted
    return bool(check_answers(model_answer, typed_gold)), extracted


# ---------------------------------------------------------------- entry points
def grade_core(text: str, gold: str, source: Optional[str] = None) -> _Result:
    """Grade one response against one gold answer. Returns ``(is_correct, extracted)``.

    Raises LatexBackendError if the LaTeX parser cannot load; any other parser
    error counts as wrong.
    """
    from .third_party.matharena.parser import LatexBackendError, ensure_latex_backend

    ensure_latex_backend()
    gold = str(gold if gold is not None else "")
    boxed_gold = _last_boxed(gold)
    if boxed_gold is not None:
        gold = boxed_gold
    try:
        text, gold = _resolve_percent(text, gold)
        # MathArena's parser joins digits across spacing and \text (5 \text{ or } 6 -> 56),
        # and so do the patches below, so this guard comes first.
        # A box identical to the gold up to layout is safe (the normalizer keeps the
        # space between digits, so "5 6" never equals "56"); check it first.
        exact = grade_normalized_exact(text, gold)
        if exact is not None:
            return exact
        last_box = _last_boxed(text)
        if last_box is not None and _has_split_number(last_box):
            return False, last_box
        if source == "theoremqa":
            tolerant = grade_theoremqa_tolerance(text, gold)
            if tolerant is not None:
                return tolerant
        for patch in (grade_choice_letter, grade_normalized_exact, grade_rounded_answer, grade_boolean, grade_tuples, grade_rounded_decimal):
            result = patch(text, gold)
            if result is not None:
                return result
        return _grade_matharena(text, gold)
    except LatexBackendError:
        raise
    except Exception:
        return False, ""
