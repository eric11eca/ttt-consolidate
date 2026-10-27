"""Removes sentences that state the boxed final answer from reasoning trajectories."""

import re


BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{")


def remove_boxed_answer_sentence(text):
    """Remove every sentence containing a LaTeX ``\\boxed{}`` answer."""
    if not isinstance(text, str) or not BOXED_ANSWER_RE.search(text):
        return text

    pieces = []
    cursor = 0
    for match in BOXED_ANSWER_RE.finditer(text):
        if match.start() < cursor:
            continue
        sentence_start = _sentence_start(text, match.start())
        boxed_end = _boxed_expression_end(text, match.end())
        sentence_end = _sentence_end(text, boxed_end)
        pieces.append(text[cursor:sentence_start])
        cursor = sentence_end
    pieces.append(text[cursor:])

    return _normalize_removed_sentence_spacing("".join(pieces))


def remove_boxed_answer_sentences(texts):
    if not isinstance(texts, list):
        return texts
    return [remove_boxed_answer_sentence(text) for text in texts]


def _sentence_start(text, boxed_start):
    for idx in range(boxed_start - 1, -1, -1):
        if text[idx] in ".!?\n":
            return idx + 1
    return 0


def _boxed_expression_end(text, boxed_content_start):
    depth = 1
    for idx in range(boxed_content_start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx + 1
    return boxed_content_start


def _sentence_end(text, search_start):
    for idx in range(search_start, len(text)):
        if text[idx] in ".!?\n":
            return idx + 1
    return len(text)


def _normalize_removed_sentence_spacing(text):
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
