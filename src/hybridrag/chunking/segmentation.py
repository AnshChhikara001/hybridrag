"""Boundary-finding primitives shared by the structure-aware and semantic chunkers.

Everything here returns character spans into the document text, never substrings, so
callers can build chunks whose offsets address `Document.text` exactly.

Code fences are the load-bearing concept. This corpus has 800 fence lines and 304 sections
containing `@app.get`; half a code example is useless both to a retriever matching on
identifiers and to a model asked to answer from it. Fences are therefore atomic: no
paragraph or sentence boundary is ever reported inside one.
"""

from __future__ import annotations

import re

_FENCE_LINE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})", re.MULTILINE)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
# A sentence ends on .!? followed by space and an opening character. Requiring an uppercase
# or quote next avoids splitting `app.get(...)` and decimals such as 0.95, both common here.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z\"'(\[])")


def find_code_fences(text: str) -> list[tuple[int, int]]:
    """Character spans of fenced code blocks, including their fence lines.

    An unterminated fence runs to end of text, which is what a markdown renderer does and
    keeps a truncated example from being split mid-way.
    """
    starts = [m.start() for m in _FENCE_LINE.finditer(text)]
    fences: list[tuple[int, int]] = []
    for index in range(0, len(starts) - 1, 2):
        closing = starts[index + 1]
        line_end = text.find("\n", closing)
        # End at the close of the fence line, excluding its trailing newline, so this
        # matches the convention paragraph_spans and sentence_spans use. Including the
        # newline made the paragraph break after a fence look like it fell *inside* the
        # fence, suppressing a legitimate split point and fusing code to the prose after it.
        fences.append((starts[index], len(text) if line_end == -1 else line_end))
    if len(starts) % 2 == 1:
        fences.append((starts[-1], len(text)))
    return fences


def _inside(position: int, regions: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in regions)


def _split_on(
    pattern: re.Pattern[str], text: str, start: int, end: int, protected: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Split [start, end) at every pattern match that falls outside a protected region."""
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in pattern.finditer(text, start, end):
        if _inside(match.start(), protected):
            continue
        if match.start() > cursor:
            spans.append((cursor, match.start()))
        cursor = match.end()
    if cursor < end:
        spans.append((cursor, end))
    return spans or [(start, end)]


def paragraph_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Paragraph spans within [start, end), treating each code fence as one paragraph."""
    fences = [f for f in find_code_fences(text) if f[0] < end and f[1] > start]
    return _split_on(_PARAGRAPH_BREAK, text, start, end, fences)


def sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence spans within [start, end), never breaking inside a code fence.

    A deliberately simple regex rather than an NLP dependency: on a corpus this code-dense,
    general-purpose sentence splitters mis-handle dotted identifiers and version numbers,
    so a heavyweight dependency would buy inaccuracy. Documented as a known limitation.
    """
    fences = [f for f in find_code_fences(text) if f[0] < end and f[1] > start]
    spans = _split_on(_SENTENCE_BREAK, text, start, end, fences)
    refined: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        # Newlines also end sentences in documentation prose (list items, table rows),
        # but only outside fences.
        refined.extend(_split_on(_PARAGRAPH_BREAK, text, span_start, span_end, fences))
    return refined
