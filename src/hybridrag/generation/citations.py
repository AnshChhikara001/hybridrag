"""Parsing `[n]` citations and checking them against the context that was supplied.

Two layers of verification exist, and keeping them apart is the point:

* **Structural** -- does every cited number correspond to a block the model was actually
  given? Free, deterministic, and it catches the failure that renders as a plausible
  source link going nowhere. That is this module.
* **Semantic** -- does block 3 genuinely support the sentence attached to it? That needs a
  model, and it lives in `verification.py`.

Unresolvable citations are reported, never rewritten. Silently stripping them would hide
exactly the hallucination the citation layer exists to expose.

**Code is not prose, and brackets inside it are not citations.** This corpus documents a
CLI, so answers quote console output such as `[2248755]` -- a process id -- and Python
like `sys.argv[1]`. Read naively, those become citations to blocks 2,248,755 and 1: the
first is reported as a fabricated source the model never claimed, and the second silently
attributes a sentence to whichever chunk happens to rank first. Both were reachable on
this corpus, so fenced blocks and inline code spans are excluded from citation parsing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from hybridrag.chunking.segmentation import find_code_fences
from hybridrag.generation.models import Citation, CitationReport
from hybridrag.retrieval import RetrievedChunk

# Matches [2] and grouped forms the model reaches for unprompted: [1, 3] and [1,3].
# Adjacent citations ([1][3]) are separate matches, which is the documented style.
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# A single-backtick span, which markdown does not allow to cross a line break.
_INLINE_CODE = re.compile(r"`[^`\n]+`")


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Character spans where a bracketed number means something other than a citation.

    Fenced blocks first, then inline spans that do not already fall inside one, so a
    stray backtick inside a fence cannot open a phantom region.
    """
    spans = find_code_fences(text)
    for match in _INLINE_CODE.finditer(text):
        if not any(start <= match.start() < end for start, end in spans):
            spans.append((match.start(), match.end()))
    return spans


def parse_citation_numbers(text: str) -> list[int]:
    """Every cited number, in order of first appearance and without duplicates."""
    protected = protected_spans(text)
    numbers: list[int] = []
    for match in CITATION_PATTERN.finditer(text):
        if any(start <= match.start() < end for start, end in protected):
            continue
        for part in match.group(1).split(","):
            number = int(part.strip())
            if number not in numbers:
                numbers.append(number)
    return numbers


def resolve_citations(text: str, results: Sequence[RetrievedChunk]) -> CitationReport:
    """Map cited numbers back onto retrieved chunks and report what did not map.

    Block numbers are 1-based positions in the supplied context, matching
    `prompt.render_context`. The two must agree exactly; if they ever drift, every citation
    in the system points one place to the left, which is why both live next to each other.
    """
    cited = parse_citation_numbers(text)

    resolved: list[Citation] = []
    unresolved: list[int] = []
    for number in cited:
        if 1 <= number <= len(results):
            chunk = results[number - 1].chunk
            resolved.append(
                Citation(
                    number=number,
                    chunk_id=chunk.chunk_id,
                    relative_path=chunk.relative_path,
                    heading_path=chunk.heading_path,
                )
            )
        else:
            unresolved.append(number)

    return CitationReport(
        citations=resolved,
        unresolved=unresolved,
        # Retrieved but never cited. Not an error -- the generator is entitled to ignore a
        # chunk -- but it is the signal that separates "retrieval was too broad" from
        # "generation ignored good context" when an answer disappoints.
        uncited_blocks=[
            number for number in range(1, len(results) + 1) if number not in set(cited)
        ],
    )
