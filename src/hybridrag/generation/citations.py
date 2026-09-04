"""Parsing `[n]` citations and checking them against the context that was supplied.

This is *structural* verification: does every cited number correspond to a block the model
was actually given? It is free, deterministic, and catches the failure that matters most at
this stage -- a citation pointing at nothing, which renders as a plausible source link that
goes nowhere. Whether block 3 genuinely supports the sentence attached to it is a semantic
question, and it belongs with the LLM-as-judge and the golden set in Phase 4, where there
is something to measure the judge against.

Unresolvable citations are reported, never rewritten. Silently stripping them would hide
exactly the hallucination the citation layer exists to expose.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from hybridrag.generation.models import Citation, CitationReport
from hybridrag.retrieval import RetrievedChunk

# Matches [2] and grouped forms the model reaches for unprompted: [1, 3] and [1,3].
# Adjacent citations ([1][3]) are separate matches, which is the documented style.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citation_numbers(text: str) -> list[int]:
    """Every cited number, in order of first appearance and without duplicates."""
    numbers: list[int] = []
    for match in _CITATION.finditer(text):
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
