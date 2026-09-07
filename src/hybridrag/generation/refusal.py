"""Building the structured refusal.

Everything here is derived from the retrieval results that were already computed, so a
refusal costs nothing beyond the search that had happened anyway. That matters: the
cheapest possible response is also the one a user is most likely to receive on a bad day,
and it should still leave them somewhere to go.

The near misses are reported with their similarity scores rather than as a bare list of
filenames. On the gate path the scores are the reason for the refusal, and showing them
lets a reader see the difference between "nothing came close" and "several pages came
close and none of them said it" without reading the chunks.
"""

from __future__ import annotations

from collections.abc import Sequence

from hybridrag.generation.models import MAX_DOCUMENTS_TO_CHECK, NearMiss, Refusal, RefusalKind
from hybridrag.retrieval import RetrievedChunk

DENSE = "dense"

_MISSING_TEXT = {
    RefusalKind.NO_RESULTS: (
        "Retrieval returned nothing for this question, so the corpus was never consulted: "
    ),
    RefusalKind.LOW_CONFIDENCE: (
        "Nothing in the indexed documentation is close enough to this question to answer it: "
    ),
    RefusalKind.MODEL_DECLINED: (
        "The pages below were retrieved for this question, but none of them states the answer: "
    ),
}


def _near_misses(results: Sequence[RetrievedChunk]) -> list[NearMiss]:
    misses: list[NearMiss] = []
    for result in results[:MAX_DOCUMENTS_TO_CHECK]:
        heading = " > ".join(result.chunk.heading_path)
        hit = result.hits.get(DENSE)
        misses.append(
            NearMiss(
                source=f"{result.chunk.relative_path}{' > ' + heading if heading else ''}",
                similarity=hit.score if hit is not None else None,
            )
        )
    return misses


def _documents(results: Sequence[RetrievedChunk]) -> list[str]:
    """Distinct source files in rank order.

    De-duplicated by path because several chunks of one page is one page to open, and a
    refusal listing `tutorial/body.md` three times reads as noise rather than a pointer.
    """
    paths: list[str] = []
    for result in results:
        path = result.chunk.relative_path
        if path not in paths:
            paths.append(path)
        if len(paths) == MAX_DOCUMENTS_TO_CHECK:
            break
    return paths


def build_refusal(
    kind: RefusalKind, question: str, reason: str, results: Sequence[RetrievedChunk]
) -> Refusal:
    """Assemble the structured refusal from what retrieval already returned."""
    return Refusal(
        kind=kind,
        reason=reason,
        what_was_found=_near_misses(results),
        what_was_missing=f"{_MISSING_TEXT[kind]}{question.strip()}",
        documents_to_check=_documents(results),
    )
