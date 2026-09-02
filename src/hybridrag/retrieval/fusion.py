"""Reciprocal Rank Fusion over any number of ranked lists.

    score(d) = sum over retrievers r of  weight[r] / (rank_constant + rank_r(d))

Deliberately a pure function over `(chunk_id, score)` lists: it needs no index, no store
and no embedder, so the fusion arithmetic -- the part that has to be exactly right -- is
tested directly rather than inferred from end-to-end behaviour.

**Why ranks and not scores.** The obvious alternative is to normalise both retrievers'
scores and add them. It does not work here: BM25 scores are unbounded and depend on corpus
statistics, cosine similarity sits in [-1, 1], and the two have no common unit. Min-max
normalising per query makes the fused ranking depend on the *spread* of each list, so one
outlier result rescales everything below it and a retriever that happens to be confident on
a given query dominates one that happens not to be. RRF reads only positions, so it is
scale-free by construction and needs no per-corpus calibration.

`rank_constant` is RRF's `k` from Cormack et al. (2009), named in full here because `k`
already means "how many results to return" everywhere else in this codebase. It damps the
top of each list: at 60, rank 1 scores 1/61 and rank 2 scores 1/62, a 1.6% difference, so
one retriever cannot win on its own first place alone -- agreement between retrievers is
what promotes a chunk. Lower it and the fusion approaches "whoever ranked it first wins".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt

DEFAULT_RANK_CONSTANT = 60


class RetrieverHit(BaseModel):
    """One retriever's opinion of one chunk, kept for explainability.

    The dashboard has to show *why* a chunk was retrieved, and a fused score alone cannot
    answer that. Holding the original rank, the original score and the contribution to the
    fused total means the breakdown is a lookup rather than a re-computation.
    """

    rank: PositiveInt = Field(description="1-based position in that retriever's own list.")
    score: float = Field(description="That retriever's native score: BM25 weight or cosine.")
    contribution: float = Field(description="weight / (rank_constant + rank).")


class FusedResult(BaseModel):
    """A chunk id with its fused score and the per-retriever breakdown behind it."""

    chunk_id: str
    score: float
    hits: dict[str, RetrieverHit] = Field(
        default_factory=dict, description="Keyed by retriever name; absent means it missed."
    )

    @property
    def retrievers(self) -> tuple[str, ...]:
        """Which retrievers found this chunk, in a stable order."""
        return tuple(sorted(self.hits))


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    weights: Mapping[str, NonNegativeFloat] | None = None,
    rank_constant: int = DEFAULT_RANK_CONSTANT,
    limit: int | None = None,
) -> list[FusedResult]:
    """Fuse ranked `(chunk_id, score)` lists into one ranking.

    Each list must already be sorted best-first; position in the list *is* the rank, which
    is the only thing fusion reads from it. Scores are carried through untouched for
    explainability and never enter the arithmetic.

    Weights default to 1.0. A name in `weights` that is not in `rankings` raises rather
    than being ignored -- a typo there silently disables a retriever and the system keeps
    answering, slightly worse, with nothing to indicate why.
    """
    if rank_constant <= 0:
        raise ValueError(f"rank_constant must be positive, got {rank_constant}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive when given, got {limit}")

    resolved = dict.fromkeys(rankings, 1.0) | dict(weights or {})
    unknown = set(resolved) - set(rankings)
    if unknown:
        raise ValueError(
            f"weights name retrievers that were not supplied: {sorted(unknown)}. "
            f"Known retrievers: {sorted(rankings)}."
        )
    if any(weight < 0.0 for weight in resolved.values()):
        raise ValueError(f"weights must be non-negative, got {resolved}")

    fused: dict[str, FusedResult] = {}
    for name, ranked in rankings.items():
        weight = resolved[name]
        for position, (chunk_id, score) in enumerate(ranked, start=1):
            contribution = weight / (rank_constant + position)
            result = fused.setdefault(chunk_id, FusedResult(chunk_id=chunk_id, score=0.0))
            result.score += contribution
            result.hits[name] = RetrieverHit(rank=position, score=score, contribution=contribution)

    # A zero total only happens when every retriever that found the chunk was weighted to
    # zero, i.e. it was deliberately switched off. Dropping those keeps an ablation's
    # output honest instead of padding it with chunks nothing voted for.
    ranked_results = [result for result in fused.values() if result.score > 0.0]
    # Ties break on chunk id so the same corpus and query rank identically on every run;
    # otherwise evaluation numbers drift between runs without the retrieval changing.
    ranked_results.sort(key=lambda result: (-result.score, result.chunk_id))
    return ranked_results if limit is None else ranked_results[:limit]
