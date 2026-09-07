"""The retriever: search every index, fuse the rankings, hydrate the winners.

One class covers hybrid and single-index retrieval, because "hybrid vs dense-only" is a
comparison the brief requires and two separate classes would let the two sides of that
comparison drift apart. `ablation()` derives the restricted retriever *from* the full one,
so they cannot disagree about candidate depth, rank constant, or which store they read.

Indexes are addressed through a structural protocol rather than by concrete type. Both
`DenseIndex.search` and `SparseIndex.search` already return `(chunk_id, score)` best-first,
so they satisfy it as written -- and Phase 2's reranker can slot in behind the same shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt

from hybridrag.chunk_store import ChunkStore
from hybridrag.models import Chunk
from hybridrag.retrieval.fusion import (
    DEFAULT_RANK_CONSTANT,
    FusedResult,
    RetrieverHit,
    reciprocal_rank_fusion,
)

# Each index is searched this deep before fusion, regardless of how many results the caller
# asked for. Fusing two top-10 lists and returning 10 throws away the agreement RRF exists
# to find: a chunk both retrievers rank 11th never enters the fusion, even though two votes
# just outside the cut is exactly the signal that should promote it into the final list.
DEFAULT_CANDIDATES = 50


@runtime_checkable
class SearchIndex(Protocol):
    """Anything that ranks chunk ids for a query, best first."""

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...


class RetrievedChunk(BaseModel):
    """A chunk with its position and the evidence for it."""

    chunk: Chunk
    rank: PositiveInt = Field(description="1-based position in the final fused ranking.")
    score: float = Field(description="Fused RRF score.")
    hits: dict[str, RetrieverHit] = Field(
        default_factory=dict, description="Per-retriever rank and score; absent means it missed."
    )
    rerank_score: float | None = Field(
        default=None,
        description="Cross-encoder score, if a reranking pass reordered this result. `score` "
        "is left as the original fused RRF value either way, so a caller can always see "
        "what fusion alone would have said and never confuses the two scales.",
    )

    @property
    def retrievers(self) -> tuple[str, ...]:
        return tuple(sorted(self.hits))


@runtime_checkable
class Retriever(Protocol):
    """Anything that turns a query into ranked, hydrated chunks.

    `HybridRetriever` satisfies this structurally, and so does `RerankingRetriever`
    (`rerank.py`) -- which is what lets `run_arm` score a reranked pipeline with no change
    to the harness, and lets a reranker wrap either one interchangeably.
    """

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]: ...


class HybridRetriever:
    """Fuses one or more indexes into a single ranked list of chunks."""

    def __init__(
        self,
        indexes: Mapping[str, SearchIndex],
        store: ChunkStore,
        *,
        weights: Mapping[str, NonNegativeFloat] | None = None,
        rank_constant: int = DEFAULT_RANK_CONSTANT,
        candidates: int = DEFAULT_CANDIDATES,
    ) -> None:
        if not indexes:
            raise ValueError("a retriever needs at least one index")
        if candidates <= 0:
            raise ValueError(f"candidates must be positive, got {candidates}")
        self.indexes = dict(indexes)
        self.store = store
        self.weights = dict(weights) if weights else None
        self.rank_constant = rank_constant
        self.candidates = candidates

    def ablation(self, *names: str) -> HybridRetriever:
        """A retriever over a subset of these indexes, everything else held constant.

        This is how the dense-only baseline is built. Deriving it from the full retriever
        rather than constructing a second one means the comparison isolates the thing being
        compared -- the presence of a retriever -- and nothing else.
        """
        unknown = set(names) - set(self.indexes)
        if unknown:
            raise ValueError(f"no such index: {sorted(unknown)}. Known: {sorted(self.indexes)}.")
        return HybridRetriever(
            {name: self.indexes[name] for name in names},
            self.store,
            weights=(
                {name: w for name, w in self.weights.items() if name in names}
                if self.weights
                else None
            ),
            rank_constant=self.rank_constant,
            candidates=self.candidates,
        )

    def fuse(self, query: str, k: int = 10) -> list[FusedResult]:
        """Ranked chunk ids without hydrating them.

        Separate from `retrieve` because evaluation grades ids and never needs the text:
        a Recall@k sweep over the golden set would otherwise deserialise the whole corpus
        for numbers that only compare identifiers.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        # A caller asking for more results than the candidate depth would otherwise be
        # silently capped at `candidates`; searching deeper is what they meant.
        depth = max(k, self.candidates)
        rankings = {name: index.search(query, depth) for name, index in self.indexes.items()}
        return reciprocal_rank_fusion(
            rankings, weights=self.weights, rank_constant=self.rank_constant, limit=k
        )

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Top-k chunks for a query, best first, with their retrieval evidence."""
        fused = self.fuse(query, k)
        if not fused:
            return []
        # get_many preserves the requested order, so the ranking survives hydration.
        chunks = self.store.get_many([result.chunk_id for result in fused])
        return [
            RetrievedChunk(chunk=chunk, rank=rank, score=result.score, hits=result.hits)
            for rank, (chunk, result) in enumerate(zip(chunks, fused, strict=True), start=1)
        ]
