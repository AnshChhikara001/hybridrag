"""Retrieval: rank fusion over the dense and sparse indexes."""

from __future__ import annotations

from hybridrag.retrieval.fusion import (
    DEFAULT_RANK_CONSTANT,
    FusedResult,
    RetrieverHit,
    reciprocal_rank_fusion,
)
from hybridrag.retrieval.hybrid import (
    DEFAULT_CANDIDATES,
    HybridRetriever,
    RetrievedChunk,
    SearchIndex,
)

__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_RANK_CONSTANT",
    "FusedResult",
    "HybridRetriever",
    "RetrievedChunk",
    "RetrieverHit",
    "SearchIndex",
    "reciprocal_rank_fusion",
]
