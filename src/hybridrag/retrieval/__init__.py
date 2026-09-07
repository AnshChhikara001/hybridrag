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
    Retriever,
    SearchIndex,
)
from hybridrag.retrieval.rerank import (
    DEFAULT_RERANK_DEPTH,
    DEFAULT_RERANK_MODEL,
    CrossEncoderReranker,
    Reranker,
    RerankingRetriever,
)

__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_RANK_CONSTANT",
    "DEFAULT_RERANK_DEPTH",
    "DEFAULT_RERANK_MODEL",
    "CrossEncoderReranker",
    "FusedResult",
    "HybridRetriever",
    "Reranker",
    "RerankingRetriever",
    "RetrievedChunk",
    "Retriever",
    "RetrieverHit",
    "SearchIndex",
    "reciprocal_rank_fusion",
]
