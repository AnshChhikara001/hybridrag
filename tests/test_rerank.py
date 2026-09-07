"""The reranker: a fixed-depth cross-encoder pass wrapped around any retriever.

The cross-encoder itself is stubbed for every test but `TestRealModel` -- what needs
checking here is the wiring (how many candidates are fetched, how they are reordered, what
happens at the edges), not that ONNX multiplies matrices correctly. That one real-model
check earns its `slow` marker: it downloads ~80 MB on a cold cache (D8).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from hybridrag.models import Chunk
from hybridrag.retrieval import Retriever
from hybridrag.retrieval.hybrid import RetrievedChunk
from hybridrag.retrieval.rerank import (
    CrossEncoderReranker,
    Reranker,
    RerankingRetriever,
)

MakeChunk = Callable[..., Chunk]


class StubReranker:
    """Scores documents by a fixed lookup, so the test controls the reordering directly."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen_query: str | None = None
        self.seen_documents: list[str] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.seen_query = query
        self.seen_documents = list(documents)
        return [self.scores[doc] for doc in documents]


class StubRetriever:
    """A `Retriever` returning a scripted, already-fused ranking."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.requested_k: list[int] = []

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        self.requested_k.append(k)
        return self.results[:k]


def result(chunk: Chunk, rank: int, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rank=rank, score=score)


@pytest.fixture
def chunks(make_chunk: MakeChunk) -> list[Chunk]:
    return [make_chunk(i, f"chunk number {i}") for i in range(4)]


class TestContracts:
    def test_cross_encoder_reranker_satisfies_the_protocol(self) -> None:
        assert isinstance(CrossEncoderReranker(), Reranker)

    def test_a_stub_retriever_satisfies_the_protocol(self, chunks: list[Chunk]) -> None:
        assert isinstance(StubRetriever([result(chunks[0], 1)]), Retriever)


class TestReranking:
    def test_the_cross_encoder_score_reorders_the_candidates(self, chunks: list[Chunk]) -> None:
        """RRF put chunk 0 first; the cross-encoder disagrees and must win."""
        base = StubRetriever([result(chunks[0], 1), result(chunks[1], 2), result(chunks[2], 3)])
        reranker = StubReranker(
            {
                "chunk number 0": 0.1,
                "chunk number 1": 0.9,
                "chunk number 2": 0.5,
            }
        )

        reranked = RerankingRetriever(base, reranker).retrieve("q", k=3)

        assert [r.chunk.chunk_id for r in reranked] == [
            chunks[1].chunk_id,
            chunks[2].chunk_id,
            chunks[0].chunk_id,
        ]
        assert [r.rank for r in reranked] == [1, 2, 3]

    def test_original_fused_score_is_untouched_and_rerank_score_is_added(
        self, chunks: list[Chunk]
    ) -> None:
        base = StubRetriever([result(chunks[0], 1, score=0.42)])
        reranker = StubReranker({"chunk number 0": 7.0})

        reranked = RerankingRetriever(base, reranker).retrieve("q", k=1)

        assert reranked[0].score == 0.42
        assert reranked[0].rerank_score == 7.0

    def test_fetches_a_fixed_depth_regardless_of_the_requested_k(self, chunks: list[Chunk]) -> None:
        """The pool size is the pipeline's shape (D8); it must not scale with an eval's `k`."""
        base = StubRetriever([result(c, i + 1) for i, c in enumerate(chunks)])
        reranker = StubReranker({f"chunk number {i}": float(i) for i in range(4)})

        RerankingRetriever(base, reranker, rerank_depth=2).retrieve("q", k=50)

        assert base.requested_k == [2]

    def test_the_final_list_is_capped_at_k_even_with_a_deeper_pool(
        self, chunks: list[Chunk]
    ) -> None:
        base = StubRetriever([result(c, i + 1) for i, c in enumerate(chunks)])
        reranker = StubReranker({f"chunk number {i}": float(i) for i in range(4)})

        reranked = RerankingRetriever(base, reranker, rerank_depth=4).retrieve("q", k=2)

        assert len(reranked) == 2

    def test_fewer_candidates_than_k_is_not_padded(self, chunks: list[Chunk]) -> None:
        base = StubRetriever([result(chunks[0], 1)])
        reranker = StubReranker({"chunk number 0": 1.0})

        reranked = RerankingRetriever(base, reranker, rerank_depth=20).retrieve("q", k=5)

        assert len(reranked) == 1

    def test_an_empty_base_result_reranks_to_nothing(self) -> None:
        base = StubRetriever([])
        reranker = StubReranker({})

        assert RerankingRetriever(base, reranker).retrieve("q", k=5) == []
        assert reranker.seen_query is None

    def test_ties_break_on_chunk_id_for_reproducibility(self, chunks: list[Chunk]) -> None:
        """Matches `reciprocal_rank_fusion`'s tie-break, so ties never depend on sort order."""
        base = StubRetriever([result(chunks[2], 1), result(chunks[0], 2), result(chunks[1], 3)])
        reranker = StubReranker(
            {"chunk number 0": 1.0, "chunk number 1": 1.0, "chunk number 2": 1.0}
        )

        reranked = RerankingRetriever(base, reranker).retrieve("q", k=3)

        assert [r.chunk.chunk_id for r in reranked] == sorted(c.chunk_id for c in chunks[:3])

    def test_a_non_positive_k_is_rejected(self, chunks: list[Chunk]) -> None:
        base = StubRetriever([result(chunks[0], 1)])
        with pytest.raises(ValueError, match="k must be positive"):
            RerankingRetriever(base, StubReranker({})).retrieve("q", k=0)

    def test_a_non_positive_rerank_depth_is_rejected(self, chunks: list[Chunk]) -> None:
        base = StubRetriever([result(chunks[0], 1)])
        with pytest.raises(ValueError, match="rerank_depth"):
            RerankingRetriever(base, StubReranker({}), rerank_depth=0)


@pytest.mark.slow
class TestRealModel:
    """One end-to-end check against the actual cross-encoder, since everything above is faked."""

    def test_the_relevant_document_scores_above_the_unrelated_one(self) -> None:
        reranker = CrossEncoderReranker()

        scores = reranker.score(
            "How do I declare a path parameter?",
            [
                "Install Docker Desktop and start the daemon.",
                "Declare path parameters with the same syntax used by Python format strings.",
            ],
        )

        assert scores[1] > scores[0]
