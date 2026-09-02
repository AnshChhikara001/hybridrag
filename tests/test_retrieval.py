"""The retriever: index -> fusion -> hydration.

Indexes are stubbed so these tests fix the wiring rather than re-testing BM25 or Chroma,
both of which have their own suites. The stub records the depth it was searched at, because
searching each index deeper than `k` is what makes fusion able to promote a chunk that both
retrievers ranked just outside the final cut.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from hybridrag.chunk_store import ChunkStore, MissingChunkError
from hybridrag.models import Chunk
from hybridrag.retrieval import HybridRetriever

MakeChunk = Callable[..., Chunk]


class StubIndex:
    """A fixed ranking, satisfying `SearchIndex` and remembering how deep it was searched."""

    def __init__(self, ranked: list[tuple[str, float]]) -> None:
        self.ranked = ranked
        self.searched_at: list[int] = []

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        self.searched_at.append(k)
        return self.ranked[:k]


@pytest.fixture
def store(make_chunk: MakeChunk) -> ChunkStore:
    store = ChunkStore.in_memory()
    store.add([make_chunk(i, f"chunk number {i}") for i in range(6)])
    return store


@pytest.fixture
def chunk_ids(store: ChunkStore, make_chunk: MakeChunk) -> list[str]:
    return [make_chunk(i).chunk_id for i in range(6)]


def test_retrieved_chunks_are_hydrated_in_fused_order(
    store: ChunkStore, chunk_ids: list[str]
) -> None:
    dense = StubIndex([(chunk_ids[0], 0.9), (chunk_ids[1], 0.8)])
    sparse = StubIndex([(chunk_ids[1], 12.0), (chunk_ids[2], 3.0)])

    results = HybridRetriever({"dense": dense, "sparse": sparse}, store).retrieve("q", k=3)

    # chunk_ids[1] is second on both lists and first after fusion.
    assert [result.chunk.chunk_id for result in results] == [
        chunk_ids[1],
        chunk_ids[0],
        chunk_ids[2],
    ]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].chunk.text == "chunk number 1"


def test_results_carry_the_evidence_for_each_retriever(
    store: ChunkStore, chunk_ids: list[str]
) -> None:
    dense = StubIndex([(chunk_ids[0], 0.83)])
    sparse = StubIndex([(chunk_ids[1], 22.4), (chunk_ids[0], 9.1)])

    results = HybridRetriever({"dense": dense, "sparse": sparse}, store).retrieve("q", k=2)
    by_id = {result.chunk.chunk_id: result for result in results}

    assert by_id[chunk_ids[0]].retrievers == ("dense", "sparse")
    assert by_id[chunk_ids[0]].hits["sparse"].rank == 2
    assert by_id[chunk_ids[0]].hits["dense"].score == 0.83
    assert by_id[chunk_ids[1]].retrievers == ("sparse",)


def test_each_index_is_searched_deeper_than_the_requested_k(
    store: ChunkStore, chunk_ids: list[str]
) -> None:
    dense = StubIndex([(chunk_ids[0], 0.9)])
    sparse = StubIndex([(chunk_ids[1], 4.0)])

    HybridRetriever({"dense": dense, "sparse": sparse}, store, candidates=25).retrieve("q", k=3)

    assert dense.searched_at == [25]
    assert sparse.searched_at == [25]


def test_asking_for_more_than_the_candidate_depth_searches_deeper(
    store: ChunkStore, chunk_ids: list[str]
) -> None:
    """Otherwise a caller asking for 100 results would be silently capped at `candidates`."""
    dense = StubIndex([(chunk_ids[0], 0.9)])

    HybridRetriever({"dense": dense}, store, candidates=10).retrieve("q", k=40)

    assert dense.searched_at == [40]


def test_an_ablation_uses_only_the_named_index(store: ChunkStore, chunk_ids: list[str]) -> None:
    dense = StubIndex([(chunk_ids[0], 0.9), (chunk_ids[1], 0.8)])
    sparse = StubIndex([(chunk_ids[2], 12.0)])
    retriever = HybridRetriever({"dense": dense, "sparse": sparse}, store)

    results = retriever.ablation("dense").retrieve("q", k=5)

    assert [result.chunk.chunk_id for result in results] == [chunk_ids[0], chunk_ids[1]]
    assert sparse.searched_at == []


def test_an_ablation_inherits_the_full_retrievers_configuration(
    store: ChunkStore, chunk_ids: list[str]
) -> None:
    """The dense-only baseline must differ from hybrid in one variable, not several."""
    dense = StubIndex([(chunk_ids[0], 0.9)])
    retriever = HybridRetriever(
        {"dense": dense, "sparse": StubIndex([])},
        store,
        weights={"dense": 2.0, "sparse": 0.5},
        rank_constant=17,
        candidates=33,
    )

    ablated = retriever.ablation("dense")

    assert ablated.rank_constant == 17
    assert ablated.candidates == 33
    assert ablated.weights == {"dense": 2.0}
    assert ablated.store is store


def test_ablating_an_unknown_index_is_an_error(store: ChunkStore) -> None:
    retriever = HybridRetriever({"dense": StubIndex([])}, store)

    with pytest.raises(ValueError, match="no such index"):
        retriever.ablation("sparse")


def test_weights_reach_the_fusion(store: ChunkStore, chunk_ids: list[str]) -> None:
    """Opposed rankings, so the weight is the only thing deciding the winner."""
    indexes = {
        "dense": StubIndex([(chunk_ids[0], 0.9), (chunk_ids[1], 0.4)]),
        "sparse": StubIndex([(chunk_ids[1], 22.0), (chunk_ids[0], 3.0)]),
    }

    favour_dense = HybridRetriever(indexes, store, weights={"dense": 2.0}).retrieve("q", k=1)
    favour_sparse = HybridRetriever(indexes, store, weights={"sparse": 2.0}).retrieve("q", k=1)

    assert favour_dense[0].chunk.chunk_id == chunk_ids[0]
    assert favour_sparse[0].chunk.chunk_id == chunk_ids[1]


def test_fuse_ranks_ids_without_touching_the_store(chunk_ids: list[str]) -> None:
    """Evaluation grades identifiers, so it should never pay to deserialise the corpus."""
    empty_store = ChunkStore.in_memory()
    retriever = HybridRetriever({"dense": StubIndex([(chunk_ids[0], 0.9)])}, empty_store)

    assert [result.chunk_id for result in retriever.fuse("q", k=5)] == [chunk_ids[0]]
    with pytest.raises(MissingChunkError):
        retriever.retrieve("q", k=5)


def test_a_query_nothing_matches_returns_nothing(store: ChunkStore) -> None:
    retriever = HybridRetriever({"dense": StubIndex([]), "sparse": StubIndex([])}, store)

    assert retriever.retrieve("q", k=5) == []
    assert retriever.fuse("q", k=5) == []


def test_fewer_results_than_requested_is_not_padded(
    store: ChunkStore, chunk_ids: list[str]
) -> None:
    retriever = HybridRetriever({"dense": StubIndex([(chunk_ids[0], 0.9)])}, store)

    assert len(retriever.retrieve("q", k=10)) == 1


def test_a_retriever_needs_at_least_one_index(store: ChunkStore) -> None:
    with pytest.raises(ValueError, match="at least one index"):
        HybridRetriever({}, store)


@pytest.mark.parametrize("candidates", [0, -1])
def test_a_non_positive_candidate_depth_is_rejected(store: ChunkStore, candidates: int) -> None:
    with pytest.raises(ValueError, match="candidates"):
        HybridRetriever({"dense": StubIndex([])}, store, candidates=candidates)


def test_a_non_positive_k_is_rejected(store: ChunkStore) -> None:
    retriever = HybridRetriever({"dense": StubIndex([])}, store)

    with pytest.raises(ValueError, match="k must be positive"):
        retriever.retrieve("q", k=0)
