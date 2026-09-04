"""The question-to-answer path, with a stubbed model.

No network here: the stub returns whatever text a test needs and records whether it was
called at all. That last part is the point of several of these -- the retrieval gate is
only worth having if it refuses *before* spending a request.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from hybridrag.chunk_store import ChunkStore
from hybridrag.generation import Answerer, Completion
from hybridrag.generation.prompt import REFUSAL_SENTINEL
from hybridrag.models import Chunk
from hybridrag.retrieval import HybridRetriever

MakeChunk = Callable[..., Chunk]


class StubModel:
    """A fixed reply, satisfying `LanguageModel` and remembering its calls."""

    def __init__(self, reply: str = "Query parameters are function arguments [1].") -> None:
        self.model_name = "stub-model"
        self.reply = reply
        self.prompts: list[str] = []

    def generate(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> Completion:
        self.prompts.append(prompt)
        return Completion(
            text=self.reply,
            model=self.model_name,
            input_tokens=120,
            output_tokens=30,
            thinking_tokens=5,
            cost_usd=0.0,
            latency_s=0.4,
        )


class StubIndex:
    def __init__(self, ranked: list[tuple[str, float]]) -> None:
        self.ranked = ranked

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self.ranked[:k]


@pytest.fixture
def store(make_chunk: MakeChunk) -> ChunkStore:
    store = ChunkStore.in_memory()
    store.add([make_chunk(i, f"block {i} content") for i in range(4)])
    return store


@pytest.fixture
def ids(make_chunk: MakeChunk) -> list[str]:
    return [make_chunk(i).chunk_id for i in range(4)]


def build_retriever(
    store: ChunkStore, ids: list[str], *, dense_score: float = 0.55, sparse: bool = True
) -> HybridRetriever:
    indexes: dict[str, StubIndex] = {
        "dense": StubIndex([(ids[0], dense_score), (ids[1], dense_score - 0.1)])
    }
    if sparse:
        indexes["sparse"] = StubIndex([(ids[0], 21.0), (ids[2], 8.0)])
    return HybridRetriever(indexes, store)


def test_a_grounded_answer_carries_its_citations_and_cost(
    store: ChunkStore, ids: list[str]
) -> None:
    model = StubModel("Declared as function arguments [1].")
    answer = Answerer(build_retriever(store, ids), model, k=3).answer("how do query params work")

    assert answer.answered
    assert answer.text == "Declared as function arguments [1]."
    assert [c.number for c in answer.citations] == [1]
    assert answer.citations[0].relative_path == "tutorial/query-params.md"
    assert answer.input_tokens == 120
    assert answer.output_tokens == 35  # visible output plus billed reasoning tokens
    assert answer.model == "stub-model"


def test_the_context_the_model_saw_is_kept_with_the_answer(
    store: ChunkStore, ids: list[str]
) -> None:
    """Phase 5's dashboard shows the retrieved chunks beside the answer.

    Asserted as "the top result plus this set, ranked 1..n" rather than as an exact
    sequence: ids[1] and ids[2] are each rank 2 on their own retriever, so they tie on
    fused score and their order is decided by the chunk-id tie-break. That rule is pinned
    in test_fusion.py; re-deriving it here would test the arithmetic twice and couple this
    test to it.
    """
    answer = Answerer(build_retriever(store, ids), StubModel(), k=3).answer("q")

    assert answer.retrieved[0].chunk.chunk_id == ids[0]
    assert {r.chunk.chunk_id for r in answer.retrieved} == {ids[0], ids[1], ids[2]}
    assert [r.rank for r in answer.retrieved] == [1, 2, 3]


def test_confidence_reads_dense_cosine_not_the_fused_score(
    store: ChunkStore, ids: list[str]
) -> None:
    """RRF scores come from ranks alone, so they say nothing about relevance."""
    answer = Answerer(build_retriever(store, ids, dense_score=0.62), StubModel()).answer("q")

    assert answer.confidence.retrieval == pytest.approx(0.62)
    assert answer.confidence.both_retrievers_agree is True


def test_phase_three_confidence_dimensions_are_declared_but_unset(
    store: ChunkStore, ids: list[str]
) -> None:
    """Declared so the API contract is settled; None so nothing reports an uncomputed score."""
    confidence = Answerer(build_retriever(store, ids), StubModel()).answer("q").confidence

    assert confidence.citation_coverage is None
    assert confidence.completeness is None
    assert confidence.composite is None


def test_low_retrieval_confidence_refuses_without_calling_the_model(
    store: ChunkStore, ids: list[str]
) -> None:
    """The gate only earns its place if it refuses before spending a request."""
    model = StubModel()
    answerer = Answerer(
        build_retriever(store, ids, dense_score=0.09), model, confidence_threshold=0.30
    )

    answer = answerer.answer("what is the capital of France")

    assert not answer.answered
    assert model.prompts == []
    assert answer.refusal_reason is not None
    assert "below the threshold" in answer.refusal_reason
    assert answer.confidence.retrieval == pytest.approx(0.09)


def test_the_model_can_refuse_even_when_retrieval_looks_confident(
    store: ChunkStore, ids: list[str]
) -> None:
    """The gate cannot see that close-looking chunks lack the fact asked for; the model can."""
    answerer = Answerer(build_retriever(store, ids), StubModel(REFUSAL_SENTINEL))

    answer = answerer.answer("which release removed this")

    assert not answer.answered
    assert answer.refusal_reason == "The model judged the retrieved context insufficient to answer."
    assert answer.text == "I don't know based on the indexed documentation."


@pytest.mark.parametrize(
    "reply", [f"{REFUSAL_SENTINEL}.", f"  {REFUSAL_SENTINEL}  ", "`INSUFFICIENT_CONTEXT`"]
)
def test_the_sentinel_is_recognised_through_stray_punctuation(
    store: ChunkStore, ids: list[str], reply: str
) -> None:
    assert not Answerer(build_retriever(store, ids), StubModel(reply)).answer("q").answered


def test_the_sentinel_inside_a_real_answer_is_not_a_refusal(
    store: ChunkStore, ids: list[str]
) -> None:
    """Otherwise an answer that merely mentions the rule would be discarded."""
    reply = f"The prompt tells the model to reply {REFUSAL_SENTINEL} when unsure [1]."
    answer = Answerer(build_retriever(store, ids), StubModel(reply)).answer("q")

    assert answer.answered
    assert answer.text == reply


def test_a_fabricated_citation_is_reported_and_the_text_is_left_alone(
    store: ChunkStore, ids: list[str]
) -> None:
    reply = "Supported claim [1]. Invented source [9]."
    answer = Answerer(build_retriever(store, ids), StubModel(reply), k=3).answer("q")

    assert answer.unresolved_citations == [9]
    assert answer.text == reply  # never silently rewritten
    assert [c.number for c in answer.citations] == [1]


def test_retrieved_but_uncited_blocks_are_recorded(store: ChunkStore, ids: list[str]) -> None:
    answer = Answerer(build_retriever(store, ids), StubModel("Only this [2]."), k=3).answer("q")

    assert answer.uncited_blocks == [1, 3]


def test_sparse_only_retrieval_has_no_confidence_and_is_not_gated(
    store: ChunkStore, ids: list[str]
) -> None:
    """BM25 scores are not comparable across queries, so there is no honest threshold."""
    retriever = build_retriever(store, ids).ablation("sparse")
    model = StubModel("Answer [1].")

    answer = Answerer(retriever, model).answer("q")

    assert answer.confidence.retrieval is None
    assert answer.answered
    assert model.prompts  # the model was still consulted


def test_a_question_matching_nothing_is_refused(store: ChunkStore, ids: list[str]) -> None:
    model = StubModel()
    retriever = HybridRetriever({"dense": StubIndex([])}, store)

    answer = Answerer(retriever, model).answer("q")

    assert not answer.answered
    assert answer.refusal_reason == "Retrieval returned no chunks for this question."
    assert model.prompts == []


def test_distinct_sources_are_listed_once_in_citation_order(
    store: ChunkStore, ids: list[str]
) -> None:
    answer = Answerer(build_retriever(store, ids), StubModel("A [1] B [2] C [1]."), k=3).answer("q")

    assert answer.sources == ["tutorial/query-params.md > Tutorial > Query Parameters"]


def test_an_empty_question_is_rejected(store: ChunkStore, ids: list[str]) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Answerer(build_retriever(store, ids), StubModel()).answer("   ")


def test_a_non_positive_k_is_rejected(store: ChunkStore, ids: list[str]) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        Answerer(build_retriever(store, ids), StubModel(), k=0)
