"""The evaluation harness: pools, arm runs, and the alignment the bootstrap depends on.

Run against stub indexes rather than a built corpus, so the whole grid's plumbing is
covered in CI where there is no index and no API key. What is asserted here is the part
that fails silently in a real run: an ideal pool built from the wrong chunks, a no-answer
question scored on recall, or two arms whose score vectors are ordered differently and
therefore compared question-against-different-question.
"""

from __future__ import annotations

import pytest

from hybridrag.chunk_store import ChunkStore
from hybridrag.evaluation import (
    AnswerSpan,
    GoldenQuestion,
    GoldenSet,
    LocatedSpan,
    QuestionCategory,
    build_pools,
    locate_all,
    run_arm,
)
from hybridrag.evaluation.report import GridResult, provenance, render
from hybridrag.models import Chunk, ChunkingStrategy, Document, SourceFormat
from hybridrag.retrieval import HybridRetriever

GUIDE = "Alpha declares the model. Beta validates it. Gamma filters the output."
OTHER = "Delta serialises the response. Epsilon logs the request."


def document(path: str, text: str) -> Document:
    return Document(
        doc_id=Document.make_id(path),
        relative_path=path,
        source_format=SourceFormat.MARKDOWN,
        text=text,
        content_hash="hash",
    )


DOCUMENTS = {"guide.md": document("guide.md", GUIDE), "other.md": document("other.md", OTHER)}


def chunk(chunk_id: str, path: str, start: int, end: int) -> Chunk:
    body = DOCUMENTS[path].text[start:end]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=Document.make_id(path),
        relative_path=path,
        text=body,
        chunk_index=0,
        strategy=ChunkingStrategy.STRUCTURE,
        token_count=max(1, len(body.split())),
        char_count=end - start,
        start_char=start,
        end_char=end,
    )


CHUNKS = [
    chunk("g1", "guide.md", 0, 26),
    chunk("g2", "guide.md", 26, 45),
    chunk("g3", "guide.md", 45, 70),
    chunk("o1", "other.md", 0, 31),
]

GOLDEN = GoldenSet(
    corpus_ref="test",
    questions=[
        GoldenQuestion(
            question_id="lookup-001",
            category=QuestionCategory.LOOKUP,
            question="What declares the model?",
            answer="Alpha.",
            spans=[AnswerSpan(relative_path="guide.md", quote="Alpha declares the model.")],
            verified=True,
        ),
        GoldenQuestion(
            question_id="lookup-002",
            category=QuestionCategory.LOOKUP,
            question="What filters the output?",
            answer="Gamma.",
            spans=[AnswerSpan(relative_path="guide.md", quote="Gamma filters the output.")],
            verified=True,
        ),
        GoldenQuestion(
            question_id="none-001",
            category=QuestionCategory.NO_ANSWER,
            question="What is the support SLA?",
            answer="",
            spans=[],
            verified=True,
        ),
    ],
)


class StubIndex:
    """A `SearchIndex` returning a scripted ranking, so arms differ by construction."""

    def __init__(self, rankings: dict[str, list[tuple[str, float]]]) -> None:
        self.rankings = rankings

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self.rankings.get(query, [])[:k]


@pytest.fixture
def store() -> ChunkStore:
    store = ChunkStore.in_memory()
    store.add(CHUNKS)
    return store


@pytest.fixture
def located() -> dict[str, list[LocatedSpan]]:
    return locate_all(GOLDEN, DOCUMENTS)


@pytest.fixture
def retriever(store: ChunkStore) -> HybridRetriever:
    dense = StubIndex(
        {
            "What declares the model?": [("g1", 0.81), ("g2", 0.44)],
            "What filters the output?": [("g2", 0.60), ("g3", 0.55)],
            "What is the support SLA?": [("o1", 0.19)],
        }
    )
    sparse = StubIndex(
        {
            "What declares the model?": [("g2", 4.1), ("g1", 3.0)],
            "What filters the output?": [("g3", 5.2)],
            "What is the support SLA?": [("o1", 0.4)],
        }
    )
    return HybridRetriever({"dense": dense, "sparse": sparse}, store)


class TestPools:
    def test_only_chunks_that_touch_a_span_enter_the_ideal(
        self, located: dict[str, list[LocatedSpan]]
    ) -> None:
        pools = build_pools(CHUNKS, located)

        assert [c.chunk_id for c in pools["lookup-001"]] == ["g1"]
        assert [c.chunk_id for c in pools["lookup-002"]] == ["g3"]

    def test_a_question_with_no_covering_chunk_gets_an_empty_pool(
        self, located: dict[str, list[LocatedSpan]]
    ) -> None:
        """Which is what marks a question unreachable rather than merely missed."""
        pools = build_pools([CHUNKS[3]], located)

        assert pools["lookup-001"] == []


class TestArmRun:
    def test_no_answer_questions_are_probed_not_scored(
        self, retriever: HybridRetriever, located: dict[str, list[LocatedSpan]]
    ) -> None:
        result = run_arm(
            "hybrid",
            ChunkingStrategy.STRUCTURE,
            retriever,
            GOLDEN,
            located,
            build_pools(CHUNKS, located),
        )

        assert result.question_ids() == ["lookup-001", "lookup-002"]
        assert [probe.question_id for probe in result.refusals] == ["none-001"]
        assert result.refusals[0].top_dense_score == pytest.approx(0.19)

    def test_scores_stay_in_golden_set_order(
        self, retriever: HybridRetriever, located: dict[str, list[LocatedSpan]]
    ) -> None:
        """The paired bootstrap subtracts arms element by element; order is the contract."""
        pools = build_pools(CHUNKS, located)
        hybrid = run_arm("hybrid", ChunkingStrategy.STRUCTURE, retriever, GOLDEN, located, pools)
        sparse = run_arm(
            "sparse",
            ChunkingStrategy.STRUCTURE,
            retriever.ablation("sparse"),
            GOLDEN,
            located,
            pools,
        )

        assert hybrid.question_ids() == sparse.question_ids()
        assert len(hybrid.series("recall@5")) == len(sparse.series("recall@5")) == 2

    def test_an_arm_without_a_dense_index_records_no_dense_score(
        self, retriever: HybridRetriever, located: dict[str, list[LocatedSpan]]
    ) -> None:
        result = run_arm(
            "sparse",
            ChunkingStrategy.STRUCTURE,
            retriever.ablation("sparse"),
            GOLDEN,
            located,
            build_pools(CHUNKS, located),
        )

        assert all(probe.top_dense_score is None for probe in result.refusals)

    def test_the_arm_names_the_comparison_it_belongs_to(
        self, retriever: HybridRetriever, located: dict[str, list[LocatedSpan]]
    ) -> None:
        result = run_arm(
            "dense",
            ChunkingStrategy.SEMANTIC,
            retriever.ablation("dense"),
            GOLDEN,
            located,
            build_pools(CHUNKS, located),
        )

        assert result.name == "dense/semantic"

    def test_a_question_no_chunk_can_answer_is_flagged_unreachable(
        self, retriever: HybridRetriever, located: dict[str, list[LocatedSpan]]
    ) -> None:
        result = run_arm(
            "hybrid",
            ChunkingStrategy.STRUCTURE,
            retriever,
            GOLDEN,
            located,
            build_pools([CHUNKS[3]], located),
        )

        assert set(result.unreachable) == {"lookup-001", "lookup-002"}

    def test_an_unknown_metric_is_refused_rather_than_scored_as_zero(
        self, retriever: HybridRetriever, located: dict[str, list[LocatedSpan]]
    ) -> None:
        result = run_arm(
            "hybrid",
            ChunkingStrategy.STRUCTURE,
            retriever,
            GOLDEN,
            located,
            build_pools(CHUNKS, located),
        )

        with pytest.raises(ValueError, match="unknown metric"):
            result.series("recall@7")


class TestReport:
    def test_the_report_renders_from_results_alone(
        self, retriever: HybridRetriever, located: dict[str, list[LocatedSpan]]
    ) -> None:
        """Every number in the report is generated; nothing is written by hand."""
        pools = build_pools(CHUNKS, located)
        arms = [
            run_arm(
                name,
                ChunkingStrategy.STRUCTURE,
                retriever if name == "hybrid" else retriever.ablation(name),
                GOLDEN,
                located,
                pools,
            )
            for name in ("hybrid", "dense", "sparse")
        ]
        grid = GridResult(
            provenance=provenance(
                GOLDEN,
                documents=2,
                embedding_model="stub",
                chunk_tokens=512,
                chunk_overlap_tokens=64,
                semantic_percentile=95.0,
                rank_constant=60,
                candidates=50,
                depth=50,
                token_budget=2000,
                min_ratio=1.0,
            ),
            arms=arms,
        )

        markdown = render(grid, GOLDEN)

        assert "hybrid/structure" in markdown
        assert "Refusal calibration" in markdown
        # The provenance header must carry what produced the numbers.
        assert "corpus `test`" in markdown
        assert "2 verified, of which **2 are scored here**" not in markdown  # 3 verified, 2 scored
        assert "3 verified, of which **2 are scored here**" in markdown
