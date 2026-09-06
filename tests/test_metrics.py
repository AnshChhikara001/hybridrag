"""Tier-1 retrieval metrics.

The load-bearing property is that `coverage` and the category rules in `is_hit` never
disagree: one is the graded form of the other, and if they drift then nDCG rewards a
ranking the recall table calls a miss. The rest guards against metrics that flatter a
chunking strategy for its shape rather than its retrieval -- overlap counted twice, large
chunks paid for being large, an ideal computed from the arm's own results.
"""

from __future__ import annotations

import pytest

from hybridrag.evaluation.golden import AnswerSpan, GoldenQuestion, QuestionCategory, is_hit
from hybridrag.evaluation.metrics import (
    budget_prefix,
    coverage,
    evaluate_question,
    ideal_gains,
    is_covered,
    marginal_gains,
)
from hybridrag.models import Chunk, ChunkingStrategy, Document, SourceFormat

GUIDE = "Alpha declares the model. Beta validates it. Gamma filters the output."
OTHER = "Delta serialises the response. Epsilon logs the request. Zeta closes it."


def document(path: str = "guide.md", text: str = GUIDE) -> Document:
    return Document(
        doc_id=Document.make_id(path),
        relative_path=path,
        source_format=SourceFormat.MARKDOWN,
        text=text,
        content_hash="hash",
    )


CORPUS = {"guide.md": document(), "other.md": document("other.md", OTHER)}


def chunk(start: int, end: int, path: str = "guide.md", tokens: int = 10) -> Chunk:
    body = CORPUS[path].text[start:end]
    return Chunk(
        chunk_id=f"{path}:{start}-{end}",
        doc_id=Document.make_id(path),
        relative_path=path,
        text=body,
        chunk_index=0,
        strategy=ChunkingStrategy.STRUCTURE,
        token_count=tokens,
        char_count=end - start,
        start_char=start,
        end_char=end,
    )


def question(
    category: QuestionCategory = QuestionCategory.LOOKUP,
    spans: list[AnswerSpan] | None = None,
) -> GoldenQuestion:
    return GoldenQuestion(
        question_id="q001",
        category=category,
        question="What declares the model?",
        answer="Alpha does.",
        spans=spans or [AnswerSpan(relative_path="guide.md", quote="Alpha declares the model.")],
        verified=True,
    )


MULTI_HOP = question(
    QuestionCategory.MULTI_HOP,
    [
        AnswerSpan(relative_path="guide.md", quote="Alpha declares the model."),
        AnswerSpan(relative_path="other.md", quote="Delta serialises the response."),
    ],
)
AMBIGUOUS = question(
    QuestionCategory.AMBIGUOUS,
    [
        AnswerSpan(relative_path="guide.md", quote="Alpha declares the model."),
        AnswerSpan(relative_path="guide.md", quote="Gamma filters the output."),
    ],
)


class TestAgreementWithTheCategoryRules:
    """`coverage` is the graded form of `is_hit`; a disagreement is a scoring bug."""

    @pytest.mark.parametrize(
        "golden, chunks",
        [
            (question(), [chunk(0, 25)]),
            (question(), [chunk(0, 12)]),
            (question(), []),
            (MULTI_HOP, [chunk(0, 25), chunk(0, 30, "other.md")]),
            (MULTI_HOP, [chunk(0, 25)]),
            (AMBIGUOUS, [chunk(44, 70)]),
            (AMBIGUOUS, [chunk(26, 43)]),
        ],
    )
    def test_the_graded_and_binary_predicates_agree(
        self, golden: GoldenQuestion, chunks: list[Chunk]
    ) -> None:
        located = golden.locate(CORPUS)

        assert is_covered(golden.category, located, chunks) == is_hit(golden, located, chunks)


class TestRecall:
    def test_a_span_split_across_two_chunks_needs_both(self) -> None:
        golden = question()
        located = golden.locate(CORPUS)

        assert not is_covered(golden.category, located, [chunk(0, 12)])
        assert is_covered(golden.category, located, [chunk(0, 12), chunk(12, 30)])

    def test_deeper_cutoffs_can_only_help(self) -> None:
        golden = question()
        ranked = [chunk(44, 70), chunk(26, 43), chunk(0, 25)]

        scored = evaluate_question(golden, golden.locate(CORPUS), ranked, ranked)

        assert scored.hits == {1: False, 3: True, 5: True, 10: True}
        assert scored.first_hit_rank == 3
        assert scored.reciprocal_rank == pytest.approx(1 / 3)


class TestFirstHitRank:
    def test_multi_hop_becomes_answerable_where_the_second_document_arrives(self) -> None:
        """The generalisation that matters: one of two documents is not a hit at rank 1."""
        located = MULTI_HOP.locate(CORPUS)
        ranked = [chunk(0, 25), chunk(26, 43), chunk(0, 30, "other.md")]

        scored = evaluate_question(MULTI_HOP, located, ranked, ranked)

        assert scored.first_hit_rank == 3

    def test_a_hit_below_the_reporting_horizon_scores_zero(self) -> None:
        """A span at rank 34 is retrieved but never reaches the generator's context."""
        golden = question()
        ranked = [chunk(44, 70)] * 12 + [chunk(0, 25)]

        scored = evaluate_question(golden, golden.locate(CORPUS), ranked, [chunk(0, 25)])

        assert scored.first_hit_rank == 13
        assert scored.reciprocal_rank == 0.0


class TestGradedGains:
    def test_a_duplicate_chunk_adds_nothing(self) -> None:
        """Otherwise the overlapping fixed-size strategy is paid twice for one passage."""
        golden = question()
        located = golden.locate(CORPUS)

        gains = marginal_gains(golden.category, located, [chunk(0, 25), chunk(0, 25)])

        assert gains[0] == pytest.approx(1.0)
        assert gains[1] == pytest.approx(0.0)

    def test_gains_sum_to_the_coverage_reached(self) -> None:
        golden = question()
        located = golden.locate(CORPUS)
        ranked = [chunk(0, 12), chunk(12, 30)]

        assert sum(marginal_gains(golden.category, located, ranked)) == pytest.approx(1.0)

    def test_the_ideal_orders_by_what_each_chunk_adds(self) -> None:
        golden = question()
        located = golden.locate(CORPUS)
        pool = [chunk(0, 12), chunk(0, 25), chunk(44, 70)]

        gains = ideal_gains(golden.category, located, pool, 10)

        # The whole span first, then nothing left worth taking.
        assert gains == pytest.approx([1.0])


class TestNdcg:
    def test_the_ideal_ordering_scores_one(self) -> None:
        golden = question()
        ranked = pool = [chunk(0, 25), chunk(44, 70)]

        assert evaluate_question(golden, golden.locate(CORPUS), ranked, pool).ndcg == pytest.approx(
            1.0
        )

    def test_burying_the_answer_below_a_distractor_costs(self) -> None:
        golden = question()
        pool = [chunk(0, 25)]
        ranked = [chunk(44, 70), chunk(0, 25)]

        scored = evaluate_question(golden, golden.locate(CORPUS), ranked, pool)

        assert scored.ndcg == pytest.approx(1 / 1.5849625007211563, rel=1e-6)

    def test_the_ideal_comes_from_the_index_not_from_the_results(self) -> None:
        """Half a multi-hop answer, ranked perfectly, is still half an answer."""
        located = MULTI_HOP.locate(CORPUS)
        pool = [chunk(0, 25), chunk(0, 30, "other.md")]
        ranked = [chunk(0, 25)]

        scored = evaluate_question(MULTI_HOP, located, ranked, pool)

        assert scored.ndcg < 1.0
        assert scored.reachable

    def test_a_span_no_chunk_can_carry_is_reported_as_unreachable(self) -> None:
        """A chunking result, not a retrieval one, and it must be visible as such."""
        golden = question()
        ranked = [chunk(44, 70)]

        scored = evaluate_question(golden, golden.locate(CORPUS), ranked, pool=[])

        assert scored.ndcg == 0.0
        assert scored.reachable is False


class TestTokenBudget:
    def test_the_prefix_ends_at_the_first_chunk_that_would_overflow(self) -> None:
        ranked = [chunk(0, 12, tokens=800), chunk(12, 30, tokens=800), chunk(44, 70, tokens=800)]

        taken, tokens = budget_prefix(ranked, budget=2000)

        assert [c.chunk_id for c in taken] == ["guide.md:0-12", "guide.md:12-30"]
        assert tokens == 1600

    def test_a_large_chunk_wins_on_recall_at_k_and_loses_on_budget(self) -> None:
        """D14's whole point: Recall@k pays a strategy for emitting bigger chunks."""
        golden = question()
        located = golden.locate(CORPUS)
        oversized = [chunk(0, 70, tokens=2500)]

        scored = evaluate_question(golden, located, oversized, oversized)

        assert scored.hits[1] is True
        assert scored.hit_at_budget is False
        assert scored.budget_chunks == 0

    def test_overlapping_text_is_charged_to_the_budget_every_time(self) -> None:
        """The context window pays for a repeated passage as many times as it appears."""
        ranked = [chunk(0, 25, tokens=1200), chunk(0, 25, tokens=1200)]

        _, tokens = budget_prefix(ranked, budget=2000)

        assert tokens == 1200


class TestRefusals:
    def test_a_no_answer_question_has_no_retrieval_score(self) -> None:
        golden = GoldenQuestion(
            question_id="n001",
            category=QuestionCategory.NO_ANSWER,
            question="What is FastAPI's enterprise SLA?",
            answer="",
            spans=[],
            verified=True,
        )

        with pytest.raises(ValueError, match="no retrieval ground truth"):
            evaluate_question(golden, [], [chunk(0, 25)], [])

    def test_coverage_of_nothing_is_zero(self) -> None:
        assert coverage(QuestionCategory.LOOKUP, [], [chunk(0, 25)]) == 0.0
