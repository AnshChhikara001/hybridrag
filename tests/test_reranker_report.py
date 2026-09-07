"""The reranker report: paired comparison, distractor rate, and the pairing contract.

Built from hand-constructed `ArmResult`s rather than a live retriever -- `render()`'s own
suite (`test_harness.py`) already covers turning retrieval into `QuestionMetrics`; what is
specific to this report is the baseline/reranked pairing and its verdict wording, which is
best tested against scores chosen to force a specific outcome.
"""

from __future__ import annotations

from hybridrag.evaluation.golden import QuestionCategory
from hybridrag.evaluation.harness import ArmResult
from hybridrag.evaluation.metrics import QuestionMetrics
from hybridrag.evaluation.report import Provenance, RerankerRun, render_reranker_report
from hybridrag.models import ChunkingStrategy


def question(question_id: str, *, recall5: bool, ndcg: float, rr: float) -> QuestionMetrics:
    return QuestionMetrics(
        question_id=question_id,
        category=QuestionCategory.LOOKUP,
        hits={1: recall5, 3: recall5, 5: recall5, 10: recall5},
        first_hit_rank=1 if recall5 else None,
        reciprocal_rank=rr,
        ndcg=ndcg,
        coverage=1.0 if recall5 else 0.0,
        hit_at_budget=recall5,
        budget_chunks=1,
        budget_tokens=100,
        reachable=True,
    )


def arm(
    name: str, strategy: ChunkingStrategy, questions: list[QuestionMetrics], paths: list[str]
) -> ArmResult:
    return ArmResult(
        retriever=name,
        strategy=strategy,
        questions=questions,
        top5_paths={q.question_id: paths for q in questions},
    )


def provenance(*, resamples: int = 500) -> Provenance:
    return Provenance(
        generated_at="2026-09-07T00:00:00+00:00",
        git_sha="abc1234",
        git_dirty=False,
        corpus_ref="test",
        embedding_model="stub",
        chunk_tokens=512,
        chunk_overlap_tokens=64,
        semantic_percentile=95.0,
        rank_constant=60,
        candidates=50,
        depth=50,
        token_budget=2000,
        min_ratio=1.0,
        resamples=resamples,
    )


# Every question worse under the baseline than under reranking, so the paired bootstrap
# has no way to land anywhere but a positive, zero-excluding interval.
BASELINE_QUESTIONS = [question(f"q{i}", recall5=False, ndcg=0.1, rr=0.1) for i in range(10)]
RERANKED_QUESTIONS = [question(f"q{i}", recall5=True, ndcg=0.9, rr=0.9) for i in range(10)]


class TestPairingContract:
    def test_mismatched_arm_counts_are_refused(self) -> None:
        run = RerankerRun(
            provenance=provenance(),
            baselines=[
                arm("hybrid", ChunkingStrategy.FIXED, BASELINE_QUESTIONS, ["guide.md"]),
                arm("hybrid", ChunkingStrategy.STRUCTURE, BASELINE_QUESTIONS, ["guide.md"]),
            ],
            reranked=[arm("reranked", ChunkingStrategy.FIXED, RERANKED_QUESTIONS, ["guide.md"])],
        )

        try:
            render_reranker_report(run)
        except ValueError as error:
            assert "must pair one-to-one" in str(error)
        else:
            raise AssertionError("expected a ValueError")

    def test_a_strategy_mismatch_at_the_same_position_is_refused(self) -> None:
        run = RerankerRun(
            provenance=provenance(),
            baselines=[arm("hybrid", ChunkingStrategy.FIXED, BASELINE_QUESTIONS, ["guide.md"])],
            reranked=[
                arm("reranked", ChunkingStrategy.STRUCTURE, RERANKED_QUESTIONS, ["guide.md"])
            ],
        )

        try:
            render_reranker_report(run)
        except ValueError as error:
            assert "strategies must match" in str(error)
        else:
            raise AssertionError("expected a ValueError")


class TestReport:
    def test_a_clear_win_is_reported_as_separated(self) -> None:
        run = RerankerRun(
            provenance=provenance(),
            baselines=[arm("hybrid", ChunkingStrategy.FIXED, BASELINE_QUESTIONS, ["guide.md"])],
            reranked=[arm("reranked", ChunkingStrategy.FIXED, RERANKED_QUESTIONS, ["guide.md"])],
            distractor_rates={"hybrid/fixed": 0.09, "reranked/fixed": 0.02},
        )

        markdown = render_reranker_report(run)

        assert "fixed" in markdown
        assert "9.0%" in markdown and "2.0%" in markdown
        assert "**not separated from zero**" not in markdown
        assert "separated" in markdown

    def test_identical_arms_are_reported_as_not_separated(self) -> None:
        run = RerankerRun(
            provenance=provenance(),
            baselines=[arm("hybrid", ChunkingStrategy.FIXED, BASELINE_QUESTIONS, ["guide.md"])],
            reranked=[arm("reranked", ChunkingStrategy.FIXED, BASELINE_QUESTIONS, ["guide.md"])],
        )

        markdown = render_reranker_report(run)

        assert "**not separated from zero**" in markdown

    def test_a_missing_distractor_rate_does_not_crash_the_render(self) -> None:
        """A caller that forgets a rate should see a visible gap, not a KeyError mid-report."""
        run = RerankerRun(
            provenance=provenance(),
            baselines=[arm("hybrid", ChunkingStrategy.FIXED, BASELINE_QUESTIONS, ["guide.md"])],
            reranked=[arm("reranked", ChunkingStrategy.FIXED, RERANKED_QUESTIONS, ["guide.md"])],
        )

        markdown = render_reranker_report(run)

        assert "nan%" in markdown
