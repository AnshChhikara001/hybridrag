"""Tier-2 scoring and failure attribution.

The attribution is what carries the weight. "34% of answers are wrong" tells nobody what
to fix; "the evidence was retrieved and the generator ignored it" does. Getting the order
of those checks wrong would hide retrieval failures behind prompt failures, so the order
is pinned by tests.
"""

from __future__ import annotations

import pytest

from hybridrag.evaluation.golden import QuestionCategory
from hybridrag.evaluation.judge import Verdict
from hybridrag.evaluation.quality import (
    AnswerRecord,
    FailureKind,
    QualityArm,
    classify,
    reference_for,
)
from hybridrag.models import ChunkingStrategy


def record(**kwargs: object) -> AnswerRecord:
    defaults: dict[str, object] = {
        "question_id": "lookup-001",
        "category": QuestionCategory.LOOKUP,
        "answered": True,
        "verdict": Verdict.CORRECT,
        "grounded": True,
    }
    defaults.update(kwargs)
    return AnswerRecord.model_validate(defaults)


class TestAttribution:
    def test_a_correct_answer_has_no_failure(self) -> None:
        assert (
            classify(
                Verdict.CORRECT,
                answered=True,
                category=QuestionCategory.LOOKUP,
                span_retrieved=False,
            )
            is None
        )

    def test_retrieval_is_blamed_before_generation(self) -> None:
        """A refusal on a question whose evidence never arrived is a retrieval failure.

        Calling it a wrong refusal would file a retrieval problem under prompt problems,
        which is the specific mistake this ordering exists to prevent.
        """
        assert (
            classify(
                Verdict.INCORRECT,
                answered=False,
                category=QuestionCategory.LOOKUP,
                span_retrieved=False,
            )
            is FailureKind.RETRIEVAL_MISS
        )

    def test_a_refusal_with_the_evidence_present_is_a_wrong_refusal(self) -> None:
        assert (
            classify(
                Verdict.INCORRECT,
                answered=False,
                category=QuestionCategory.LOOKUP,
                span_retrieved=True,
            )
            is FailureKind.WRONG_REFUSAL
        )

    def test_a_wrong_answer_with_the_evidence_present_blames_generation(self) -> None:
        assert (
            classify(
                Verdict.PARTIAL,
                answered=True,
                category=QuestionCategory.LOOKUP,
                span_retrieved=True,
            )
            is FailureKind.IGNORED_CONTEXT
        )

    def test_answering_an_unanswerable_question_is_a_missing_refusal(self) -> None:
        assert (
            classify(
                Verdict.INCORRECT,
                answered=True,
                category=QuestionCategory.NO_ANSWER,
                span_retrieved=None,
            )
            is FailureKind.MISSING_REFUSAL
        )

    def test_every_incorrect_answer_lands_in_some_column(self) -> None:
        """Otherwise the failure table stops summing to the number of errors."""
        for category in QuestionCategory:
            for answered in (True, False):
                for retrieved in (True, False, None):
                    assert (
                        classify(
                            Verdict.INCORRECT,
                            answered=answered,
                            category=category,
                            span_retrieved=retrieved,
                        )
                        is not None
                    )


class TestArm:
    def _arm(self) -> QualityArm:
        return QualityArm(
            retriever="hybrid",
            strategy=ChunkingStrategy.FIXED,
            judge_model="judge",
            generation_model="generator",
            records=[
                record(question_id="a", verdict=Verdict.CORRECT, modelled_cost_usd=0.0002),
                record(question_id="b", verdict=Verdict.PARTIAL, modelled_cost_usd=0.0002),
                record(
                    question_id="c",
                    verdict=Verdict.INCORRECT,
                    modelled_cost_usd=0.0002,
                    fabricated_citations=1,
                ),
            ],
        )

    def test_partial_credit_is_reported_separately_not_blended(self) -> None:
        arm = self._arm()

        assert arm.series("correct") == [1.0, 0.0, 0.0]
        assert arm.series("correct_or_partial") == [1.0, 1.0, 0.0]

    def test_a_fabricated_citation_fails_the_deterministic_check(self) -> None:
        assert self._arm().series("cited_honestly") == [1.0, 1.0, 0.0]

    def test_cost_per_answer_survives_a_fully_cached_run(self) -> None:
        """Averaging what a run *paid* reads $0.000000 when everything came from cache."""
        arm = self._arm()
        for entry in arm.records:
            entry.cached = True
            entry.generation_cost_usd = 0.0

        assert arm.total_cost_usd == 0.0
        assert arm.modelled_cost_per_answer == pytest.approx(0.0002)
        assert arm.live_answers == 0

    def test_an_unknown_metric_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            self._arm().series("helpfulness")


class TestReference:
    def test_an_unanswerable_question_has_an_empty_reference(self) -> None:
        """Which is how the judge is told that declining is the correct behaviour."""
        from hybridrag.evaluation.golden import GoldenQuestion

        question = GoldenQuestion(
            question_id="n1",
            category=QuestionCategory.NO_ANSWER,
            question="What is the SLA?",
            answer="",
            spans=[],
        )

        assert reference_for(question) == ""
