"""Calibration, the composite, and the completeness scorer.

The composite's job is to combine measurements without inventing any. Most of these tests
pin the ways a weighted average quietly lies: treating an unmeasured component as zero,
saturating on a raw score that never spans its range, or reporting a partial composite as
if everything had been checked.
"""

from __future__ import annotations

import pytest

from hybridrag.generation.base import Completion
from hybridrag.generation.confidence import (
    CALIBRATION_CEILING,
    CALIBRATION_FLOOR,
    Completeness,
    CompletenessError,
    CompletenessScorer,
    ConfidenceComponent,
    ConfidenceInputs,
    calibrated_retrieval,
    composite_confidence,
    parse_completeness,
)

RETRIEVAL = ConfidenceComponent.RETRIEVAL
COVERAGE = ConfidenceComponent.CITATION_COVERAGE
COMPLETENESS = ConfidenceComponent.COMPLETENESS


class TestCalibration:
    def test_the_gate_threshold_is_the_floor(self) -> None:
        assert calibrated_retrieval(CALIBRATION_FLOOR) == 0.0

    def test_the_measured_ceiling_saturates(self) -> None:
        assert calibrated_retrieval(CALIBRATION_CEILING) == 1.0

    def test_a_cosine_above_the_ceiling_clamps_rather_than_exceeding_one(self) -> None:
        """0.721 was observed on the golden set, so this case is real, not defensive."""
        assert calibrated_retrieval(0.721) == 1.0

    def test_below_the_floor_clamps_to_zero(self) -> None:
        assert calibrated_retrieval(0.14) == 0.0

    def test_the_median_answerable_question_lands_mid_ramp(self) -> None:
        """Measured median was 0.5657; the ramp should put it well inside the range."""
        assert 0.6 < calibrated_retrieval(0.5657) < 0.7  # type: ignore[operator]

    def test_no_dense_component_stays_none(self) -> None:
        """A sparse-only retriever has no cosine, and inventing 0.5 would be a fiction."""
        assert calibrated_retrieval(None) is None


class TestComposite:
    def test_all_three_components_average_equally(self) -> None:
        score = composite_confidence({RETRIEVAL: 1.0, COVERAGE: 0.5, COMPLETENESS: 0.0})
        assert score is not None
        assert score.value == pytest.approx(0.5)
        assert score.is_fully_measured

    def test_a_missing_component_is_renormalised_not_zeroed(self) -> None:
        """Scoring it zero would penalise an answer for a check nobody ran."""
        score = composite_confidence({RETRIEVAL: 1.0, COVERAGE: 1.0, COMPLETENESS: None})
        assert score is not None
        assert score.value == pytest.approx(1.0)
        assert not score.is_fully_measured

    def test_components_record_what_was_measured(self) -> None:
        score = composite_confidence({RETRIEVAL: 0.4, COVERAGE: None, COMPLETENESS: 0.5})
        assert score is not None
        assert score.components == (COMPLETENESS, RETRIEVAL)

    def test_nothing_measured_gives_no_composite(self) -> None:
        assert composite_confidence({RETRIEVAL: None, COVERAGE: None}) is None

    def test_weights_are_equal_by_default(self) -> None:
        """An unfitted weighting makes no tuning claim there is nothing to defend."""
        one = composite_confidence({RETRIEVAL: 1.0, COVERAGE: 0.0})
        two = composite_confidence({RETRIEVAL: 0.0, COVERAGE: 1.0})
        assert one is not None and two is not None
        assert one.value == two.value


class TestConfidenceInputs:
    def test_the_raw_cosine_is_calibrated_before_it_is_combined(self) -> None:
        """Combining raw cosine would let its narrow band dominate two full-range rates."""
        inputs = ConfidenceInputs(retrieval_cosine=0.50, citation_coverage=1.0)
        score = inputs.composite()
        assert score is not None
        assert score.value == pytest.approx((calibrated_retrieval(0.50) + 1.0) / 2)  # type: ignore[operator]

    def test_a_refusal_composite_uses_retrieval_alone(self) -> None:
        score = ConfidenceInputs(retrieval_cosine=0.55).composite()
        assert score is not None
        assert score.components == (RETRIEVAL,)


class TestParseCompleteness:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("COMPLETENESS: complete\nREASON: all of it", Completeness.COMPLETE),
            ("COMPLETENESS: partial\nREASON: half", Completeness.PARTIAL),
            ("**COMPLETENESS:** missing\n**REASON:** none", Completeness.MISSING),
        ],
    )
    def test_readable_replies(self, reply: str, expected: Completeness) -> None:
        level, _ = parse_completeness(reply)
        assert level is expected

    def test_unreadable_reply_raises_rather_than_defaulting(self) -> None:
        with pytest.raises(CompletenessError, match="no COMPLETENESS line"):
            parse_completeness("It covers most of the question I think.")

    def test_unknown_level_raises(self) -> None:
        with pytest.raises(CompletenessError, match="unknown completeness"):
            parse_completeness("COMPLETENESS: mostly\nREASON: hedging")


class StubCompleteness:
    def __init__(self, reply: str) -> None:
        self.model_name = "stub-completeness"
        self.reply = reply
        self.prompts: list[str] = []

    @property
    def fingerprint(self) -> str:
        return f"stub|{self.model_name}"

    def generate(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> Completion:
        self.prompts.append(prompt)
        return Completion(text=self.reply, model=self.model_name, cost_usd=0.0004)


class TestCompletenessScorer:
    def test_levels_map_onto_scores(self) -> None:
        model = StubCompleteness("COMPLETENESS: partial\nREASON: only the first half")
        result = CompletenessScorer(model).score("q?", "a")
        assert result.score == 0.5
        assert result.reason == "only the first half"
        assert result.cost_usd == pytest.approx(0.0004)

    def test_the_scorer_never_sees_the_context(self) -> None:
        """Shown the passages it would grade correctness, which is the judge's job."""
        model = StubCompleteness("COMPLETENESS: complete\nREASON: yes")
        CompletenessScorer(model).score("How do I bind a host?", "Use --host [1].")
        (prompt,) = model.prompts
        assert "How do I bind a host?" in prompt
        assert "Use --host [1]." in prompt
        assert "CONTEXT" not in prompt
