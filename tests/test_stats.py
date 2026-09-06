"""Bootstrap intervals.

Two properties carry the weight. The interval must be reproducible, or a report cannot be
regenerated from its inputs. And the comparison must be *paired* -- if it resampled the
arms independently it would mostly measure which questions each draw contained, which at
n=29 is enough to hide a real difference or invent one.
"""

from __future__ import annotations

import pytest

from hybridrag.evaluation.stats import (
    bootstrap_ci,
    bootstrap_kappa,
    label_agreement,
    paired_delta,
)


class TestInterval:
    def test_the_interval_brackets_the_sample_mean(self) -> None:
        values = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

        interval = bootstrap_ci(values, resamples=2000)

        assert interval.mean == pytest.approx(0.7)
        assert interval.low < interval.mean < interval.high
        assert interval.n == 10

    def test_a_constant_sample_has_no_uncertainty(self) -> None:
        interval = bootstrap_ci([1.0] * 12, resamples=500)

        assert (interval.low, interval.mean, interval.high) == (1.0, 1.0, 1.0)

    def test_the_interval_stays_inside_the_scale(self) -> None:
        """Percentile, not normal-theory: recall cannot exceed 1.0 and must not appear to."""
        interval = bootstrap_ci([1.0] * 19 + [0.0], resamples=4000)

        assert interval.high <= 1.0
        assert interval.low >= 0.0

    def test_the_same_seed_reproduces_the_report(self) -> None:
        values = [0.0, 1.0, 0.5, 0.25, 0.75, 1.0, 0.0]

        assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)

    def test_the_reported_bounds_do_not_depend_on_the_seed(self) -> None:
        """10,000 draws makes the Monte-Carlo error far smaller than the interval itself,
        so no bound in the report is an artefact of which seed was chosen."""
        values = [index / 40 for index in range(40)]

        first = bootstrap_ci(values, seed=7)
        second = bootstrap_ci(values, seed=8)

        assert abs(first.low - second.low) < 0.01
        assert abs(first.high - second.high) < 0.01

    def test_more_questions_narrow_the_interval(self) -> None:
        """Why n=35 was worth arguing about: the interval is the honest cost of a small set."""
        small = bootstrap_ci([1.0, 0.0] * 5, resamples=4000)
        large = bootstrap_ci([1.0, 0.0] * 50, resamples=4000)

        assert (large.high - large.low) < (small.high - small.low)

    def test_an_empty_sample_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty sample"):
            bootstrap_ci([])


class TestPairedComparison:
    def test_a_constant_advantage_has_a_zero_width_interval(self) -> None:
        """Only pairing can produce this: every resample sees the same per-question gap."""
        baseline = [0.1, 0.4, 0.9, 0.3, 0.6, 0.2]
        treatment = [value + 0.1 for value in baseline]

        delta = paired_delta(treatment, baseline, resamples=2000)

        assert delta.difference == pytest.approx(0.1)
        assert delta.low == pytest.approx(0.1)
        assert delta.high == pytest.approx(0.1)
        assert delta.prob_positive == 1.0
        assert delta.significant

    def test_identical_arms_show_no_difference(self) -> None:
        values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]

        delta = paired_delta(values, values, resamples=2000)

        assert delta.difference == 0.0
        assert (delta.low, delta.high) == (0.0, 0.0)
        assert not delta.significant

    def test_a_difference_smaller_than_the_noise_is_not_called_significant(self) -> None:
        """The result this exists to produce honestly: two arms that cannot be separated."""
        baseline = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        treatment = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]

        delta = paired_delta(treatment, baseline, resamples=4000)

        assert delta.low < 0.0 < delta.high
        assert not delta.significant

    def test_mismatched_arms_are_refused_rather_than_compared(self) -> None:
        with pytest.raises(ValueError, match="one score per question"):
            paired_delta([1.0, 0.0], [1.0])


class TestLabelAgreement:
    """Cohen's kappa exists here for one failure: a judge that always says "correct"."""

    def test_a_judge_that_always_agrees_by_default_scores_zero(self) -> None:
        human = ["correct"] * 18 + ["incorrect", "partial"]
        lazy_judge = ["correct"] * 20

        result = label_agreement(lazy_judge, human)

        assert result.raw == pytest.approx(0.9)
        assert result.kappa == pytest.approx(0.0, abs=1e-9)

    def test_perfect_agreement_scores_one(self) -> None:
        labels = ["correct", "partial", "incorrect", "correct"]

        assert label_agreement(labels, labels).kappa == pytest.approx(1.0)

    def test_a_uniform_sample_is_reported_as_perfect_not_undefined(self) -> None:
        """Chance agreement is 1.0 there, so the usual formula is 0/0."""
        assert label_agreement(["correct"] * 5, ["correct"] * 5).kappa == 1.0

    def test_disagreements_are_located_so_they_can_be_reviewed(self) -> None:
        result = label_agreement(
            ["correct", "partial", "correct"], ["correct", "correct", "incorrect"]
        )

        assert result.disagreements == [1, 2]
        assert result.raw == pytest.approx(1 / 3)

    def test_worse_than_chance_is_negative(self) -> None:
        result = label_agreement(["a", "b", "a", "b"], ["b", "a", "b", "a"])

        assert result.kappa < 0.0

    def test_mismatched_label_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="one label per item"):
            label_agreement(["correct"], ["correct", "partial"])


class TestBootstrapKappa:
    def test_the_interval_brackets_the_point_estimate(self) -> None:
        human = ["correct"] * 15 + ["incorrect"] * 5
        judge = ["correct"] * 14 + ["incorrect"] * 6

        interval = bootstrap_kappa(judge, human, resamples=2000)

        assert interval.low <= interval.mean <= interval.high

    def test_a_chance_level_judge_has_an_interval_near_zero(self) -> None:
        """The result that disqualifies a judge outright, and it must be unmistakable."""
        human = ["correct"] * 10 + ["incorrect"] * 10
        judge = ["correct", "incorrect"] * 10

        interval = bootstrap_kappa(judge, human, resamples=2000)

        assert interval.high < 0.5

    def test_mismatched_label_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="one label per item"):
            bootstrap_kappa(["correct"], ["correct", "partial"])
