"""Uncertainty for the evaluation tables: percentile bootstrap, paired for comparisons.

The golden set has 29 answerable questions. At that size a difference of five points
between two arms is roughly one question changing its mind, so a table of bare means
invites a claim the data does not support. Every reported number therefore carries an
interval, and every *comparison* is bootstrapped **paired** -- the same resampled
questions scored under both arms.

Pairing is the part that matters. Question difficulty dominates the variance here: a
multi-hop question is hard for every arm, so resampling the arms independently measures
mostly which questions each draw happened to include. Resampling once and scoring both
arms on that draw cancels that shared term, leaving the arms' actual disagreement. In the
limiting case where one arm beats the other on every single question, the paired interval
correctly excludes zero while an unpaired one need not.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field, PositiveInt

# 10,000 draws puts the Monte-Carlo error on a 95% percentile bound well below the width
# the interval itself reports at n=29, so the numbers do not wobble between runs.
DEFAULT_RESAMPLES = 10_000

# Fixed so a report is reproducible from its inputs alone. Recorded in every report header.
DEFAULT_SEED = 20260906


class Interval(BaseModel):
    """A point estimate and its percentile bootstrap interval."""

    mean: float
    low: float
    high: float
    n: PositiveInt
    confidence: float = Field(gt=0.0, lt=1.0)
    resamples: PositiveInt

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"


class Delta(BaseModel):
    """A paired difference between two arms, with its interval."""

    difference: float = Field(description="mean(treatment) - mean(baseline)")
    low: float
    high: float
    prob_positive: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of resamples where the treatment led. Reported instead of a "
        "p-value: it answers the question actually being asked, and needs no null model.",
    )
    n: PositiveInt
    confidence: float = Field(gt=0.0, lt=1.0)
    resamples: PositiveInt

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero -- the only claim this design licenses."""
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        mark = "*" if self.significant else " "
        return f"{self.difference:+.3f} [{self.low:+.3f}, {self.high:+.3f}]{mark}"


def _resample_indices(n: int, resamples: int, seed: int) -> np.typing.NDArray[np.int64]:
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Percentile bootstrap interval for the mean of `values`.

    Percentile rather than normal-theory: these are means of bounded, heavily skewed
    quantities (a recall value is 0 or 1, an nDCG piles up at 1.0), so a symmetric
    interval would run past the ends of the scale.
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    means = data[_resample_indices(data.size, resamples, seed)].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(means, [100.0 * tail, 100.0 * (1.0 - tail)])
    return Interval(
        mean=float(data.mean()),
        low=float(low),
        high=float(high),
        n=int(data.size),
        confidence=confidence,
        resamples=resamples,
    )


def paired_delta(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Delta:
    """Bootstrap the difference between two arms scored on the *same* questions.

    Both sequences must be aligned question by question; the caller owns that ordering,
    and getting it wrong silently compares unrelated questions, so the harness builds
    both from one iteration over the golden set.
    """
    left = np.asarray(treatment, dtype=float)
    right = np.asarray(baseline, dtype=float)
    if left.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if left.size != right.size:
        raise ValueError(
            f"paired comparison needs one score per question on both arms, got "
            f"{left.size} and {right.size}"
        )

    indices = _resample_indices(left.size, resamples, seed)
    differences = (left[indices] - right[indices]).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(differences, [100.0 * tail, 100.0 * (1.0 - tail)])
    return Delta(
        difference=float(left.mean() - right.mean()),
        low=float(low),
        high=float(high),
        prob_positive=float((differences > 0).mean()),
        n=int(left.size),
        confidence=confidence,
        resamples=resamples,
    )
