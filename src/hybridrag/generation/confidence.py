"""The composite answer confidence score, and the calibration behind it.

Three dimensions, kept visible rather than collapsed into one opaque number, because they
fail independently and a reader who cannot see which one dropped cannot act on the score:

* **retrieval** -- how close the best chunk was, calibrated onto 0-1.
* **citation coverage** -- what share of the answer's claims carry a citation that holds.
* **completeness** -- whether the answer addressed every part of the question.

**Why retrieval needs calibrating.** Raw cosine on this corpus occupies a narrow band, so
averaging it directly against two rates that genuinely span 0-1 would let it drag every
composite toward the middle regardless of what happened. Measured over the 35 golden
questions against the fixed-size index, the top cosine for answerable questions ran
0.3606-0.7210, median 0.5657, 90th percentile 0.6636. The ramp therefore runs from the
refusal threshold (0.30, below which nothing is answered anyway) to 0.70, which sits
between that 90th percentile and the observed maximum: almost every answerable question
lands in the upper half of the ramp and exactly one saturates it.

**What that same measurement says about the gate.** The six unanswerable questions scored
0.3788-0.6396 -- inside the answerable range, not below it. The threshold in `answerer.py`
was calibrated against *out-of-domain* questions, which score 0.14-0.24 and are caught
easily; a question that is in-domain but simply unanswered by the corpus retrieves
confident-looking neighbours and sails through. Retrieval confidence alone is therefore a
poor guide to whether an answer can be trusted on this corpus, which is the argument for
combining it with evidence gathered *after* generation rather than the argument against
measuring it.

**Weights are equal and unfitted.** With 29 answerable questions and a handful of failures
there is nothing to fit them on that would not simply memorise this corpus. Equal weights
over whichever components are present make no tuning claim to defend, and the composite's
worth is then a measured quantity -- whether it actually separates correct answers from
wrong ones -- rather than an assertion.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, Field, NonNegativeFloat

from hybridrag.generation.base import LanguageModel

# Both measured over the golden set; the module docstring shows the arithmetic.
CALIBRATION_FLOOR = 0.30
CALIBRATION_CEILING = 0.70


class ConfidenceComponent(StrEnum):
    """The dimensions a composite may be built from."""

    RETRIEVAL = "retrieval"
    CITATION_COVERAGE = "citation_coverage"
    COMPLETENESS = "completeness"


DEFAULT_WEIGHTS: Mapping[ConfidenceComponent, float] = {
    ConfidenceComponent.RETRIEVAL: 1.0,
    ConfidenceComponent.CITATION_COVERAGE: 1.0,
    ConfidenceComponent.COMPLETENESS: 1.0,
}


def calibrated_retrieval(cosine: float | None) -> float | None:
    """Map a raw top cosine onto 0-1 through the measured ramp.

    None in, None out: a sparse-only retriever has no cosine, and inventing a 0.5 for it
    would be a confidence number that moves for no reason.
    """
    if cosine is None:
        return None
    span = CALIBRATION_CEILING - CALIBRATION_FLOOR
    return min(1.0, max(0.0, (cosine - CALIBRATION_FLOOR) / span))


class Completeness(StrEnum):
    """How much of the question the answer actually addressed."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


COMPLETENESS_SCORES: Mapping[Completeness, float] = {
    Completeness.COMPLETE: 1.0,
    Completeness.PARTIAL: 0.5,
    Completeness.MISSING: 0.0,
}


class CompletenessError(ValueError):
    """The completeness reply could not be read."""


COMPLETENESS_SYSTEM_PROMPT = """\
You check whether an answer addresses everything a question asked. You are given the \
question and the assistant's answer, and nothing else.

Judge scope only, never correctness. You do not know the right answer and must not guess \
at it: an answer that is wrong but addresses every part of the question is complete.

* complete -- every part of the question is addressed.
* partial  -- some part is addressed and some part is left unanswered, including when the \
answer says outright that it cannot cover that part.
* missing  -- the question is not addressed at all, or the answer declines entirely.

Reply in exactly this form and nothing else:

COMPLETENESS: complete|partial|missing
REASON: one sentence
"""

_COMPLETENESS_LINE = re.compile(
    r"^\s*\**\s*COMPLETENESS\s*\**\s*:\s*\**\s*(\w+)", re.IGNORECASE | re.MULTILINE
)
_REASON_LINE = re.compile(r"^\s*\**\s*REASON\s*\**\s*:\s*(.+)", re.IGNORECASE | re.MULTILINE)


def parse_completeness(text: str) -> tuple[Completeness, str]:
    """Read a completeness level out of a reply, raising rather than defaulting."""
    match = _COMPLETENESS_LINE.search(text)
    if match is None:
        raise CompletenessError(f"no COMPLETENESS line in the reply. Began: {text[:120]!r}")
    raw = match.group(1).strip().lower()
    try:
        level = Completeness(raw)
    except ValueError as error:
        raise CompletenessError(
            f"unknown completeness {raw!r}; expected one of {[c.value for c in Completeness]}"
        ) from error
    reason = _REASON_LINE.search(text)
    return level, reason.group(1).strip() if reason else ""


class CompletenessResult(BaseModel):
    """One completeness judgement and what it cost."""

    level: Completeness
    reason: str = ""
    model: str = ""
    cost_usd: NonNegativeFloat = 0.0
    cached: bool = False

    @property
    def score(self) -> float:
        return COMPLETENESS_SCORES[self.level]


class CompletenessScorer:
    """Asks a model whether an answer covered the whole question.

    Deliberately blind to the context and to the reference answer. Shown the retrieved
    passages it would start grading correctness, which is the Tier-2 judge's job and a
    different measurement; shown only the question and the answer it can judge scope,
    which is the one thing a reader cannot check at a glance on a long answer.
    """

    def __init__(self, model: LanguageModel, *, max_output_tokens: int | None = None) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens

    def score(self, question: str, answer: str) -> CompletenessResult:
        completion = self.model.generate(
            f"QUESTION\n{question.strip()}\n\nANSWER\n{answer.strip()}\n",
            system=COMPLETENESS_SYSTEM_PROMPT,
            max_output_tokens=self.max_output_tokens,
        )
        level, reason = parse_completeness(completion.text)
        return CompletenessResult(
            level=level,
            reason=reason,
            model=completion.model,
            cost_usd=completion.cost_usd,
            cached=completion.cached,
        )


class CompositeScore(BaseModel):
    """A composite and the components it was actually built from.

    `components` is not decoration. A composite of retrieval alone and a composite of all
    three are different measurements, and a dashboard that shows 0.72 for both is lying by
    omission about how much was checked.
    """

    value: float
    components: tuple[ConfidenceComponent, ...]

    @property
    def is_fully_measured(self) -> bool:
        return len(self.components) == len(ConfidenceComponent)


def composite_confidence(
    values: Mapping[ConfidenceComponent, float | None],
    weights: Mapping[ConfidenceComponent, float] = DEFAULT_WEIGHTS,
) -> CompositeScore | None:
    """Weighted mean over the components that were measured. None when none were.

    Weights are renormalised over what is present rather than treating an unmeasured
    component as zero. Scoring it zero would mean an answer loses confidence for a check
    nobody ran, so the cheap path would look worse than the thorough one on identical
    answers -- a number that measures the caller's configuration, not the answer.
    """
    present = {
        component: value
        for component, value in values.items()
        if value is not None and weights.get(component, 0.0) > 0.0
    }
    if not present:
        return None
    total = sum(weights[component] for component in present)
    return CompositeScore(
        value=sum(weights[component] * value for component, value in present.items()) / total,
        components=tuple(sorted(present, key=lambda component: component.value)),
    )


class ConfidenceInputs(BaseModel):
    """Everything the composite reads, gathered in one place for the API to serialise."""

    retrieval_cosine: float | None = None
    citation_coverage: float | None = None
    citation_coverage_verified: bool = Field(
        default=False,
        description="True when coverage came from claim-level verification; False when it "
        "is the free structural check, which asks only whether a claim was attributed.",
    )
    completeness: float | None = None

    def composite(self) -> CompositeScore | None:
        return composite_confidence(
            {
                ConfidenceComponent.RETRIEVAL: calibrated_retrieval(self.retrieval_cosine),
                ConfidenceComponent.CITATION_COVERAGE: self.citation_coverage,
                ConfidenceComponent.COMPLETENESS: self.completeness,
            }
        )
