"""What an answer is.

Declared in full, including the dimensions filled at different stages of the pipeline,
because this shape is what the FastAPI response model serialises and what the evaluation
harness grades. A field added later means changing an API contract and re-running every
stored result, so the contract is settled here and anything not computed on a given path
stays explicitly `None` rather than being defaulted to a number nobody measured.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveInt

from hybridrag.retrieval import RetrievedChunk

NO_ANSWER_TEXT = "I don't know based on the indexed documentation."

# Enough to point a reader at the right page without turning a refusal into a reading list.
MAX_DOCUMENTS_TO_CHECK = 3


class Citation(BaseModel):
    """One `[n]` in the answer, resolved back to the chunk it points at."""

    number: PositiveInt = Field(description="The bracketed number as written in the answer.")
    chunk_id: str
    relative_path: str
    heading_path: tuple[str, ...] = ()

    @property
    def source(self) -> str:
        """Human-readable provenance: the file, then the heading breadcrumb."""
        heading = " > ".join(self.heading_path)
        return f"{self.relative_path}{' > ' + heading if heading else ''}"


class CitationReport(BaseModel):
    """The outcome of checking an answer's citations against its context."""

    citations: list[Citation] = Field(default_factory=list)
    unresolved: list[int] = Field(
        default_factory=list,
        description="Cited numbers with no such block. Each one is a fabricated source.",
    )
    uncited_blocks: list[int] = Field(
        default_factory=list,
        description="Blocks retrieved but never cited. Not an error; a diagnostic.",
    )

    @property
    def is_structurally_sound(self) -> bool:
        return not self.unresolved


class RefusalKind(StrEnum):
    """Which of the refusal paths fired.

    Kept apart because they are different failures with different fixes. A gate refusal
    says the corpus has nothing close; a model refusal says the corpus had something close
    that did not contain the answer. Collapsing them would hide a retrieval problem inside
    a generation problem, which is the confusion the failure attribution in Phase 4 exists
    to prevent.
    """

    NO_RESULTS = "no_results"
    LOW_CONFIDENCE = "low_confidence"
    MODEL_DECLINED = "model_declined"


class NearMiss(BaseModel):
    """A page retrieval did surface, and how close it actually was."""

    source: str
    similarity: float | None = Field(
        default=None, description="Dense cosine. None when the retriever had no dense side."
    )


class Refusal(BaseModel):
    """A refusal that says something useful.

    "I don't know" is honest and nearly worthless: the reader learns only that this route
    failed. Three facts turn it into a next step -- what the search did surface, what it
    could not establish, and which pages are worth opening by hand -- and all three come
    free from the retrieval results, with no extra model call.
    """

    kind: RefusalKind
    reason: str = Field(description="Why the pipeline stopped, in one line.")
    what_was_found: list[NearMiss] = Field(default_factory=list)
    what_was_missing: str = ""
    documents_to_check: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """The refusal as prose, opening with the sentinel phrase.

        The first line is unchanged from the bare refusal on purpose: downstream string
        checks, the Tier-2 judge and a reader skimming all key off it, and the structure
        is an addition rather than a replacement.
        """
        lines = [NO_ANSWER_TEXT, "", f"What I could not establish: {self.what_was_missing}"]
        if self.what_was_found:
            lines.append("")
            lines.append("What the search did find:")
            lines.extend(
                f"  - {miss.source}"
                + ("" if miss.similarity is None else f" (similarity {miss.similarity:.2f})")
                for miss in self.what_was_found
            )
        if self.documents_to_check:
            lines.append("")
            lines.append("Worth checking by hand:")
            lines.extend(f"  - {path}" for path in self.documents_to_check)
        return "\n".join(lines)


class AnswerConfidence(BaseModel):
    """Confidence, broken down rather than collapsed into one opaque number.

    Which fields are populated depends on how the answer was produced: `retrieval` and the
    structural half of `citation_coverage` are free and always present, while verified
    coverage and `completeness` each cost a model call and appear only when verification
    was asked for. `components` records which of them actually fed `composite`, so a
    cheap composite and a thorough one are never mistaken for the same measurement.
    """

    retrieval: float | None = Field(
        default=None,
        description=(
            "Highest dense cosine similarity among the retrieved chunks. None when the "
            "retriever had no dense component, because BM25 scores are not comparable "
            "across queries and would make this number meaningless."
        ),
    )
    retrieval_calibrated: float | None = Field(
        default=None,
        description="`retrieval` mapped onto 0-1 by the ramp measured over the golden set.",
    )
    both_retrievers_agree: bool | None = Field(
        default=None,
        description="Whether the top-ranked chunk was found by dense and sparse alike.",
    )
    citation_coverage: float | None = Field(
        default=None, description="Share of the answer's claims carrying a citation."
    )
    citation_coverage_verified: bool = Field(
        default=False,
        description=(
            "True when coverage counts only citations a verifier confirmed support their "
            "claim. False when it is the free structural check, which counts a claim as "
            "covered merely because a bracketed number is attached to it."
        ),
    )
    citation_precision: float | None = Field(
        default=None,
        description="Of the claims that were cited, the share whose citations hold up.",
    )
    completeness: float | None = Field(
        default=None, description="Whether every part of the question was addressed."
    )
    composite: float | None = Field(
        default=None, description="Weighted mean of the components that were measured."
    )
    components: tuple[str, ...] = Field(
        default=(), description="Which components fed `composite`. Empty when none did."
    )


class Answer(BaseModel):
    """A grounded answer with its citations, evidence and cost."""

    question: str
    text: str
    answered: bool = Field(
        description="False when the model declined, or retrieval fell below the threshold."
    )
    refusal_reason: str | None = Field(
        default=None, description="Why an unanswered question was declined."
    )
    refusal: Refusal | None = Field(
        default=None, description="The structured form of that refusal, for API consumers."
    )

    citations: list[Citation] = Field(default_factory=list)
    unresolved_citations: list[int] = Field(default_factory=list)
    uncited_blocks: list[int] = Field(default_factory=list)
    claims: int = Field(
        default=0, description="Assertions found in the answer; the coverage denominator."
    )
    unsupported_citations: list[int] = Field(
        default_factory=list,
        description=(
            "Claims whose cited blocks do not support them, by claim index. Populated only "
            "when claim-level verification ran; empty otherwise, which is not the same as "
            "verified sound."
        ),
    )
    retrieved: list[RetrievedChunk] = Field(
        default_factory=list, description="The context the answer was generated from."
    )
    confidence: AnswerConfidence = Field(default_factory=AnswerConfidence)

    model: str = ""
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cost_usd: NonNegativeFloat = Field(
        default=0.0, description="What generating this answer cost. Verification is separate."
    )
    verification_cost_usd: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "What checking this answer cost: claim verification plus the completeness "
            "call. Kept apart from `cost_usd` so turning verification on does not silently "
            "change what every previously reported cost-per-answer figure meant."
        ),
    )
    latency_s: NonNegativeFloat = 0.0
    cached: bool = Field(
        default=False,
        description=(
            "Generation was served from the response cache, so `latency_s` is a lookup "
            "and `cost_usd` is 0. A latency benchmark must filter these out."
        ),
    )

    @property
    def sources(self) -> list[str]:
        """Distinct cited sources, in citation order, for a compact footer."""
        seen: list[str] = []
        for citation in self.citations:
            if citation.source not in seen:
                seen.append(citation.source)
        return seen
