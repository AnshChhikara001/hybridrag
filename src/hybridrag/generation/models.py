"""What an answer is.

Declared in full now, including the dimensions Phase 3 will fill, because this shape is
what the FastAPI response model serialises and what the evaluation harness grades. Adding a
field later means changing an API contract and re-running every stored result; declaring it
now and leaving it explicitly `None` costs nothing and says plainly which parts are built.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveInt

from hybridrag.retrieval import RetrievedChunk


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


class AnswerConfidence(BaseModel):
    """Confidence, broken down rather than collapsed into one opaque number.

    `retrieval` is the only dimension populated today. The rest are declared so the API
    contract and the evaluation schema are settled, and left `None` so nothing reports a
    score it has not actually computed.
    """

    retrieval: float | None = Field(
        default=None,
        description=(
            "Highest dense cosine similarity among the retrieved chunks. None when the "
            "retriever had no dense component, because BM25 scores are not comparable "
            "across queries and would make this number meaningless."
        ),
    )
    both_retrievers_agree: bool | None = Field(
        default=None,
        description="Whether the top-ranked chunk was found by dense and sparse alike.",
    )
    citation_coverage: float | None = Field(
        default=None, description="Phase 3: share of claims carrying a verified citation."
    )
    completeness: float | None = Field(
        default=None, description="Phase 3: whether every part of the question was addressed."
    )
    composite: float | None = Field(
        default=None, description="Phase 3: the weighted combination, once its parts exist."
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

    citations: list[Citation] = Field(default_factory=list)
    unresolved_citations: list[int] = Field(default_factory=list)
    uncited_blocks: list[int] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(
        default_factory=list, description="The context the answer was generated from."
    )
    confidence: AnswerConfidence = Field(default_factory=AnswerConfidence)

    model: str = ""
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cost_usd: NonNegativeFloat = 0.0
    latency_s: NonNegativeFloat = 0.0

    @property
    def sources(self) -> list[str]:
        """Distinct cited sources, in citation order, for a compact footer."""
        seen: list[str] = []
        for citation in self.citations:
            if citation.source not in seen:
                seen.append(citation.source)
        return seen
