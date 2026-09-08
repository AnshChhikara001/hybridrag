"""Request and response shapes specific to the HTTP boundary.

`Answer` (generation/models.py) is used as-is for `POST /v1/ask`'s response -- its own
docstring already says it is what the FastAPI response model serialises, so a second
answer-shaped schema here would just be a copy that can drift from the real one. Everything
in this file exists because the API needs it and nothing upstream already provides it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from hybridrag.models import ChunkingStrategy, SourceFormat


class RetrievalMode(StrEnum):
    """What `/v1/ask` searches with. Settled now even though only the Phase 5.2 dashboard
    needs the ablation -- adding the field later would mean a second pass over an API
    contract already in use, per CLAUDE.md's cross-cutting-constraint rule.
    """

    HYBRID = "hybrid"
    DENSE_ONLY = "dense_only"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="The question to answer.")
    k: int = Field(default=5, ge=1, le=20, description="Context blocks to retrieve.")
    mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID, description="hybrid (default) or dense_only, an ablation."
    )
    verify: bool = Field(
        default=False,
        description="Check each cited claim against the block it cites, and score "
        "completeness. Costs one model call per cited claim plus one for the answer, so it "
        "is off by default -- matching scripts/ask.py's --verify flag.",
    )


class DocumentSummary(BaseModel):
    """One indexed document, for `GET /v1/documents`."""

    relative_path: str
    title: str | None
    source_format: SourceFormat
    content_hash: str
    sections: int
    chunks: int = Field(description="Chunk count under the strategy the API serves.")


class DocumentsResponse(BaseModel):
    strategy: ChunkingStrategy
    documents: list[DocumentSummary]


class IngestResponse(BaseModel):
    relative_path: str
    doc_id: str
    sections: int
    chunks_added: int
    chunks_removed: int = Field(
        description="Chunks from a previous version of this same document, now superseded."
    )
    duplicates_dropped: int = Field(
        description="Near-duplicate chunks found within this one document (D42's scope) "
        "and excluded from indexing."
    )
    cost_usd: float = Field(description="Embedding spend for this document's new chunks.")


class HealthResponse(BaseModel):
    status: str = "ok"
    strategy: ChunkingStrategy
    documents_indexed: int
    chunks_indexed: int
