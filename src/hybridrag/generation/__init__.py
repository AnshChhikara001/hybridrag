"""Grounded answer generation with inline citations."""

from __future__ import annotations

from hybridrag.generation.answerer import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    Answerer,
    retrieval_confidence,
)
from hybridrag.generation.base import Completion, GenerationError, LanguageModel
from hybridrag.generation.cache import CachedLanguageModel
from hybridrag.generation.citations import parse_citation_numbers, resolve_citations
from hybridrag.generation.gemini import DEFAULT_MODEL, GeminiModel
from hybridrag.generation.models import Answer, AnswerConfidence, Citation, CitationReport
from hybridrag.generation.openai_model import BudgetExceededError, OpenAIModel
from hybridrag.generation.prompt import (
    REFUSAL_SENTINEL,
    SYSTEM_PROMPT,
    build_prompt,
    render_context,
)

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_MODEL",
    "REFUSAL_SENTINEL",
    "SYSTEM_PROMPT",
    "Answer",
    "AnswerConfidence",
    "Answerer",
    "BudgetExceededError",
    "CachedLanguageModel",
    "Citation",
    "CitationReport",
    "Completion",
    "GeminiModel",
    "GenerationError",
    "LanguageModel",
    "OpenAIModel",
    "build_prompt",
    "parse_citation_numbers",
    "render_context",
    "resolve_citations",
    "retrieval_confidence",
]
