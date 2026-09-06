"""Retrieve, ground, generate, verify: the whole question-to-answer path.

Two independent refusal paths, deliberately, because they catch different failures:

* **The retrieval gate**, before any generation. Cheap, deterministic, and the only one
  that can refuse without spending a request. It catches out-of-domain questions, where
  nothing in the corpus is close to the query.
* **The model's own refusal**, via the sentinel in the system prompt. It catches
  in-domain questions the corpus happens not to answer -- where retrieval returns
  plausibly-close chunks that do not contain the fact asked for. The gate cannot see this;
  only something that has read the chunks can.

The gate reads *dense cosine similarity*, never the fused RRF score. RRF is computed from
ranks alone, so its top result scores about 1/(k+1) whether that chunk is perfect or
useless -- a threshold on it would be a confidence number that moves for no reason.
"""

from __future__ import annotations

from hybridrag.generation.base import LanguageModel
from hybridrag.generation.citations import resolve_citations
from hybridrag.generation.models import Answer, AnswerConfidence
from hybridrag.generation.prompt import REFUSAL_SENTINEL, SYSTEM_PROMPT, build_prompt
from hybridrag.retrieval import HybridRetriever, RetrievedChunk

DENSE = "dense"

# Measured on this corpus over 6 in-domain and 6 out-of-domain questions: in-domain top
# cosine ran 0.4602-0.6541, out-of-domain 0.1410-0.2414, a gap of 0.219. Set below the
# midpoint deliberately, because the two errors are not symmetric: a false refusal is final
# and the user simply gets nothing, while a false accept is caught downstream by the
# model's own refusal at the cost of one request. Re-calibrated against the golden set in
# Phase 4, where there are enough questions for the number to mean something.
DEFAULT_CONFIDENCE_THRESHOLD = 0.30

NO_ANSWER_TEXT = "I don't know based on the indexed documentation."


def retrieval_confidence(results: list[RetrievedChunk]) -> float | None:
    """Best dense cosine similarity across the retrieved chunks.

    None when no chunk carries a dense score, which means the retriever had no dense
    component. BM25 scores are unbounded and corpus-dependent, so there is no honest way
    to convert them into a confidence comparable across queries -- and inventing one is
    how a dashboard ends up displaying a number nobody can interpret.
    """
    scores = [result.hits[DENSE].score for result in results if DENSE in result.hits]
    return max(scores) if scores else None


class Answerer:
    """Answers a question from the corpus, with citations."""

    def __init__(
        self,
        retriever: HybridRetriever,
        model: LanguageModel,
        *,
        k: int = 5,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_output_tokens: int | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.retriever = retriever
        self.model = model
        self.k = k
        self.confidence_threshold = confidence_threshold
        self.max_output_tokens = max_output_tokens

    def answer(self, question: str) -> Answer:
        """Retrieve context, generate a grounded answer, and resolve its citations."""
        if not question.strip():
            raise ValueError("question must not be empty")

        results = self.retriever.retrieve(question, k=self.k)
        confidence = AnswerConfidence(
            retrieval=retrieval_confidence(results),
            both_retrievers_agree=len(results[0].hits) > 1 if results else None,
        )

        refusal = self._gate(results, confidence.retrieval)
        if refusal is not None:
            return Answer(
                question=question,
                text=NO_ANSWER_TEXT,
                answered=False,
                refusal_reason=refusal,
                retrieved=results,
                confidence=confidence,
                model=self.model.model_name,
            )

        completion = self.model.generate(
            build_prompt(question, results),
            system=SYSTEM_PROMPT,
            max_output_tokens=self.max_output_tokens,
        )

        # The sentinel may arrive with punctuation or stray whitespace around it; anything
        # longer than the sentinel plus a little is the model answering *and* hedging,
        # which is an answer, not a refusal.
        stripped = completion.text.strip().strip(".\"'` ")
        declined = stripped == REFUSAL_SENTINEL
        report = resolve_citations(completion.text, results)

        return Answer(
            question=question,
            text=NO_ANSWER_TEXT if declined else completion.text,
            answered=not declined,
            refusal_reason=(
                "The model judged the retrieved context insufficient to answer."
                if declined
                else None
            ),
            citations=report.citations,
            unresolved_citations=report.unresolved,
            uncited_blocks=report.uncited_blocks,
            retrieved=results,
            confidence=confidence,
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens + completion.thinking_tokens,
            cost_usd=completion.cost_usd,
            latency_s=completion.latency_s,
            cached=completion.cached,
        )

    def _gate(self, results: list[RetrievedChunk], confidence: float | None) -> str | None:
        """The pre-generation refusal, or None to proceed."""
        if not results:
            return "Retrieval returned no chunks for this question."
        if confidence is None:
            # Sparse-only retrieval. Refusing here would reject every query on a retriever
            # that simply has no dense side, so the model's own refusal carries this case.
            return None
        if confidence < self.confidence_threshold:
            return (
                f"Retrieval confidence {confidence:.3f} is below the threshold "
                f"{self.confidence_threshold:.2f}: nothing in the corpus is close to "
                "this question."
            )
        return None
