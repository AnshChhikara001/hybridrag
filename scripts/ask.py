"""Ask the corpus a question and get a grounded, cited answer.

The end of the vertical slice: question in, answer with inline `[n]` citations out, plus
the evidence behind it -- what was retrieved, which retriever found each chunk, the
confidence that cleared the gate, and what the call cost.

    uv run python scripts/ask.py "how do I run FastAPI in Docker"
    uv run python scripts/ask.py "what is the capital of France"   # refused, no request sent
    uv run python scripts/ask.py "how do I use response_model" --chunks
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from google.genai.errors import APIError

from hybridrag.chunk_store import ChunkStore
from hybridrag.config import get_settings
from hybridrag.embedding import Embedder, FastEmbedEmbedder
from hybridrag.embedding_cache import CachedEmbedder
from hybridrag.embedding_openai import OpenAIEmbedder
from hybridrag.generation import (
    Answer,
    Answerer,
    CachedLanguageModel,
    CitationVerifier,
    CompletenessScorer,
    GeminiModel,
    GenerationError,
    LanguageModel,
    OpenAIModel,
)
from hybridrag.indexing import (
    DenseIndex,
    SparseIndex,
    chroma_path,
    collection_name,
    sparse_path,
    store_path,
)
from hybridrag.models import ChunkingStrategy
from hybridrag.retrieval import HybridRetriever

app = typer.Typer(add_completion=False)

OPENAI_DEFAULT = "gpt-5-nano-2025-08-07"

# The judge validated in Phase 4 (kappa 0.730 against a human on correctness). Verification
# and completeness are different tasks from the one that agreement was measured on, so the
# figure does not transfer -- but a model already shown to disagree with a human at chance
# level would be indefensible here, and gpt-5-nano was shown to do exactly that.
VERIFIER_MODEL = "gpt-5-mini-2025-08-07"
VERIFIER_MAX_TOKENS = 1024


def _embedder() -> Embedder:
    """Whatever built the index must also embed the query, or the vectors do not compare."""
    settings = get_settings()
    if settings.openai_api_key is None:
        return FastEmbedEmbedder(settings.embedding_model)
    return OpenAIEmbedder(settings.openai_api_key)


def _language_model(provider: str | None) -> LanguageModel:
    """The configured provider, or an override. D4's "reversible in one line", in one line.

    Written because Gemini's free tier began returning 429 mid-build: a single hardcoded
    provider would have blocked the slice on someone else's capacity.
    """
    settings = get_settings()
    chosen = provider or settings.generation_provider
    if chosen == "openai":
        if settings.openai_api_key is None:
            raise typer.BadParameter("OPENAI_API_KEY is not set. Put it in .env.")
        model = settings.generation_model
        return OpenAIModel(
            settings.openai_api_key,
            model if model.startswith(("gpt-", "o1", "o3", "o4")) else OPENAI_DEFAULT,
            max_output_tokens=settings.answer_max_tokens,
        )
    if settings.gemini_api_key is None:
        raise typer.BadParameter("GEMINI_API_KEY is not set. Put it in .env.")
    return GeminiModel(
        settings.gemini_api_key,
        settings.generation_model,
        max_output_tokens=settings.answer_max_tokens,
    )


def _report_confidence(answer: Answer) -> None:
    """The confidence breakdown, saying plainly which parts were actually measured."""
    confidence = answer.confidence
    typer.echo("\nCONFIDENCE")
    typer.echo(f"  retrieval (top dense cosine) : {_number(confidence.retrieval)}")
    typer.echo(f"  retrieval (calibrated 0-1)   : {_number(confidence.retrieval_calibrated)}")
    typer.echo(f"  both retrievers agree        : {confidence.both_retrievers_agree}")

    kind = "verified" if confidence.citation_coverage_verified else "structural only"
    typer.echo(
        f"  citation coverage            : {_number(confidence.citation_coverage)} ({kind}"
        f", {answer.claims} claim(s))"
    )
    if confidence.citation_precision is not None:
        typer.echo(f"  citation precision           : {_number(confidence.citation_precision)}")
    typer.echo(f"  completeness                 : {_number(confidence.completeness)}")

    measured = ", ".join(confidence.components) or "nothing"
    typer.echo(f"  COMPOSITE                    : {_number(confidence.composite)}")
    typer.echo(f"  ... built from               : {measured}")


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _report(answer: Answer, *, show_chunks: bool) -> None:
    typer.echo(f"\n{answer.text}\n")

    if answer.citations:
        typer.echo("SOURCES")
        for citation in answer.citations:
            # No per-source flag here: `unsupported_citations` holds claim indices, not
            # citation numbers, and a citation can be shared by several claims, so which
            # of *these* citations is implicated is not recoverable at this point. The
            # correct, unambiguous view is the claim-indexed list printed below.
            typer.echo(f"  [{citation.number}] {citation.source}")

    if answer.unresolved_citations:
        # Surfaced rather than swallowed: a citation pointing at no block is a fabricated
        # source, and hiding it defeats the purpose of citing at all.
        typer.echo(f"\n  ! FABRICATED CITATIONS: {answer.unresolved_citations}")

    if answer.unsupported_citations:
        # The failure structural checking cannot see: the number resolves, so it renders
        # as a working source link, and the block it points at does not say the thing.
        typer.echo(f"\n  ! UNSUPPORTED CLAIMS (by claim index): {answer.unsupported_citations}")

    _report_confidence(answer)

    if answer.refusal is not None:
        typer.echo(f"\n  refused ({answer.refusal.kind.value}): {answer.refusal.reason}")

    typer.echo("\nRETRIEVED")
    for result in answer.retrieved:
        cited = "*" if any(c.number == result.rank for c in answer.citations) else " "
        votes = "+".join(result.retrievers)
        heading = " > ".join(result.chunk.heading_path) or "(preamble)"
        typer.echo(f" {cited}[{result.rank}] {result.chunk.relative_path}  [{heading}]  ({votes})")
        if show_chunks:
            typer.echo(f"      {' '.join(result.chunk.text.split())[:220]}...")

    source = "cached" if answer.cached else "generated"
    total = answer.cost_usd + answer.verification_cost_usd
    checking = (
        f" (+${answer.verification_cost_usd:.6f} checking)" if answer.verification_cost_usd else ""
    )
    typer.echo(
        f"\n{answer.model}  {answer.input_tokens} in / {answer.output_tokens} out  "
        f"${total:.6f}{checking}  {answer.latency_s:.2f}s  ({source})"
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    k: Annotated[int, typer.Option(help="Context blocks to retrieve.")] = 5,
    chunks: Annotated[bool, typer.Option(help="Show a snippet of each retrieved chunk.")] = False,
    dense_only: Annotated[bool, typer.Option(help="Ablate sparse, for comparison.")] = False,
    no_cache: Annotated[
        bool, typer.Option(help="Bypass the response cache, for true latency or a fresh answer.")
    ] = False,
    verify: Annotated[
        bool,
        typer.Option(
            help="Check each cited claim against the block it cites, and score completeness. "
            "Costs one model call per cited claim plus one for the answer."
        ),
    ] = False,
    provider: Annotated[
        str | None, typer.Option(help="Override the configured provider: gemini or openai.")
    ] = None,
    strategy: Annotated[
        ChunkingStrategy, typer.Option(help="Which chunking strategy's indexes to answer from.")
    ] = ChunkingStrategy.FIXED,
    index_dir: Annotated[Path | None, typer.Option(help="Overrides the configured path.")] = None,
) -> None:
    """Answer one question from the indexed corpus, with citations."""
    settings = get_settings()
    root = index_dir or settings.index_dir
    bm25 = sparse_path(root, strategy)
    if not bm25.is_file():
        typer.echo(
            f"error: no {strategy.value} index at {root}. Build it first:\n"
            f"  uv run python scripts/build_index.py <corpus> --strategy {strategy.value}"
        )
        raise typer.Exit(code=1)
    cached = CachedEmbedder(_embedder(), settings.cache_dir / "embeddings.sqlite")
    store = ChunkStore(store_path(root))
    retriever = HybridRetriever(
        {
            "dense": DenseIndex.embedded(
                cached, chroma_path(root), collection_name=collection_name(strategy)
            ),
            "sparse": SparseIndex.load(bm25),
        },
        store,
    )
    if dense_only:
        retriever = retriever.ablation("dense")

    # Cached by default: the same demo question is asked many times while iterating, and
    # the free tier rate-limited this project once already.
    model = CachedLanguageModel(
        _language_model(provider),
        settings.cache_dir / "completions.sqlite",
        read_only=no_cache,
    )
    # Off by default: verification is right for an evaluation run and wrong for a
    # dashboard that must answer in under a second. What is free is computed either way.
    checker = (
        CachedLanguageModel(
            OpenAIModel(
                settings.openai_api_key,
                VERIFIER_MODEL,
                max_output_tokens=VERIFIER_MAX_TOKENS,
            ),
            settings.cache_dir / "completions.sqlite",
        )
        if verify and settings.openai_api_key is not None
        else None
    )
    if verify and checker is None:
        raise typer.BadParameter("--verify needs OPENAI_API_KEY. Put it in .env.")

    answerer = Answerer(
        retriever,
        model,
        k=k,
        confidence_threshold=settings.retrieval_confidence_threshold,
        verifier=CitationVerifier(checker, max_output_tokens=VERIFIER_MAX_TOKENS)
        if checker
        else None,
        completeness_scorer=CompletenessScorer(checker, max_output_tokens=VERIFIER_MAX_TOKENS)
        if checker
        else None,
    )

    typer.echo(f"Q: {question}")
    try:
        _report(answerer.answer(question), show_chunks=chunks)
    except (APIError, GenerationError) as error:
        # A provider outage is not a bug in this program, and a 200-line SDK traceback
        # buries the one line that says what to do about it. Backoff has already been
        # exhausted by the time this is reached.
        typer.echo(f"\nerror: the model could not be reached.\n  {error}")
        typer.echo(
            "\nRetries with backoff are already exhausted. Try again shortly, or set "
            "HYBRIDRAG_GENERATION_MODEL to another model in .env."
        )
        raise typer.Exit(code=1) from error
    finally:
        model.close()
        if checker is not None:
            checker.close()
        cached.close()
        store.close()


if __name__ == "__main__":
    sys.exit(app())
