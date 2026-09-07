"""Verify every citation the system makes, and check that the verifier is really reading.

    uv run python scripts/verify_citations.py --dry-run   # what it would cost
    uv run python scripts/verify_citations.py

Structural citation checking asks whether `[3]` points at a block that exists. This asks
whether block 3 says the thing the sentence attached to it claims -- the quality layer the
brief calls the one most RAG systems skip.

A model grading a model is worth nothing unless the grader is itself checked, which is the
rule the Tier-2 judge was held to and this holds to as well. Here the check needs no human
labelling: every cited claim is verified twice, once against the blocks it actually cited
and once against blocks drawn from the corpus that it never cited. A verifier that says
"supported" out of politeness cannot separate those, and the separation is reported above
the headline numbers rather than below them.

Cost control is not advisory: the verifier carries a hard `cost_budget_usd` that raises
before a request is sent, every call goes through the persistent cache, and `--dry-run`
prices the run before a key is touched.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from hybridrag.chunk_store import ChunkStore
from hybridrag.config import get_settings
from hybridrag.embedding import Embedder, FastEmbedEmbedder
from hybridrag.embedding_cache import CachedEmbedder
from hybridrag.embedding_openai import OpenAIEmbedder
from hybridrag.evaluation import GoldenSet, QuestionCategory
from hybridrag.evaluation.report import (
    VerificationRun,
    VerifiedClaim,
    git_provenance,
    render_verification_report,
)
from hybridrag.evaluation.stats import DEFAULT_SEED
from hybridrag.generation import (
    Answerer,
    BudgetExceededError,
    CachedLanguageModel,
    CitationVerifier,
    CompletenessScorer,
    OpenAIModel,
)
from hybridrag.generation.claims import split_claims
from hybridrag.generation.verification import VerificationError
from hybridrag.indexing import (
    DenseIndex,
    SparseIndex,
    chroma_path,
    collection_name,
    sparse_path,
    store_path,
)
from hybridrag.models import ChunkingStrategy
from hybridrag.retrieval import HybridRetriever, RetrievedChunk

app = typer.Typer(add_completion=False)

GENERATION_MODEL = "gpt-5-nano-2025-08-07"
# The judge validated in Phase 4 at kappa 0.730 on correctness. That figure does not
# transfer to this task -- it is a different question -- but gpt-5-nano was measured at
# chance level as a judge (0.007), so using it here would be indefensible.
VERIFIER_MODEL = "gpt-5-mini-2025-08-07"
VERIFIER_MAX_TOKENS = 1024

# Measured, not estimated: one `--verify` run on a real question cost $0.001740 across
# four claim checks and one completeness call, so $0.000348 per call of either kind.
COST_PER_CALL = 0.000348
# Measured over the golden set: 108 claims across 29 answers, 56 of them cited.
CLAIMS_PER_ANSWER = 56 / 29

# The control establishes that the verifier discriminates; it does not need every claim to
# do that. Capped so the check costs a third of what verification does rather than doubling
# it, and sampled with the run's seed so a re-run controls the same claims.
DEFAULT_CONTROL_SAMPLE = 40


def _embedder() -> Embedder:
    settings = get_settings()
    if settings.openai_api_key is None:
        return FastEmbedEmbedder(settings.embedding_model)
    return OpenAIEmbedder(settings.openai_api_key)


def _control_results(
    results: list[RetrievedChunk],
    cited: tuple[int, ...],
    store: ChunkStore,
    pool: list[str],
    offset: int,
) -> list[RetrievedChunk]:
    """The same result list with every cited position swapped for an unrelated chunk.

    The claim, the prompt and the block numbering are all held constant; only the passage
    behind the number changes. That isolates the one thing being tested -- whether the
    verifier's verdict depends on what the passage actually says.

    `offset` walks the shuffled pool so different claims are controlled against different
    decoys. Without it every claim is checked against the same first few chunks, and the
    control would measure whether those particular chunks happen to be unconvincing rather
    than whether the verifier reads at all.
    """
    retrieved = {result.chunk.chunk_id for result in results}
    swapped = list(results)
    cursor = offset
    for number in cited:
        if not 1 <= number <= len(swapped):
            continue
        for step in range(len(pool)):
            candidate = pool[(cursor + step) % len(pool)]
            if candidate in retrieved:
                continue
            chunk = store.get(candidate)
            if chunk is not None:
                swapped[number - 1] = swapped[number - 1].model_copy(update={"chunk": chunk})
                retrieved.add(candidate)
                cursor = (cursor + step + 1) % len(pool)
                break
    return swapped


@app.command()
def verify(
    strategy: Annotated[
        ChunkingStrategy, typer.Option(help="Chunking strategy, defaulting to Tier 1's winner.")
    ] = ChunkingStrategy.FIXED,
    k: Annotated[int, typer.Option(help="Context blocks per answer.")] = 5,
    budget: Annotated[
        float, typer.Option(help="Hard spend cap. Raises before a request is sent.")
    ] = 0.20,
    control: Annotated[
        bool, typer.Option(help="Also verify claims against blocks they never cited.")
    ] = True,
    control_sample: Annotated[
        int,
        typer.Option(
            help="How many claims to run the negative control on. 0 runs it on all of them."
        ),
    ] = DEFAULT_CONTROL_SAMPLE,
    seed: Annotated[int, typer.Option(help="Seed for the control's chunk sampling.")] = (
        DEFAULT_SEED
    ),
    golden_path: Annotated[Path, typer.Option()] = Path("evals/golden_set.yaml"),
    out_dir: Annotated[Path, typer.Option()] = Path("evals/reports"),
    dry_run: Annotated[bool, typer.Option(help="Price the run, spend nothing.")] = False,
    index_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Verify every citation on the golden set, with a negative control."""
    settings = get_settings()
    golden = GoldenSet.load(golden_path)
    # no_answer questions should be refused, so they carry no citations to verify.
    # Filtered on the category rather than on an empty reference: these do carry
    # reference text, explaining *why* the corpus cannot answer them.
    answerable = [
        question
        for question in golden.questions
        if question.category is not QuestionCategory.NO_ANSWER
    ]

    if dry_run:
        expected = len(answerable) * CLAIMS_PER_ANSWER
        controlled = 0.0
        if control:
            controlled = min(expected, control_sample) if control_sample else expected
        calls = expected + controlled + len(answerable)
        typer.echo(
            f"{len(answerable)} answerable questions x ~{CLAIMS_PER_ANSWER:.0f} cited "
            f"claims = ~{expected:.0f} claims\n"
            f"  verification      ~{expected:.0f} calls\n"
            f"  negative control  ~{controlled:.0f} calls\n"
            f"  completeness      {len(answerable)} calls\n"
            f"  ~{calls:.0f} calls x ${COST_PER_CALL:.6f} measured = "
            f"${calls * COST_PER_CALL:.4f}   (budget ${budget:.2f})"
        )
        raise typer.Exit()

    if settings.openai_api_key is None:
        typer.echo("error: OPENAI_API_KEY is not set. Put it in .env.")
        raise typer.Exit(code=1)

    root = index_dir or settings.index_dir
    bm25 = sparse_path(root, strategy)
    if not bm25.is_file():
        typer.echo(f"error: no {strategy.value} index at {root}. Build it first.")
        raise typer.Exit(code=1)

    embedder = CachedEmbedder(_embedder(), settings.cache_dir / "embeddings.sqlite")
    store = ChunkStore(store_path(root))
    retriever = HybridRetriever(
        {
            "dense": DenseIndex.embedded(
                embedder, chroma_path(root), collection_name=collection_name(strategy)
            ),
            "sparse": SparseIndex.load(bm25),
        },
        store,
    )

    generator = CachedLanguageModel(
        OpenAIModel(
            settings.openai_api_key,
            GENERATION_MODEL,
            max_output_tokens=settings.answer_max_tokens,
            cost_budget_usd=budget,
        ),
        settings.cache_dir / "completions.sqlite",
    )
    checker = CachedLanguageModel(
        OpenAIModel(
            settings.openai_api_key,
            VERIFIER_MODEL,
            max_output_tokens=VERIFIER_MAX_TOKENS,
            cost_budget_usd=budget,
        ),
        settings.cache_dir / "completions.sqlite",
    )
    verifier = CitationVerifier(checker, max_output_tokens=VERIFIER_MAX_TOKENS)
    answerer = Answerer(
        retriever,
        generator,
        k=k,
        confidence_threshold=settings.retrieval_confidence_threshold,
        verifier=verifier,
        completeness_scorer=CompletenessScorer(checker, max_output_tokens=VERIFIER_MAX_TOKENS),
    )

    # Sampled once with a fixed seed rather than per claim, so a re-run pairs each claim
    # with the same unrelated chunks and the control is reproducible.
    pool = sorted(store.chunk_ids(strategy))
    random.Random(seed).shuffle(pool)

    controlled_so_far = 0
    unreadable = 0
    sha, dirty = git_provenance()
    run = VerificationRun(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=sha,
        git_dirty=dirty,
        arm="hybrid",
        strategy=strategy.value,
        generation_model=GENERATION_MODEL,
        verifier_model=VERIFIER_MODEL,
        questions=len(answerable),
        answered=0,
        seed=seed,
        price_per_call=COST_PER_CALL,
    )

    try:
        for question in answerable:
            answer = answerer.answer(question.question)
            if not answer.answered:
                typer.echo(f"  {question.question_id}: refused")
                continue

            run.answered += 1
            if answer.confidence.citation_coverage is not None:
                run.coverage.append(answer.confidence.citation_coverage)
            if answer.confidence.citation_precision is not None:
                run.precision.append(answer.confidence.citation_precision)

            claims = split_claims(answer.text)
            try:
                report = verifier.verify(claims, answer.retrieved)
            except VerificationError as error:
                # One unreadable reply already cost a run 46 calls in. Counted and
                # reported rather than fatal, the way the judge counts its own.
                unreadable += 1
                typer.echo(f"  {question.question_id}: unreadable verifier reply ({error})")
                continue
            run.cost_usd += report.cost_usd
            run.calls += report.cited_claims
            run.cached_calls += sum(item.cached for item in report.verifications)

            for item in report.verifications:
                control_supported: bool | None = None
                budget_left = not control_sample or controlled_so_far < control_sample
                if control and budget_left:
                    swapped = _control_results(
                        answer.retrieved, item.cited, store, pool, controlled_so_far * 7
                    )
                    try:
                        control_item = verifier.verify_claim(claims[item.claim_index], swapped)
                    except VerificationError:
                        unreadable += 1
                    else:
                        run.control_cost_usd += control_item.cost_usd
                        run.calls += 1
                        run.cached_calls += int(control_item.cached)
                        control_supported = control_item.is_supported
                        controlled_so_far += 1

                run.claims.append(
                    VerifiedClaim(
                        question_id=question.question_id,
                        category=question.category.value,
                        claim_index=item.claim_index,
                        claim_text=item.claim_text,
                        cited=item.cited,
                        supported=item.is_supported,
                        control_supported=control_supported,
                        reason=item.reason,
                    )
                )

            flagged = len(report.unsupported)
            typer.echo(
                f"  {question.question_id}: {report.cited_claims}/{report.total_claims} "
                f"claims cited, {flagged} unsupported"
            )
    except BudgetExceededError as error:
        typer.echo(f"\nstopped: {error}")
    finally:
        generator.close()
        checker.close()
        embedder.close()
        store.close()

    run.unreadable = unreadable
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "citation_verification.md").write_text(
        render_verification_report(run), encoding="utf-8"
    )
    (out_dir / "citation_verification.json").write_text(
        json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    if unreadable:
        typer.echo(f"\n{unreadable} verifier repl(ies) were unreadable and were skipped.")
    typer.echo(
        f"\nwrote {out_dir / 'citation_verification.md'}  "
        f"(${run.cost_usd + run.control_cost_usd:.4f} spent)"
    )


if __name__ == "__main__":
    sys.exit(app())
