"""Run Tier 2: answer the golden set, judge the answers, attribute the failures.

    uv run python scripts/evaluate_answers.py --dry-run    # what it would cost
    uv run python scripts/evaluate_answers.py

Two arms by default -- hybrid and dense-only over the Tier-1 winning chunking -- because
"does hybrid retrieval produce better *answers*" is a different question from "does it
retrieve better", and only the second one has been answered so far.

Cost control is not advisory here. Both models carry a hard `cost_budget_usd` that raises
before a request is sent rather than after, every answer and every judgement goes through
the persistent cache, and `--dry-run` prices the run before a key is touched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from hybridrag.chunk_store import ChunkStore
from hybridrag.config import get_settings
from hybridrag.embedding import Embedder, FastEmbedEmbedder
from hybridrag.embedding_cache import CachedEmbedder
from hybridrag.embedding_openai import OpenAIEmbedder
from hybridrag.evaluation import GoldenSet, LabelSheet, QuestionCategory
from hybridrag.evaluation.judge import Judge, JudgeError, LabelItem, Verdict
from hybridrag.evaluation.quality import AnswerRecord, QualityArm, classify, reference_for
from hybridrag.evaluation.report import render_quality_report
from hybridrag.evaluation.stats import bootstrap_kappa
from hybridrag.generation import Answerer, BudgetExceededError, CachedLanguageModel, OpenAIModel
from hybridrag.generation.openai_model import price_of
from hybridrag.generation.prompt import render_context
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

GENERATION_MODEL = "gpt-5-nano-2025-08-07"
# Validated against human labels at kappa 0.730; gpt-5-nano was disqualified at 0.007 (D37).
JUDGE_MODEL = "gpt-5-mini-2025-08-07"
JUDGE_MAX_TOKENS = 2048

# Measured, not estimated: $0.000143 per answer and $0.000677 per judgement, from the
# vertical slice and the judge-agreement run respectively.
COST_PER_ANSWER = 0.000143
COST_PER_JUDGEMENT = 0.000677


def _embedder() -> Embedder:
    settings = get_settings()
    if settings.openai_api_key is None:
        return FastEmbedEmbedder(settings.embedding_model)
    return OpenAIEmbedder(settings.openai_api_key)


def _retrieved_spans(results_path: Path, arm: str) -> dict[str, bool] | None:
    """Whether Tier 1 found each question's answer span in this arm's top 10.

    This is the join that turns "the answer is wrong" into "retrieval never delivered the
    evidence" or "the evidence was there and the generator missed it".
    """
    if not results_path.is_file():
        return None
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    for entry in payload.get("arms", []):
        if f"{entry['retriever']}/{entry['strategy']}" != arm:
            continue
        return {
            question["question_id"]: bool(
                question["hits"].get("10", question["hits"].get(10, False))
            )
            for question in entry["questions"]
        }
    return None


def _judge_agreement(
    labels: Path, judge_model: str, reports: Path
) -> tuple[float, float, float, int]:
    """The measured agreement this judge earned, to print beside every number it produces."""
    agreement_path = reports / "judge_agreement.json"
    if not (labels.is_file() and agreement_path.is_file()):
        return (0.0, 0.0, 0.0, 0)
    sheet = LabelSheet.load(labels)
    items = sheet.labelled()
    human = [item.human_verdict.value for item in items if item.human_verdict is not None]
    payload = json.loads(agreement_path.read_text(encoding="utf-8"))
    for entry in payload.get("judges", []):
        if entry["model"] != judge_model:
            continue
        verdicts = [entry["verdicts"][item.question_id] for item in items]
        interval = bootstrap_kappa(verdicts, human)
        return (interval.mean, interval.low, interval.high, len(items))
    return (0.0, 0.0, 0.0, 0)


@app.command()
def evaluate(
    arm: Annotated[
        list[str] | None, typer.Option(help="Retriever arms to run. Repeatable.")
    ] = None,
    strategy: Annotated[
        ChunkingStrategy, typer.Option(help="Chunking strategy, defaulting to Tier 1's winner.")
    ] = ChunkingStrategy.FIXED,
    k: Annotated[int, typer.Option(help="Context blocks per answer.")] = 5,
    budget: Annotated[
        float, typer.Option(help="Hard spend cap. Raises before a request is sent.")
    ] = 0.15,
    golden_path: Annotated[Path, typer.Option()] = Path("evals/golden_set.yaml"),
    labels_path: Annotated[Path, typer.Option()] = Path("evals/judge_labels.yaml"),
    out_dir: Annotated[Path, typer.Option()] = Path("evals/reports"),
    dry_run: Annotated[bool, typer.Option(help="Price the run, spend nothing.")] = False,
    index_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Answer every golden question on each arm, judge it, and attribute the failures."""
    settings = get_settings()
    root = index_dir or settings.index_dir
    names = arm or ["hybrid", "dense"]
    golden = GoldenSet.load(golden_path)
    questions = golden.verified()

    calls = len(questions) * len(names)
    estimate = calls * (COST_PER_ANSWER + COST_PER_JUDGEMENT)
    typer.echo(
        f"{len(questions)} questions x {len(names)} arm(s) = {calls} answers and "
        f"{calls} judgements\n"
        f"estimated worst case: {calls} x (${COST_PER_ANSWER:.6f} + ${COST_PER_JUDGEMENT:.6f}) "
        f"= ${estimate:.4f}\n"
        f"hard cap: ${budget:.4f} per model. Anything already cached costs $0."
    )
    if dry_run:
        typer.echo("\n--dry-run: nothing generated, nothing spent.")
        return
    if estimate > budget:
        typer.echo(
            f"\nerror: the estimate (${estimate:.4f}) exceeds the cap (${budget:.4f}). "
            "Raise --budget deliberately, or run fewer arms."
        )
        raise typer.Exit(code=1)

    bm25 = sparse_path(root, strategy)
    if not bm25.is_file():
        typer.echo(f"error: no {strategy.value} index at {root}. Build it first.")
        raise typer.Exit(code=1)
    if settings.openai_api_key is None:
        raise typer.BadParameter("OPENAI_API_KEY is not set. Put it in .env.")

    embedder = CachedEmbedder(_embedder(), settings.cache_dir / "embeddings.sqlite")
    store = ChunkStore(store_path(root))
    full = HybridRetriever(
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
    # Held by name rather than inlined: the cache is what has to be closed, and the
    # protocol the judge takes does not promise a close().
    judge_model = CachedLanguageModel(
        OpenAIModel(
            settings.openai_api_key,
            JUDGE_MODEL,
            max_output_tokens=JUDGE_MAX_TOKENS,
            cost_budget_usd=budget,
        ),
        settings.cache_dir / "completions.sqlite",
    )
    judge = Judge(judge_model, max_output_tokens=JUDGE_MAX_TOKENS)

    arms: list[QualityArm] = []
    started = time.perf_counter()
    try:
        for name in names:
            retriever = full if name == "hybrid" else full.ablation(name)
            retrieved = _retrieved_spans(
                out_dir / "retrieval_results.json", f"{name}/{strategy.value}"
            )
            answerer = Answerer(
                retriever,
                generator,
                k=k,
                confidence_threshold=settings.retrieval_confidence_threshold,
            )
            records: list[AnswerRecord] = []
            for question in questions:
                answer = answerer.answer(question.question)
                item = LabelItem(
                    question_id=question.question_id,
                    category=question.category,
                    question=question.question,
                    reference_answer=reference_for(question),
                    system_answer=answer.text,
                    refused=not answer.answered,
                    context=render_context(answer.retrieved),
                )
                try:
                    judged = judge.judge(item)
                    verdict, grounded, reason = (
                        judged.decision.verdict,
                        judged.decision.grounded,
                        judged.decision.reason,
                    )
                    judge_cost = judged.cost_usd
                except JudgeError as error:
                    # An unreadable judgement is scored as the worst case rather than
                    # dropped, so a judge that cannot answer never improves a number.
                    verdict, grounded, reason = Verdict.INCORRECT, None, f"unreadable: {error}"
                    judge_cost = 0.0

                span_retrieved = (
                    None
                    if question.category is QuestionCategory.NO_ANSWER or retrieved is None
                    else retrieved.get(question.question_id, False)
                )
                records.append(
                    AnswerRecord(
                        question_id=question.question_id,
                        category=question.category,
                        answered=answer.answered,
                        verdict=verdict,
                        grounded=grounded,
                        judge_reason=reason,
                        citations=len(answer.citations),
                        fabricated_citations=len(answer.unresolved_citations),
                        span_retrieved=span_retrieved,
                        failure=classify(
                            verdict,
                            answered=answer.answered,
                            category=question.category,
                            span_retrieved=span_retrieved,
                        ),
                        generation_cost_usd=answer.cost_usd,
                        modelled_cost_usd=price_of(
                            GENERATION_MODEL, answer.input_tokens, answer.output_tokens
                        ),
                        judge_cost_usd=judge_cost,
                        latency_s=answer.latency_s,
                        cached=answer.cached,
                    )
                )
            correct = sum(1 for record in records if record.verdict is Verdict.CORRECT)
            typer.echo(
                f"[{time.perf_counter() - started:6.1f}s] {name}/{strategy.value}: "
                f"{correct}/{len(records)} correct"
            )
            arms.append(
                QualityArm(
                    retriever=name,
                    strategy=strategy,
                    judge_model=JUDGE_MODEL,
                    generation_model=GENERATION_MODEL,
                    records=records,
                )
            )
    except BudgetExceededError as error:
        typer.echo(f"\nstopped by the budget cap: {error}")
        raise typer.Exit(code=1) from error
    finally:
        generator.close()
        judge_model.close()
        embedder.close()
        store.close()

    kappa, low, high, labelled = _judge_agreement(labels_path, JUDGE_MODEL, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answers_results.json").write_text(
        json.dumps([arm.model_dump(mode="json") for arm in arms], indent=2), encoding="utf-8"
    )
    (out_dir / "answers.md").write_text(
        render_quality_report(
            arms,
            golden,
            judge_kappa=kappa,
            judge_kappa_low=low,
            judge_kappa_high=high,
            judge_labels=labelled,
        ),
        encoding="utf-8",
    )
    spent = sum(arm.total_cost_usd for arm in arms)
    typer.echo(f"\nwrote {out_dir / 'answers.md'} and answers_results.json")
    typer.echo(f"spend this run: ${spent:.6f} (cached answers and judgements cost $0)")


if __name__ == "__main__":
    sys.exit(app())
