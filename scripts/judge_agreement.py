"""Measure whether an LLM judge agrees with a human, before trusting it with Tier 2.

Two commands, with your labelling in between:

    uv run python scripts/judge_agreement.py prepare --dry-run   # what it would cost
    uv run python scripts/judge_agreement.py prepare             # generates the answers
    # ... you fill in human_verdict for each item in evals/judge_labels.yaml ...
    uv run python scripts/judge_agreement.py score               # runs both judges

The sample is deliberately adversarial. Every question Tier 1 shows the retriever missing
is included, and every no-answer question, because a sample where the assistant is right
every time measures nothing: a judge that replies "correct" unconditionally would score
95% agreement on it. That is what Cohen's kappa is reported for, and why the sample is
stratified rather than random.

This sample is **not** the Tier-2 answer-quality measurement. It over-represents failures
on purpose, so its correctness rate says nothing about the system.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from hybridrag.chunk_store import ChunkStore
from hybridrag.config import get_settings
from hybridrag.embedding import Embedder, FastEmbedEmbedder
from hybridrag.embedding_cache import CachedEmbedder
from hybridrag.embedding_openai import OpenAIEmbedder
from hybridrag.evaluation import GoldenSet, QuestionCategory
from hybridrag.evaluation.judge import Judge, JudgeError, LabelItem, LabelSheet, Verdict
from hybridrag.evaluation.report import JudgeRun, render_judge_report
from hybridrag.evaluation.stats import bootstrap_kappa, label_agreement
from hybridrag.generation import Answerer, CachedLanguageModel, OpenAIModel
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

# The generator under test. Cheap on purpose: what Tier 2 measures is whether the cheap
# model is good enough, and the judge experiment must grade the answers that will actually
# be reported, not better ones from a model we would not ship.
GENERATION_MODEL = "gpt-5-nano-2025-08-07"

# Candidate judges, cheapest first. The experiment exists to find out whether the cheap one
# is good enough, so the expensive one is only a yardstick.
CANDIDATE_JUDGES = ("gpt-5-nano-2025-08-07", "gpt-5-mini-2025-08-07")

# Generous, because gpt-5 models spend this budget on reasoning before writing anything:
# a cap sized for a three-line verdict returns an empty reply with finish_reason=length.
JUDGE_MAX_TOKENS = 2048

# Landis & Koch's conventional bar for "substantial" agreement. Not a law of nature -- it
# is quoted so the threshold is someone else's published convention rather than one chosen
# after seeing the result.
SUBSTANTIAL_KAPPA = 0.60


def _embedder() -> Embedder:
    settings = get_settings()
    if settings.openai_api_key is None:
        return FastEmbedEmbedder(settings.embedding_model)
    return OpenAIEmbedder(settings.openai_api_key)


def _openai(model_name: str, max_output_tokens: int) -> OpenAIModel:
    settings = get_settings()
    if settings.openai_api_key is None:
        raise typer.BadParameter("OPENAI_API_KEY is not set. Put it in .env.")
    return OpenAIModel(settings.openai_api_key, model_name, max_output_tokens=max_output_tokens)


def _retrieval_misses(results_path: Path, arm: str) -> list[str]:
    """Question ids the Tier-1 winner failed to retrieve by rank 10.

    Read from the committed results rather than re-derived, so the judge sample is tied to
    a specific measured run and the two cannot drift apart.
    """
    if not results_path.is_file():
        return []
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    for entry in payload.get("arms", []):
        if f"{entry['retriever']}/{entry['strategy']}" != arm:
            continue
        return [
            question["question_id"]
            for question in entry["questions"]
            # Model-dumped JSON turns the integer cutoffs into strings.
            if not question["hits"].get("10", question["hits"].get(10, False))
        ]
    return []


def _select(golden: GoldenSet, misses: list[str], size: int) -> list[tuple[str, str]]:
    """Choose the sample, and record why each item is in it.

    Failures first: a judge is easy to agree with when everything is correct, and the
    decisions that matter are the ones near the boundary. Filling the remainder in
    round-robin across categories keeps lookup questions from crowding out the harder
    kinds, and sorting by id makes the choice reproducible without a random seed.
    """
    chosen: list[tuple[str, str]] = []
    taken: set[str] = set()

    for question in golden.verified():
        if question.question_id in misses:
            chosen.append((question.question_id, "retrieval missed this at rank 10"))
            taken.add(question.question_id)

    for question in golden.verified():
        if question.category is QuestionCategory.NO_ANSWER and question.question_id not in taken:
            chosen.append((question.question_id, "unanswerable: a refusal is the correct answer"))
            taken.add(question.question_id)

    remaining = {
        category: sorted(
            question.question_id
            for question in golden.by_category(category)
            if question.verified and question.question_id not in taken
        )
        for category in QuestionCategory
        if category is not QuestionCategory.NO_ANSWER
    }
    while len(chosen) < size and any(remaining.values()):
        for ids in remaining.values():
            if not ids or len(chosen) >= size:
                continue
            chosen.append((ids.pop(0), "retrieved successfully"))
    return chosen[:size]


@app.command()
def prepare(
    sample: Annotated[int, typer.Option(help="How many answers to label.")] = 20,
    strategy: Annotated[
        ChunkingStrategy, typer.Option(help="Chunking strategy to answer from.")
    ] = ChunkingStrategy.FIXED,
    retriever_name: Annotated[str, typer.Option(help="hybrid, dense or sparse.")] = "hybrid",
    k: Annotated[int, typer.Option(help="Context blocks per answer.")] = 5,
    golden_path: Annotated[Path, typer.Option()] = Path("evals/golden_set.yaml"),
    results_path: Annotated[Path, typer.Option()] = Path("evals/reports/retrieval_results.json"),
    out: Annotated[Path, typer.Option(help="The labelling sheet to write.")] = Path(
        "evals/judge_labels.yaml"
    ),
    dry_run: Annotated[
        bool, typer.Option(help="Show the sample and the estimated cost, spend nothing.")
    ] = False,
    force: Annotated[
        bool, typer.Option(help="Overwrite a sheet that already carries human labels.")
    ] = False,
    index_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Generate the answers a human will hand-label."""
    settings = get_settings()
    root = index_dir or settings.index_dir
    golden = GoldenSet.load(golden_path)
    arm = f"{retriever_name}/{strategy.value}"

    misses = _retrieval_misses(results_path, arm)
    selected = _select(golden, misses, sample)
    by_id = {question.question_id: question for question in golden.verified()}

    typer.echo(f"sample of {len(selected)} for arm {arm}:")
    for question_id, why in selected:
        typer.echo(f"  {question_id:<16} {by_id[question_id].category.value:<10} {why}")

    # Measured, not guessed: gpt-5-nano answered this corpus at $0.000125 per question in
    # the vertical slice, and the judge prompt is roughly twice the size of an answer's.
    typer.echo(
        f"\nestimated cost: generation {len(selected)} x $0.000125 = "
        f"${len(selected) * 0.000125:.4f}; judging both candidates later, "
        f"~$0.005 (nano) + ~$0.026 (mini). Cached answers cost $0."
    )
    if dry_run:
        typer.echo("\n--dry-run: nothing generated, nothing spent.")
        return

    # The labels are the only thing here a person's time went into, and `prepare` rewrites
    # the sheet from scratch. Re-running it to regenerate answers would silently destroy
    # them, so it refuses instead.
    if out.is_file():
        existing = LabelSheet.load(out).labelled()
        if existing and not force:
            typer.echo(
                f"\nerror: {out} already holds {len(existing)} human label(s), and prepare "
                "rewrites the file. Pass --force only if you mean to discard them; to add "
                "labels use `label`, and to re-read the answers use `extract`."
            )
            raise typer.Exit(code=1)

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
    if retriever_name != "hybrid":
        retriever = retriever.ablation(retriever_name)

    model = CachedLanguageModel(
        _openai(GENERATION_MODEL, settings.answer_max_tokens),
        settings.cache_dir / "completions.sqlite",
    )
    answerer = Answerer(
        retriever, model, k=k, confidence_threshold=settings.retrieval_confidence_threshold
    )

    items: list[LabelItem] = []
    spent = 0.0
    for question_id, why in selected:
        question = by_id[question_id]
        answer = answerer.answer(question.question)
        spent += answer.cost_usd
        items.append(
            LabelItem(
                question_id=question_id,
                category=question.category,
                question=question.question,
                reference_answer=question.answer,
                system_answer=answer.text,
                refused=not answer.answered,
                citations=[f"[{c.number}] {c.source}" for c in answer.citations],
                context=render_context(answer.retrieved),
                selected_because=why,
            )
        )
        mark = "refused" if not answer.answered else f"{len(answer.citations)} citation(s)"
        typer.echo(f"  {question_id:<16} {mark}")

    LabelSheet(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        generation_model=GENERATION_MODEL,
        arm=arm,
        items=items,
    ).save(out)

    typer.echo(f"\nwrote {out} -- {len(items)} items, ${spent:.6f} spent")
    typer.echo(
        "\nNow label each item: set `human_verdict` to correct, partial or incorrect, and\n"
        "optionally `human_grounded` to true or false. Leave an item's verdict empty to\n"
        "exclude it. Then run: uv run python scripts/judge_agreement.py score"
    )
    model.close()
    embedder.close()
    store.close()


def _render_item(item: LabelItem, position: int, *, with_context: bool) -> str:
    """One item, small enough to actually read.

    The sheet carries the full retrieved context because grounding cannot be judged
    without it -- but a *correctness* verdict needs only the question, the reference and
    the answer, and burying those under 7,000 characters of passages is what made the raw
    file unreadable. Context is therefore opt-in.
    """
    lines = [
        f"### {position}. `{item.question_id}` — {item.category.value}",
        f"*Sampled because: {item.selected_because}.*",
        "",
        f"**Question:** {item.question}",
        "",
    ]
    reference = item.reference_answer.strip()
    lines += [
        "**Reference answer:** "
        + (
            reference
            or "*(empty — the documentation does not answer this, so declining "
            "is the correct behaviour)*"
        ),
        "",
        "**Assistant's answer:**",
        "",
        "> " + "\n> ".join(item.system_answer.strip().splitlines()),
        "",
    ]
    if item.citations:
        lines += ["**Citations:** " + "; ".join(item.citations), ""]
    if with_context:
        lines += [
            "<details><summary>Context the assistant was shown</summary>",
            "",
            "```",
            item.context.strip(),
            "```",
            "",
            "</details>",
            "",
        ]
    return "\n".join(lines)


@app.command()
def extract(
    labels: Annotated[Path, typer.Option()] = Path("evals/judge_labels.yaml"),
    out: Annotated[Path, typer.Option()] = Path("evals/judge_labels_review.md"),
    context: Annotated[
        bool, typer.Option(help="Include the retrieved passages, needed only for grounding.")
    ] = False,
) -> None:
    """Write a compact review document -- what a labeller actually has to read."""
    sheet = LabelSheet.load(labels)
    parts = [
        f"# Judge-agreement labelling — {len(sheet.items)} items",
        "",
        f"Answers from `{sheet.generation_model}` on arm `{sheet.arm}`, generated "
        f"{sheet.generated_at}.",
        "",
        "Verdict for each item: `correct`, `partial`, or `incorrect`.",
        "",
        "* **correct** — conveys what the reference conveys. Extra detail, length, wording "
        "and formatting do not count against it.",
        "* **partial** — gets part of it, but omits something the reference treats as "
        "essential, or adds a claim that is wrong.",
        "* **incorrect** — contradicts the reference, or does not answer the question.",
        "* Where the reference answer is **empty**, the documentation does not answer the "
        "question: declining is **correct**, answering anyway is **incorrect**.",
        "",
        "---",
        "",
    ]
    parts += [
        _render_item(item, index, with_context=context)
        for index, item in enumerate(sheet.items, start=1)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    size = out.stat().st_size
    typer.echo(f"wrote {out} -- {len(sheet.items)} items, {size / 1024:.0f} KB")


@app.command(name="apply")
def apply_verdicts(
    verdicts: Annotated[Path, typer.Argument(help="YAML of question_id: verdict.")],
    labels: Annotated[Path, typer.Option()] = Path("evals/judge_labels.yaml"),
    field: Annotated[str, typer.Option(help="Which column to fill: human or proposed.")] = "human",
) -> None:
    """Merge a verdict list back into the sheet.

    Accepts `question_id: correct` lines, or a mapping per item carrying `verdict`,
    `grounded` and `notes`. Unknown ids are refused rather than ignored, because a typo
    that silently drops a label would shrink the sample without saying so.
    """
    payload = yaml.safe_load(verdicts.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"{verdicts} is not a mapping of question_id to verdict")

    sheet = LabelSheet.load(labels)
    by_id = {item.question_id: item for item in sheet.items}
    unknown = sorted(set(payload) - set(by_id))
    if unknown:
        raise typer.BadParameter(f"no such question_id in {labels}: {unknown}")

    applied = 0
    for question_id, value in payload.items():
        item = by_id[question_id]
        entry = value if isinstance(value, dict) else {"verdict": value}
        verdict = Verdict(str(entry["verdict"]).strip().lower())
        if field == "human":
            item.human_verdict = verdict
            if "grounded" in entry:
                item.human_grounded = bool(entry["grounded"])
        else:
            item.proposed_verdict = verdict
            item.proposed_reason = str(entry.get("reason", ""))
        if entry.get("notes"):
            item.notes = str(entry["notes"])
        applied += 1

    sheet.save(labels)
    typer.echo(f"applied {applied} {field} verdict(s) to {labels}")
    if field == "human":
        typer.echo(f"{len(sheet.labelled())}/{len(sheet.items)} items now labelled")


@app.command()
def label(
    labels: Annotated[Path, typer.Option()] = Path("evals/judge_labels.yaml"),
    context: Annotated[bool, typer.Option(help="Show the retrieved passages too.")] = False,
    review: Annotated[
        bool, typer.Option(help="Re-visit items that already have a verdict.")
    ] = False,
) -> None:
    """Label interactively, one item at a time. Progress is saved after every answer."""
    sheet = LabelSheet.load(labels)
    queue = [item for item in sheet.items if review or not item.labelled]
    if not queue:
        typer.echo(f"every item in {labels} is labelled. Pass --review to revisit them.")
        return

    keys = {"c": Verdict.CORRECT, "p": Verdict.PARTIAL, "i": Verdict.INCORRECT}
    typer.echo(f"{len(queue)} item(s) to go. c=correct  p=partial  i=incorrect  s=skip  q=quit\n")
    for position, item in enumerate(queue, start=1):
        typer.echo("=" * 96)
        typer.echo(_render_item(item, position, with_context=context))
        if item.proposed_verdict is not None:
            typer.echo(
                f"  proposed: {item.proposed_verdict.value} -- {item.proposed_reason}\n"
                "  (a proposal, not a label: press the key you actually agree with)"
            )
        choice = typer.prompt("verdict [c/p/i/s/q]", default="s").strip().lower()[:1]
        if choice == "q":
            break
        if choice in keys:
            item.human_verdict = keys[choice]
            # Saved every time, so an interrupted session keeps everything answered so far.
            sheet.save(labels)
    typer.echo(f"\n{len(sheet.labelled())}/{len(sheet.items)} labelled, saved to {labels}")


@app.command()
def score(
    labels: Annotated[Path, typer.Option()] = Path("evals/judge_labels.yaml"),
    judge_model: Annotated[
        list[str] | None, typer.Option(help="Candidate judge. Repeatable.")
    ] = None,
    out_dir: Annotated[Path, typer.Option()] = Path("evals/reports"),
) -> None:
    """Run each candidate judge over the labelled items and report agreement."""
    sheet = LabelSheet.load(labels)
    items = sheet.labelled()
    if not items:
        typer.echo(
            f"error: no item in {labels} has a human_verdict yet. Label them first -- the "
            "whole point is to compare a judge against a person."
        )
        raise typer.Exit(code=1)
    unlabelled = len(sheet.items) - len(items)
    typer.echo(f"{len(items)} labelled item(s), {unlabelled} skipped\n")

    settings = get_settings()
    human = [item.human_verdict.value for item in items if item.human_verdict is not None]
    runs: list[JudgeRun] = []
    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "arm": sheet.arm,
        "generation_model": sheet.generation_model,
        "labelled": len(items),
        "judges": [],
    }

    for name in judge_model or CANDIDATE_JUDGES:
        model = CachedLanguageModel(
            _openai(name, JUDGE_MAX_TOKENS), settings.cache_dir / "completions.sqlite"
        )
        judge = Judge(model, max_output_tokens=JUDGE_MAX_TOKENS)
        verdicts: list[str] = []
        spent = 0.0
        failures: list[str] = []
        for item in items:
            try:
                judged = judge.judge(item)
            except JudgeError as error:
                # An unreadable reply is a real result about that judge, not a crash.
                failures.append(f"{item.question_id}: {error}")
                verdicts.append("unparseable")
                continue
            verdicts.append(judged.decision.verdict.value)
            spent += judged.cost_usd
        model.close()

        result = label_agreement(verdicts, human)
        interval = bootstrap_kappa(verdicts, human)
        runs.append(JudgeRun(model=name, verdicts=verdicts, cost_usd=spent, unreadable=failures))
        typer.echo(f"{name}")
        typer.echo(
            f"  agreement with you : {result.raw:.0%} raw, kappa {interval.mean:.3f} "
            f"[{interval.low:+.3f}, {interval.high:+.3f}]"
        )
        typer.echo(f"  cost               : ${spent:.6f}")
        if failures:
            typer.echo(f"  unreadable replies : {len(failures)}")
        for position in result.disagreements:
            typer.echo(
                f"    {items[position].question_id:<16} you: {human[position]:<10} "
                f"judge: {verdicts[position]}"
            )
        typer.echo("")

        judges = report["judges"]
        assert isinstance(judges, list)
        judges.append(
            {
                "model": name,
                "raw_agreement": result.raw,
                "kappa": result.kappa,
                "cost_usd": spent,
                "unreadable": failures,
                "verdicts": dict(zip([i.question_id for i in items], verdicts, strict=True)),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judge_agreement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "judge_agreement.md").write_text(render_judge_report(sheet, runs), encoding="utf-8")
    typer.echo(f"wrote {out_dir / 'judge_agreement.md'} and judge_agreement.json")

    judges = report["judges"]
    assert isinstance(judges, list)
    good = [entry for entry in judges if entry["kappa"] >= SUBSTANTIAL_KAPPA]
    if good:
        cheapest = min(good, key=lambda entry: float(entry["cost_usd"]))
        typer.echo(
            f"\nrecommendation: {cheapest['model']} -- cheapest judge reaching "
            f"kappa >= {SUBSTANTIAL_KAPPA} (got {cheapest['kappa']:.2f})"
        )
    else:
        typer.echo(
            f"\nrecommendation: none of the candidates reached kappa >= {SUBSTANTIAL_KAPPA}. "
            "Report Tier 2 with the human labels as the standard, or improve the judge "
            "prompt before trusting either."
        )


if __name__ == "__main__":
    sys.exit(app())
