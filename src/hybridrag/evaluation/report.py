"""Turns grid results into the report a reader can check.

Two outputs from one object: `results.json` for reproduction and any later analysis, and
`report.md` for a human. The markdown is generated, never hand-edited -- a number typed
into a report by hand is a number nothing verifies.

Every claim in the rendered report is either a measurement or a comparison with its
interval attached. Where an interval spans zero the report says the arms cannot be
separated, rather than reporting the larger mean as a win: at 29 questions that is the
result the data supports most of the time, and printing it is the point of measuring.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from statistics import fmean

from pydantic import BaseModel, Field, NonNegativeInt

from hybridrag.evaluation.golden import GoldenSet, QuestionCategory
from hybridrag.evaluation.harness import METRICS, ArmResult
from hybridrag.evaluation.judge import LabelSheet, Verdict
from hybridrag.evaluation.quality import QUALITY_METRICS, FailureKind, QualityArm
from hybridrag.evaluation.stats import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Delta,
    Interval,
    bootstrap_ci,
    bootstrap_kappa,
    label_agreement,
    paired_delta,
)
from hybridrag.models import ChunkingStrategy

# What the headline table shows with intervals. Recall@5 leads because the generator is
# given five blocks, so it is the number that decides what the answer can be grounded in.
HEADLINE: tuple[str, ...] = ("recall@5", "recall@10", "ndcg@10", "recall@budget")


class Provenance(BaseModel):
    """Everything needed to say what produced these numbers, and to produce them again."""

    generated_at: str
    git_sha: str
    git_dirty: bool
    corpus_ref: str
    documents: int = 0
    embedding_model: str
    chunk_tokens: int
    chunk_overlap_tokens: int
    semantic_percentile: float
    rank_constant: int
    candidates: int
    depth: int
    token_budget: int
    min_ratio: float
    confidence_threshold: float = 0.0
    seed: int = DEFAULT_SEED
    resamples: int = DEFAULT_RESAMPLES
    counts: dict[str, int] = Field(default_factory=dict, description="Questions per category.")


class ChunkStats(BaseModel):
    """What one strategy actually produced, which is half the explanation of its scores.

    Recall@k and Recall@budget diverge exactly where these differ: a strategy emitting
    fewer, larger chunks reaches more text per rank and less text per token.
    """

    strategy: ChunkingStrategy
    chunks: int
    mean_tokens: float
    median_tokens: float
    p95_tokens: float
    total_tokens: int


class GridResult(BaseModel):
    """The full grid: every arm, plus what produced it."""

    provenance: Provenance
    arms: list[ArmResult]
    corpus: list[ChunkStats] = Field(default_factory=list)

    def find(self, retriever: str, strategy: ChunkingStrategy) -> ArmResult:
        for arm in self.arms:
            if arm.retriever == retriever and arm.strategy is strategy:
                return arm
        raise KeyError(f"no arm {retriever}/{strategy.value} in this grid")

    @property
    def strategies(self) -> list[ChunkingStrategy]:
        seen: list[ChunkingStrategy] = []
        for arm in self.arms:
            if arm.strategy not in seen:
                seen.append(arm.strategy)
        return seen

    @property
    def retrievers(self) -> list[str]:
        seen: list[str] = []
        for arm in self.arms:
            if arm.retriever not in seen:
                seen.append(arm.retriever)
        return seen


def git_provenance() -> tuple[str, bool]:
    """The commit these numbers came from, and whether the tree was dirty when they did."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False
    return sha, bool(status)


def provenance(
    golden: GoldenSet,
    *,
    documents: int,
    embedding_model: str,
    chunk_tokens: int,
    chunk_overlap_tokens: int,
    semantic_percentile: float,
    rank_constant: int,
    candidates: int,
    depth: int,
    token_budget: int,
    min_ratio: float,
    confidence_threshold: float = 0.0,
) -> Provenance:
    sha, dirty = git_provenance()
    counts: dict[str, int] = {}
    for question in golden.verified():
        counts[question.category.value] = counts.get(question.category.value, 0) + 1
    return Provenance(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=sha,
        git_dirty=dirty,
        corpus_ref=golden.corpus_ref,
        documents=documents,
        embedding_model=embedding_model,
        chunk_tokens=chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        semantic_percentile=semantic_percentile,
        rank_constant=rank_constant,
        candidates=candidates,
        depth=depth,
        token_budget=token_budget,
        min_ratio=min_ratio,
        confidence_threshold=confidence_threshold,
        counts=counts,
    )


def interval(arm: ArmResult, metric: str, prov: Provenance) -> Interval:
    return bootstrap_ci(
        arm.series(metric), resamples=prov.resamples, seed=prov.seed, confidence=0.95
    )


def delta(treatment: ArmResult, baseline: ArmResult, metric: str, prov: Provenance) -> Delta:
    return paired_delta(
        treatment.series(metric),
        baseline.series(metric),
        resamples=prov.resamples,
        seed=prov.seed,
        confidence=0.95,
    )


def _table(rows: Sequence[Sequence[str]], header: Sequence[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell.strip() for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _best(grid: GridResult, metric: str = "recall@5") -> ArmResult:
    """The leading arm on one metric. Whether the lead *means* anything is tested below."""
    return max(
        grid.arms,
        key=lambda arm: (
            sum(arm.series(metric)) / max(1, len(arm.questions)),
            sum(arm.series("ndcg@10")) / max(1, len(arm.questions)),
        ),
    )


def render(grid: GridResult, golden: GoldenSet) -> str:
    prov = grid.provenance
    total = sum(prov.counts.values())
    answerable = total - prov.counts.get("no_answer", 0)
    dirty = " **(uncommitted changes)**" if prov.git_dirty else ""
    parts: list[str] = []

    parts.append(
        f"""# Retrieval evaluation — Tier 1

Deterministic retrieval metrics over the hand-verified golden set. No language model is
involved anywhere in this report, so it costs nothing to reproduce and cannot drift with a
model release.

**Provenance** · generated {prov.generated_at} · commit `{prov.git_sha}`{dirty} ·
corpus `{prov.corpus_ref}` · embedder `{prov.embedding_model}` ·
chunks {prov.chunk_tokens} tokens / {prov.chunk_overlap_tokens} overlap ·
semantic percentile {prov.semantic_percentile} · RRF rank constant {prov.rank_constant} ·
candidate depth {prov.candidates} · scored to depth {prov.depth} ·
budget {prov.token_budget} tokens · coverage threshold {prov.min_ratio} ·
bootstrap {prov.resamples:,} resamples, seed {prov.seed}

**Questions** · {total} verified, of which **{answerable} are scored here**
({", ".join(f"{count} {name}" for name, count in sorted(prov.counts.items()))}).
`no_answer` questions carry no retrieval ground truth (D29) and appear only in the
refusal-calibration section."""
    )

    if grid.corpus:
        parts.append(
            "## What each strategy produced\n\n"
            f"The same {prov.documents} documents, cut three ways. These numbers explain most of "
            "the gap between Recall@k and Recall@budget below."
        )
        parts.append(
            _table(
                [
                    [
                        stats.strategy.value,
                        f"{stats.chunks:,}",
                        f"{stats.mean_tokens:.0f}",
                        f"{stats.median_tokens:.0f}",
                        f"{stats.p95_tokens:.0f}",
                        f"{stats.total_tokens:,}",
                    ]
                    for stats in grid.corpus
                ],
                ["chunking", "chunks", "mean tokens", "median", "p95", "total tokens"],
            )
        )

    parts.append("## Headline\n\nMean over questions, with a 95% percentile bootstrap interval.")
    rows = [
        [arm.name, *(str(interval(arm, metric, prov)) for metric in HEADLINE)] for arm in grid.arms
    ]
    parts.append(_table(rows, ["arm", *HEADLINE]))

    parts.append(
        "## Every metric, point estimates\n\n"
        "`recall@budget` fixes the retrieved-context budget instead of the number of "
        "chunks (D14), so a strategy earns nothing for emitting larger chunks. Latency is "
        "the median wall-clock time of one retrieval, query embeddings served from cache."
    )
    rows = []
    for arm in grid.arms:
        scores = [f"{sum(arm.series(m)) / max(1, len(arm.questions)):.3f}" for m in METRICS]
        rows.append([arm.name, *scores, f"{arm.median_latency_ms:.1f} ms"])
    parts.append(_table(rows, ["arm", *METRICS, "latency"]))

    parts.append(
        "## Does hybrid actually beat its parts?\n\n"
        "Paired bootstrap: the same resampled questions scored under both arms, which "
        "cancels question difficulty. `*` marks an interval that excludes zero — the only "
        "circumstance under which a difference here is a claim rather than a hint."
    )
    rows = []
    for strategy in grid.strategies:
        hybrid = grid.find("hybrid", strategy)
        for baseline_name in ("dense", "sparse"):
            baseline = grid.find(baseline_name, strategy)
            for metric in ("recall@5", "recall@budget"):
                difference = delta(hybrid, baseline, metric, prov)
                rows.append(
                    [
                        f"hybrid - {baseline_name}",
                        strategy.value,
                        metric,
                        str(difference),
                        f"{difference.prob_positive:.0%}",
                    ]
                )
    parts.append(_table(rows, ["contrast", "chunking", "metric", "difference [95% CI]", "P(>0)"]))

    parts.append(
        "## Does the chunking strategy matter?\n\n"
        "The same paired comparison across chunking strategies, holding the retriever fixed "
        "at hybrid."
    )
    rows = []
    strategies = grid.strategies
    for index, strategy in enumerate(strategies):
        for other in strategies[index + 1 :]:
            for metric in ("recall@5", "recall@budget"):
                difference = delta(
                    grid.find("hybrid", strategy), grid.find("hybrid", other), metric, prov
                )
                rows.append(
                    [
                        f"{strategy.value} - {other.value}",
                        metric,
                        str(difference),
                        f"{difference.prob_positive:.0%}",
                    ]
                )
    parts.append(_table(rows, ["contrast", "metric", "difference [95% CI]", "P(>0)"]))

    leader = _best(grid)
    parts.append(
        f"## By question category — {leader.name}\n\n"
        "Each category is scored by its own rule (D29): a lookup needs its one span, "
        "multi-hop needs **every** span, ambiguous needs any acceptable one."
    )
    rows = []
    for category in (
        QuestionCategory.LOOKUP,
        QuestionCategory.MULTI_HOP,
        QuestionCategory.AMBIGUOUS,
    ):
        series = leader.series("recall@5", category=category)
        if not series:
            continue
        rows.append(
            [
                category.value,
                str(len(series)),
                f"{sum(series) / len(series):.3f}",
                f"{sum(leader.series('recall@10', category=category)) / len(series):.3f}",
                f"{sum(leader.series('ndcg@10', category=category)) / len(series):.3f}",
            ]
        )
    parts.append(_table(rows, ["category", "n", "recall@5", "recall@10", "nDCG@10"]))

    misses = leader.misses(10)
    unreachable = set(leader.unreachable)
    parts.append(f"## What {leader.name} still misses at rank 10\n")
    if not misses:
        parts.append("Nothing: every answerable question is retrieved within the top 10.")
    else:
        by_id = {q.question_id: q for q in golden.verified()}
        rows = []
        for question_id in misses:
            question = by_id[question_id]
            cause = "no chunk covers the span" if question_id in unreachable else "ranking"
            rows.append([question_id, question.category.value, cause, question.question[:70]])
        parts.append(_table(rows, ["id", "category", "cause", "question"]))

    parts.append(
        "## Refusal calibration\n\n"
        "`no_answer` questions have nothing to retrieve, but they measure the "
        "pre-generation refusal gate (D23), which thresholds the best dense similarity a "
        "query attracts. A threshold is defensible only if these two distributions "
        "separate."
    )
    rows = []
    for strategy in grid.strategies:
        arm = grid.find("hybrid", strategy)
        answered = [p.top_dense_score for p in arm.answerable_scores if p.top_dense_score]
        refused = [p.top_dense_score for p in arm.refusals if p.top_dense_score]
        if not answered or not refused:
            continue
        caught = sum(1 for score in refused if score < prov.confidence_threshold)
        rows.append(
            [
                strategy.value,
                f"{min(answered):.3f}",
                f"{sum(answered) / len(answered):.3f}",
                f"{max(refused):.3f}",
                f"{sum(refused) / len(refused):.3f}",
                f"{min(answered) - max(refused):+.3f}",
                str(sum(1 for score in answered if score < prov.confidence_threshold)),
                f"{caught}/{len(refused)}",
            ]
        )
    parts.append(
        _table(
            rows,
            [
                "chunking",
                "answerable min",
                "answerable mean",
                "no_answer max",
                "no_answer mean",
                "gap",
                "false refusals",
                "gate catches",
            ],
        )
    )
    parts.append(
        f"A positive gap would mean one threshold separates the two sets perfectly. A "
        f"negative one means no threshold can -- which is the expected result here and the "
        f"reason two refusal paths exist (D23). These `no_answer` questions are *in-domain "
        f"but unanswerable*: they are about FastAPI, so they retrieve FastAPI chunks and "
        f"score like real questions. The pre-generation gate was measured against "
        f"*out-of-domain* questions, which it does separate. **false refusals** counts "
        f"answerable questions that the configured threshold of "
        f"{prov.confidence_threshold} would wrongly refuse; **gate catches** is how many "
        f"unanswerable ones it stops before spending a request. The rest are the model's "
        f"own refusal sentinel to catch."
    )

    parts.append(
        f"""## How to read this, and what it does not say

* **{answerable} questions.** The intervals are wide because the sample is small, and they
  are reported rather than hidden. Most differences between adjacent arms here are not
  separable at 95%.
* **Coverage threshold {prov.min_ratio}.** A span counts as retrieved only when the union of
  the retrieved chunks contains **all** of it. Relaxing this would raise every number.
* **Approximate search.** Chroma's HNSW is approximate, so a fraction of a point of
  movement between runs is the index, not the code.
* **The golden set was drafted by a model and verified by hand** (D11). The questions are
  therefore ones a model found answerable in this corpus, which is a real selection effect
  even after human verification.
* Tier 2 — answer correctness, faithfulness and citation accuracy — is judged separately
  and is the only part of the evaluation that costs money."""
    )

    return "\n\n".join(parts) + "\n"


class JudgeRun(BaseModel):
    """What one candidate judge said about every labelled item, and what it cost."""

    model: str
    verdicts: list[str] = Field(description="One verdict per item, in the sheet's order.")
    cost_usd: float = 0.0
    unreadable: list[str] = Field(default_factory=list)


def render_judge_report(
    sheet: LabelSheet,
    runs: Sequence[JudgeRun],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    substantial: float = 0.60,
) -> str:
    """The judge-validation report: agreement, its uncertainty, and what may be claimed.

    The trivial baseline is printed beside every judge on purpose. On a sample this
    lopsided a judge that answers "correct" unconditionally scores 90% raw agreement, so a
    raw percentage alone is not evidence of anything -- and a reader who does not know the
    base rate cannot tell the difference without seeing it.
    """
    items = [item for item in sheet.items if item.human_verdict is not None]
    human = [item.human_verdict.value for item in items if item.human_verdict is not None]
    sha, dirty = git_provenance()
    counts: dict[str, int] = {}
    for label in human:
        counts[label] = counts.get(label, 0) + 1

    distribution = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
    trivial = label_agreement([Verdict.CORRECT.value] * len(human), human)
    proposed = sum(
        1
        for item in items
        if item.proposed_verdict is not None and item.proposed_verdict == item.human_verdict
    )
    with_proposals = sum(1 for item in items if item.proposed_verdict is not None)

    parts: list[str] = [
        f"""# Judge validation — does an LLM judge agree with a human?

Tier 2 grades answers with a language model, which is circular unless the judge is itself
measured. Before any answer-quality number is reported, a human labelled a sample by hand
and each candidate judge labelled the same sample; what follows is how far they agreed
(D11).

**Provenance** · commit `{sha}`{" **(uncommitted changes)**" if dirty else ""} ·
answers from `{sheet.generation_model}` on arm `{sheet.arm}` · generated {sheet.generated_at} ·
{len(items)} labelled items · bootstrap {resamples:,} resamples, seed {seed}

**Human label distribution** · {distribution}""",
        """## Agreement

Cohen's kappa removes the agreement two labellers would reach by chance alone. The last row
is the reason it is reported: a judge that replies "correct" to everything needs no
intelligence at all and still scores the raw agreement shown.""",
    ]

    rows: list[list[str]] = []
    for run in runs:
        agreement = label_agreement(run.verdicts, human)
        interval = bootstrap_kappa(run.verdicts, human, resamples=resamples, seed=seed)
        rows.append(
            [
                f"`{run.model}`",
                f"{agreement.raw:.0%}",
                f"{interval.mean:.3f} [{interval.low:+.3f}, {interval.high:+.3f}]",
                f"${run.cost_usd:.4f}",
                str(len(run.unreadable)),
            ]
        )
    rows.append(
        [
            "*always answers “correct”*",
            f"{trivial.raw:.0%}",
            f"{trivial.kappa:.3f}",
            "$0.0000",
            "0",
        ]
    )
    parts.append(_table(rows, ["judge", "raw agreement", "kappa [95% CI]", "cost", "unreadable"]))

    parts.append("## Where each judge parted company with the human")
    for run in runs:
        agreement = label_agreement(run.verdicts, human)
        if not agreement.disagreements:
            parts.append(f"**`{run.model}`** agreed on every item.")
            continue
        detail = [
            [
                items[position].question_id,
                items[position].category.value,
                human[position],
                run.verdicts[position],
            ]
            for position in agreement.disagreements
        ]
        parts.append(f"**`{run.model}`** — {len(detail)} of {len(items)}")
        parts.append(_table(detail, ["id", "category", "human", "judge"]))

    parts.append(
        f"""## How these labels were made

{with_proposals} of {len(items)} items carried a model-proposed label for the human to
adjudicate, and the human's final label matched the proposal on **{proposed}** of them.
Proposals are stored in a separate field and never enter the statistics above; only the
human's decision does. That distinction is what the numbers here rest on, so it is recorded
rather than assumed."""
    )

    best = max(
        runs,
        key=lambda run: label_agreement(run.verdicts, human).kappa,
        default=None,
    )
    if best is not None:
        interval = bootstrap_kappa(best.verdicts, human, resamples=resamples, seed=seed)
        verdict_line = (
            f"clears the conventional {substantial} bar for substantial agreement"
            if interval.mean >= substantial
            else f"falls short of the conventional {substantial} bar"
        )
        caveat = (
            " Its interval reaches below that bar, so at this sample size the point "
            "estimate is the claim and the interval is the honest caveat."
            if interval.low < substantial
            else ""
        )
        parts.append(
            f"""## What this licenses

`{best.model}` {verdict_line} (kappa {interval.mean:.3f}).{caveat} Every Tier-2 number
produced by this judge should be reported with that agreement figure beside it, not
without.

Twenty items is a small sample and the labels are lopsided, which is why the interval is
wide rather than why it should be ignored: a judge indistinguishable from chance is still
conclusively identified as such, and that is the decision this experiment existed to make."""
        )
    return "\n\n".join(parts) + "\n"


# Read beside every Tier-2 number, never without it (D37).
_JUDGE_CAVEAT = (
    "**Correctness** is judged by `{judge}`, whose agreement with a human was measured "
    "before it was trusted: **kappa {kappa:.3f}** [{low:+.3f}, {high:+.3f}] on {n} "
    "hand-adjudicated items (D37). Every correctness figure below inherits that uncertainty "
    "on top of its own sampling error.\n\n"
    "**Grounding is not validated.** The same judge produces it, but no human labelled "
    "grounding, so its kappa is unknown and the number below is reported for completeness "
    "rather than as a claim. **Citation honesty is deterministic** -- a bracketed number "
    "either resolves to a block that was in the prompt or it does not -- and involves no "
    "judge at all."
)


def render_quality_report(
    arms: Sequence[QualityArm],
    golden: GoldenSet,
    *,
    judge_kappa: float,
    judge_kappa_low: float,
    judge_kappa_high: float,
    judge_labels: int,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> str:
    """The Tier-2 report: answer quality, and which stage caused each failure."""
    if not arms:
        raise ValueError("a quality report needs at least one arm")
    sha, dirty = git_provenance()
    leader = max(arms, key=lambda arm: fmean(arm.series("correct")) if arm.records else 0.0)
    answerable_ids = {
        question.question_id
        for question in golden.verified()
        if question.category is not QuestionCategory.NO_ANSWER
    }
    parts: list[str] = [
        f"""# Answer evaluation — Tier 2

What the system answers, whether the answer is grounded in what it retrieved, and where a
wrong answer went wrong.

**Provenance** · commit `{sha}`{" **(uncommitted changes)**" if dirty else ""} ·
generator `{arms[0].generation_model}` · judge `{arms[0].judge_model}` ·
{len(arms[0].records)} questions · bootstrap {resamples:,} resamples, seed {seed}

"""
        + _JUDGE_CAVEAT.format(
            judge=arms[0].judge_model,
            kappa=judge_kappa,
            low=judge_kappa_low,
            high=judge_kappa_high,
            n=judge_labels,
        ),
        """## Answer quality

`correct` and `correct or partial` are reported side by side deliberately: partial credit
flatters a system, and one blended score would hide which of the two moved. Cost per
answer is priced from token counts rather than from what this run paid, so it stays
meaningful when the answers come from cache; latency can only be measured on answers
actually generated, so the live count is shown beside it.""",
    ]

    rows: list[list[str]] = []
    for arm in arms:
        cells = [arm.name]
        for metric in QUALITY_METRICS:
            interval = bootstrap_ci(
                arm.series(metric), resamples=resamples, seed=seed, confidence=0.95
            )
            cells.append(f"{interval.mean:.3f} [{interval.low:.3f}, {interval.high:.3f}]")
        cells.append(f"${arm.modelled_cost_per_answer:.6f}")
        latency = (
            f"{arm.median_latency_s:.2f}s ({arm.live_answers}/{len(arm.records)} live)"
            if arm.live_answers
            else f"— (0/{len(arm.records)} live)"
        )
        cells.append(latency)
        rows.append(cells)
    parts.append(_table(rows, ["arm", *QUALITY_METRICS, "cost/answer", "median latency"]))

    if len(arms) > 1:
        parts.append(
            """## Does hybrid retrieval produce better *answers*?

Tier 1 showed hybrid retrieves better. This is the question that matters to a user, and it
is not the same question: better context only helps if the generator uses it. Paired
bootstrap over the same resampled questions; `*` marks an interval excluding zero."""
        )
        rows = []
        baseline = min(arms, key=lambda arm: fmean(arm.series("correct")))
        for arm in arms:
            if arm.name == baseline.name:
                continue
            for metric in ("correct", "correct_or_partial", "grounded"):
                difference = paired_delta(
                    arm.series(metric),
                    baseline.series(metric),
                    resamples=resamples,
                    seed=seed,
                )
                rows.append(
                    [
                        f"{arm.name} - {baseline.name}",
                        metric,
                        str(difference),
                        f"{difference.prob_positive:.0%}",
                    ]
                )
        parts.append(_table(rows, ["contrast", "metric", "difference [95% CI]", "P(>0)"]))

    parts.append(f"## By question category — {leader.name}")
    rows = []
    for category in QuestionCategory:
        records = leader.subset(category=category)
        if not records:
            continue
        rows.append(
            [
                category.value,
                str(len(records)),
                f"{fmean(leader.series('correct', records)):.3f}",
                f"{fmean(leader.series('correct_or_partial', records)):.3f}",
                f"{fmean(leader.series('grounded', records)):.3f}",
                f"{leader.refusal_rate(category):.3f}",
            ]
        )
    parts.append(
        _table(rows, ["category", "n", "correct", "correct or partial", "grounded", "refused"])
    )

    parts.append(
        """## Where the failures come from

The join between the two tiers, and the only table here that says what to fix. A wrong
answer is attributed to retrieval whenever Tier 1 records that this same arm failed to
retrieve the answer span -- refusing or fumbling a question whose evidence never arrived is
not a generation fault, and counting it as one would hide a retrieval problem behind a
prompt problem."""
    )
    rows = []
    for arm in arms:
        counts = arm.failures()
        rows.append(
            [
                arm.name,
                str(sum(counts.values())),
                *(str(counts.get(kind.value, 0)) for kind in FailureKind),
                str(sum(1 for record in arm.records if not record.cited_honestly)),
            ]
        )
    parts.append(
        _table(
            rows,
            [
                "arm",
                "failures",
                *(kind.value for kind in FailureKind),
                "fabricated citations",
            ],
        )
    )

    parts.append(f"## Every answer {leader.name} got wrong")
    wrong = [record for record in leader.records if record.verdict is not Verdict.CORRECT]
    if not wrong:
        parts.append("None: every answer was judged correct.")
    else:
        questions = {q.question_id: q for q in golden.verified()}
        rows = [
            [
                record.question_id,
                record.category.value,
                record.verdict.value,
                record.failure.value if record.failure else "-",
                (record.judge_reason or questions[record.question_id].question)[:90],
            ]
            for record in wrong
        ]
        parts.append(_table(rows, ["id", "category", "verdict", "attributed to", "judge's reason"]))

    parts.append(
        "## Refusal behaviour\n\n"
        "The two errors are not symmetric and are counted apart. These columns count "
        "**explicit** refusals -- the confidence gate firing, or the model emitting its "
        "sentinel. A model can also decline in prose without the sentinel, which is counted "
        "here as an answer and by the judge as a correct decline; that is why an arm can "
        "show fewer refusals than it has correct no-answer verdicts."
    )
    rows = []
    for arm in arms:
        unanswerable = arm.subset(category=QuestionCategory.NO_ANSWER)
        answerable = arm.answerable()
        false_refusals = [r for r in answerable if not r.answered]
        rows.append(
            [
                arm.name,
                f"{arm.refusal_rate(QuestionCategory.NO_ANSWER):.0%} ({len(unanswerable)} asked)",
                f"{len(false_refusals)}/{len(answerable)}",
                ", ".join(r.question_id for r in false_refusals) or "-",
            ]
        )
    parts.append(
        _table(
            rows,
            ["arm", "refused when unanswerable", "refused when answerable", "which ones"],
        )
    )

    parts.append(
        f"""## Limits of this measurement

* **{len(answerable_ids)} answerable questions and {len(arms[0].records) - len(answerable_ids)}
  unanswerable ones.** Intervals are wide at this size and are reported rather than hidden.
* **The judge is imperfect and its imperfection is measured**, not assumed away: kappa
  {judge_kappa:.3f} against a human on {judge_labels} adjudicated items.
* **Answers are served from a response cache.** That is what makes a re-run free and
  reproducible on a model that rejects `temperature` (D27), but it means the numbers
  describe one sampled generation per question, not an average over several.
* **`cited_honestly` here checks resolution, not support.** A citation pointing at a real
  block that does not actually contain the claim is counted honest in this column. Whether
  the cited block supports the claim attached to it is measured separately and per claim,
  in `citation_verification.md`: on this arm 0.917 of cited claims hold, and the six that
  do not are listed there by name."""
    )
    return "\n\n".join(parts) + "\n"


class VerifiedClaim(BaseModel):
    """One claim, whether its own citations held it up, and whether a random block did."""

    question_id: str
    category: str
    claim_index: NonNegativeInt
    claim_text: str
    cited: tuple[int, ...]
    supported: bool
    control_supported: bool | None = Field(
        default=None,
        description="Result of pairing this claim with blocks it never cited. None when "
        "the control was not run.",
    )
    reason: str = ""


class VerificationRun(BaseModel):
    """A claim-level citation verification pass over the golden set."""

    generated_at: str
    git_sha: str
    git_dirty: bool
    arm: str
    strategy: str
    generation_model: str
    verifier_model: str
    questions: NonNegativeInt
    answered: NonNegativeInt
    coverage: list[float] = Field(default_factory=list)
    precision: list[float] = Field(default_factory=list)
    claims: list[VerifiedClaim] = Field(default_factory=list)
    cost_usd: float = 0.0
    control_cost_usd: float = 0.0
    calls: NonNegativeInt = 0
    cached_calls: NonNegativeInt = Field(
        default=0,
        description="Calls served from the response cache. They report $0, which is true "
        "of this run and false of the check, so both counts are kept (D40).",
    )
    unreadable: NonNegativeInt = Field(
        default=0, description="Replies that could not be parsed and were skipped."
    )
    price_per_call: float = 0.0
    seed: int = DEFAULT_SEED
    resamples: int = DEFAULT_RESAMPLES

    @property
    def controlled(self) -> list[VerifiedClaim]:
        return [claim for claim in self.claims if claim.control_supported is not None]


def render_verification_report(run: VerificationRun) -> str:
    """The claim-level citation verification report.

    Structured around one question the supported-rate alone cannot answer: is the verifier
    reading, or agreeing? A checker that returns "supported" unconditionally produces a
    perfect-looking precision, so the negative control -- the same claims paired with
    blocks they never cited -- is reported before the headline number rather than after it.
    """
    dirty = " **(uncommitted changes)**" if run.git_dirty else ""
    lines = [
        "# Citation verification — does the cited block support the claim?",
        "",
        "Structural citation checking asks whether `[3]` points at a block that exists.",
        "This asks the question a reader assumes is already answered: whether block 3",
        "actually says the thing the sentence attached to it claims. A citation can resolve",
        "perfectly and still be attached to a passage that never makes the claim, which",
        "renders as a working source link under a fabricated statement.",
        "",
        f"**Provenance** · commit `{run.git_sha}`{dirty} ·",
        f"arm `{run.arm}` / `{run.strategy}` · generator `{run.generation_model}` ·",
        f"verifier `{run.verifier_model}` · {run.questions} questions, {run.answered} answered ·",
        f"bootstrap {run.resamples:,} resamples, seed {run.seed}",
        "",
    ]

    controlled = run.controlled
    lines += [
        "## Is the verifier reading, or agreeing?",
        "",
        "Every cited claim was checked twice: once against the blocks it actually cited,",
        "and once against blocks drawn from the corpus that it never cited. A verifier that",
        "rubber-stamps cannot tell those apart, and its supported rate means nothing however",
        "high it is. This is the check that licenses every number below it, and it needs no",
        "human labelling to run.",
        "",
    ]
    if controlled:
        real = [float(claim.supported) for claim in controlled]
        control = [float(bool(claim.control_supported)) for claim in controlled]
        separation = paired_delta(real, control, resamples=run.resamples, seed=run.seed)
        lines += [
            _table(
                [
                    ["cited blocks (real)", f"{fmean(real):.3f}", str(len(real))],
                    ["random blocks (control)", f"{fmean(control):.3f}", str(len(control))],
                    ["**separation**", f"**{separation}**", str(separation.n)],
                ],
                ["pairing", "supported rate", "n"],
            ),
            "",
            f"The interval on the separation {'excludes' if separation.low > 0 else 'includes'}"
            " zero.",
            "",
        ]
    else:
        lines += ["The control was not run, so nothing here licenses the rates below.", ""]

    lines += ["## Coverage and precision", ""]
    if run.coverage:
        coverage = bootstrap_ci(run.coverage, resamples=run.resamples, seed=run.seed)
        rows = [["citation coverage", str(coverage), str(coverage.n)]]
        if run.precision:
            precision = bootstrap_ci(run.precision, resamples=run.resamples, seed=run.seed)
            rows.append(["citation precision", str(precision), str(precision.n)])
        lines += [
            "Coverage counts every claim the answer made, so an unattributed sentence lowers",
            "it. Precision counts only the claims that were cited. They are reported apart",
            "because an answer that cites one sentence in six and gets it right scores badly",
            "on the first and perfectly on the second, and both facts matter.",
            "",
            _table(rows, ["measure", "mean [95% CI]", "n"]),
            "",
        ]

    unsupported = [claim for claim in run.claims if not claim.supported]
    lines += [
        "## Every claim whose citations did not hold",
        "",
        f"{len(unsupported)} of {len(run.claims)} cited claims.",
        "",
    ]
    if run.unreadable:
        lines += [
            f"A further **{run.unreadable}** repl(ies) could not be parsed and were skipped "
            "rather than guessed at, so they appear in no rate above.",
            "",
        ]
    if unsupported:
        lines.append(
            _table(
                [
                    [
                        claim.question_id,
                        ", ".join(str(number) for number in claim.cited),
                        " ".join(claim.claim_text.split())[:70],
                        " ".join(claim.reason.split())[:70],
                    ]
                    for claim in unsupported
                ],
                ["id", "cited", "claim", "why it does not hold"],
            )
        )
        lines.append("")

    total = run.cost_usd + run.control_cost_usd
    modelled = run.calls * run.price_per_call
    lines += [
        "## What this cost",
        "",
        "Two figures, because they answer different questions and D40 records what happens",
        "when only the first is reported: a cached call truthfully bills $0, so a re-run of",
        "a fully cached check reads as free and is not.",
        "",
        f"* **what this run paid**: ${total:.4f} "
        f"(${run.cost_usd:.4f} verification + ${run.control_cost_usd:.4f} control), with "
        f"{run.cached_calls} of {run.calls} calls served from cache",
        f"* **what the check costs**: ${modelled:.4f} — {run.calls} calls at the measured "
        f"${run.price_per_call:.6f} each",
        "",
        "## Limits of this measurement",
        "",
        "* **The verifier's agreement with a human is not measured.** The negative control",
        "  shows it discriminates rather than rubber-stamps, which is a weaker claim than",
        "  the kappa reported for the correctness judge and is not a substitute for it.",
        "* **A claim citing several blocks is judged against them together.** When such a",
        "  set fails, every citation in it is flagged, because nothing here can say which",
        "  half was at fault.",
        "* **Claim splitting is deterministic and imperfect.** Coverage is a ratio over",
        "  units this project defines; a different splitter would move the denominator.",
    ]
    return "\n".join(lines) + "\n"
