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

from pydantic import BaseModel, Field

from hybridrag.evaluation.golden import GoldenSet, QuestionCategory
from hybridrag.evaluation.harness import METRICS, ArmResult
from hybridrag.evaluation.judge import LabelSheet, Verdict
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
