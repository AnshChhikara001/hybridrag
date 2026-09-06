"""Draft golden-set candidates for hand-verification (D11).

The model proposes; a human disposes. Every candidate lands with `verified: false`, and
`GoldenSet.verified()` is what metrics read -- so nothing reaches a reported number until a
person has read it. This script's job is to make that reading task as short as possible by
throwing away everything it can check mechanically first:

* a quote that does not occur **exactly once** in its document is rejected outright,
  because a paraphrase or an ambiguous match cannot be ground truth (see `golden.py`);
* a candidate whose span shape contradicts its category is rejected by the schema;
* a `no_answer` candidate whose retrieval confidence is high is flagged, because the
  corpus probably does answer it after all.

Multi-hop pairs are chosen by **retrieval overlap**: for a document, the dense index says
which other document its own chunks retrieve. Those are documents the retriever already
confuses, which is exactly where a two-document question is both natural to ask and hard to
answer -- rather than an arbitrary pairing that no retriever would ever have to separate.

Generation runs on Gemini's free tier through the response cache, so re-running after a
prompt tweak only pays for what actually changed.

    uv run python scripts/generate_golden.py data/raw/fastapi/docs/en/docs
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer

from hybridrag.chunk_store import ChunkStore
from hybridrag.config import get_settings
from hybridrag.embedding import Embedder, FastEmbedEmbedder
from hybridrag.embedding_cache import CachedEmbedder
from hybridrag.embedding_openai import OpenAIEmbedder
from hybridrag.evaluation import AnswerSpan, GoldenQuestion, GoldenSet, QuestionCategory
from hybridrag.evaluation.golden import SpanNotFoundError
from hybridrag.generation import CachedLanguageModel, GeminiModel, LanguageModel, OpenAIModel
from hybridrag.indexing import DenseIndex
from hybridrag.loaders import CorpusLoader
from hybridrag.models import Document

app = typer.Typer(add_completion=False)

# All of these stay in the corpus -- they are realistic distractor content (decision Q1) --
# but none is a question source. They are project narrative and community process rather
# than technical reference, and questions drawn from them ("which libraries did the author
# contribute to?") measure nothing a documentation assistant is for. Observed directly:
# the first three lookup candidates all came from history-design-future.md.
EXCLUDED_SOURCES = (
    "release-notes.md",
    "history-design-future.md",
    "alternatives.md",
    "benchmarks.md",
    "help-fastapi.md",
    "contributing.md",
    "fastapi-people.md",
    "management.md",
    "management-tasks.md",
    "external-links.md",
    "project-generation.md",
    "about/",
    "resources/",
)

# Below this a document is a stub -- a redirect or a two-line note -- and produces
# questions with no substance behind them.
MIN_DOCUMENT_CHARS = 1500

# Sentinel-delimited blocks, not JSON. Two failures forced this, both caused by the corpus
# itself rather than by the model being careless:
#
#  * quotes are verbatim documentation, and this documentation contains `"` characters
#    ('a "context manager"'). Emitted inside a JSON string they must be escaped, and models
#    do not do that reliably -- 52 of 130 replies were unparseable.
#  * the quotes also contain ``` code fences, so any regex that strips a markdown fence
#    around a JSON envelope strips a Python example out of the middle of a quote instead.
#
# Sentinels sidestep both: nothing needs escaping, and no documentation line will ever be
# `<<<ENDSPAN>>>`.
ITEM = "<<<ITEM>>>"
END_ITEM = "<<<ENDITEM>>>"
SPAN = "<<<SPAN>>>"
END_SPAN = "<<<ENDSPAN>>>"

REPLY_FORMAT = f"""\
Reply using EXACTLY this block format, and nothing else -- no JSON, no markdown fence,
no commentary:

{ITEM}
QUESTION: <one line>
ANSWER: <one line>
{SPAN} <relative_path>
<the verbatim quote, which may span several lines and contain any characters>
{END_SPAN}
{END_ITEM}

Repeat the whole block once per item. Use one {SPAN}...{END_SPAN} pair per span."""


RULES = """\
Rules for every item you produce:
- The quote must be copied CHARACTER FOR CHARACTER from the document shown. Do not
  paraphrase, do not fix typos, do not change whitespace, punctuation or backticks.
- The quote must be long enough to appear only once in that document. Use TWO OR MORE
  complete sentences, at least 20 words. Short single lines recur across these docs and
  will be discarded.
- The question must read like something a developer would actually type, not a quiz
  question about the document. Never mention "the document" or "the context"."""


def _language_model(provider: str) -> LanguageModel:
    settings = get_settings()
    if provider == "openai":
        if settings.openai_api_key is None:
            raise typer.BadParameter("OPENAI_API_KEY is not set.")
        # 16k, not 4k. Reasoning tokens are billed against the same cap on a reasoning
        # model, and at 4096 twenty-four replies were cut off mid-JSON -- paid for in full
        # and unusable. The same trap as Gemini's thinking budget, in a second SDK.
        return OpenAIModel(
            settings.openai_api_key, "gpt-5-mini-2025-08-07", max_output_tokens=16384
        )
    if settings.gemini_api_key is None:
        raise typer.BadParameter("GEMINI_API_KEY is not set.")
    return GeminiModel(settings.gemini_api_key, settings.generation_model, max_output_tokens=4096)


def _embedder() -> Embedder:
    settings = get_settings()
    if settings.openai_api_key is None:
        return FastEmbedEmbedder(settings.embedding_model)
    return OpenAIEmbedder(settings.openai_api_key)


def _parse_items(text: str, failures: Counter[str] | None = None) -> list[dict[str, Any]]:
    """Split a reply into items. Nothing here can be broken by quote content.

    Malformed blocks are counted rather than swallowed, so a run reports whether the model
    ignored the format or simply produced less than asked.
    """
    items: list[dict[str, Any]] = []
    blocks = [b for b in text.split(ITEM)[1:]]
    if not blocks and failures is not None and text.strip():
        failures["reply ignored the block format"] += 1

    for block in blocks:
        body = block.split(END_ITEM)[0]
        head, *span_parts = body.split(SPAN)

        question = _field(head, "QUESTION:")
        answer = _field(head, "ANSWER:")
        if not question:
            if failures is not None:
                failures["block had no QUESTION line"] += 1
            continue

        spans: list[dict[str, str]] = []
        for part in span_parts:
            first_line, _, rest = part.partition("\n")
            quote = rest.split(END_SPAN)[0]
            path = first_line.strip()
            if path and quote.strip():
                # Only the trailing newline before the sentinel is stripped: internal
                # blank lines are part of the verbatim quote and removing them would
                # break the exact-match check that makes a span trustworthy.
                spans.append({"relative_path": path, "quote": quote.strip("\n")})
        items.append({"question": question, "answer": answer, "spans": spans})
    return items


def _field(text: str, label: str) -> str:
    """The single-line value following `label`, or empty when absent."""
    for line in text.splitlines():
        if line.strip().startswith(label):
            return line.strip()[len(label) :].strip()
    return ""


def _excerpt(document: Document, limit: int = 6000) -> str:
    """Enough of a document to ask about, capped so a long page cannot dominate a prompt."""
    return document.text[:limit]


def _pair_by_retrieval_overlap(
    documents: list[Document], dense: DenseIndex, store: ChunkStore, samples: int = 3
) -> dict[str, str]:
    """For each document, the other document its own chunks most often retrieve.

    This is the retriever's own opinion of which pages are confusable, which makes it the
    honest place to look for a question that genuinely needs both.
    """
    pairs: dict[str, str] = {}
    for document in documents:
        chunks = [c for c in store.iter_chunks() if c.doc_id == document.doc_id][:samples]
        neighbours: Counter[str] = Counter()
        for chunk in chunks:
            for chunk_id, _score in dense.search(chunk.text[:800], k=8):
                other = store.get(chunk_id)
                if other is not None and other.relative_path != document.relative_path:
                    neighbours[other.relative_path] += 1
        if neighbours:
            pairs[document.relative_path] = neighbours.most_common(1)[0][0]
    return pairs


def _accept(
    item: dict[str, Any],
    category: QuestionCategory,
    documents: dict[str, Document],
    index: int,
    rejected: Counter[str],
) -> GoldenQuestion | None:
    """Validate one candidate, or count why it was thrown away."""
    # A no_answer question has no ground-truth span by definition, and the model supplies
    # them anyway -- it is being asked for an unanswerable question while looking at a
    # document, so it reaches for a passage. The category already settles this, so the
    # spans are dropped rather than treated as a rejection. Whether the question is
    # genuinely unanswerable is the human's call, and the confidence flag below triages it.
    raw_spans = [] if category is QuestionCategory.NO_ANSWER else item.get("spans", [])
    spans = [
        AnswerSpan(relative_path=str(s["relative_path"]), quote=str(s["quote"]))
        for s in raw_spans
        if isinstance(s, dict) and s.get("relative_path") and s.get("quote")
    ]
    try:
        question = GoldenQuestion(
            question_id=f"{category.value}-{index:03d}",
            category=category,
            question=str(item.get("question", "")).strip(),
            answer=str(item.get("answer", "")).strip(),
            spans=spans,
            verified=False,
        )
    except ValueError as error:
        # Report the validator's own sentence, not Pydantic's header line, so a run says
        # which rule was broken and the prompt can be fixed rather than guessed at.
        lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        detail = next(
            (line.split("Value error, ")[-1] for line in lines if "Value error" in line),
            lines[-1] if lines else "unknown",
        )
        rejected[detail[:90]] += 1
        return None

    try:
        question.locate(documents)
    except SpanNotFoundError as error:
        reason = "quote not verbatim" if "not found" in str(error) else "quote not unique"
        rejected[reason] += 1
        return None
    return question


def _prompt_for(category: QuestionCategory, docs: list[Document], count: int) -> str:
    shown = (
        ""
        if category is QuestionCategory.NO_ANSWER
        else "\n\n".join(f"### DOCUMENT: {d.relative_path}\n{_excerpt(d)}" for d in docs)
    )
    if category is QuestionCategory.MULTI_HOP:
        task = (
            f"Write {count} question(s) that CANNOT be answered from either document alone "
            "and genuinely require combining both. Give exactly one span from each document."
        )
    elif category is QuestionCategory.AMBIGUOUS:
        task = (
            f"Write {count} under-specified question(s) a developer might ask, where at "
            "least two DIFFERENT passages of this document are each a reasonable answer -- "
            "for example a question that does not say which of two mechanisms it means.\n"
            f"MANDATORY: give at least TWO separate {SPAN} blocks per question, one for "
            "each passage that could reasonably answer it. An item with only one span is "
            "discarded, and every item in the previous run was discarded for this reason."
        )
    elif category is QuestionCategory.NO_ANSWER:
        task = (
            f"Write {count} question(s) about FastAPI that a developer would plausibly ask "
            "but that this documentation does NOT answer -- operational, commercial or "
            "version-specific facts it never states (e.g. support SLAs, benchmarks against "
            "a named competitor, internal roadmap). Return an empty `spans` list."
        )
    else:
        task = f"Write {count} straightforward factual question(s) answered by this document."

    if category is QuestionCategory.NO_ANSWER:
        # Neither the quote rules nor the span block apply: there is no document and no
        # ground truth to point at. Sending them anyway produced one unusable reply,
        # because the format demanded a span the task forbade.
        no_span_format = (
            f"Reply using EXACTLY this block format and nothing else:\n\n"
            f"{ITEM}\nQUESTION: <one line>\nANSWER: <one line explaining why the "
            f"documentation does not cover this>\n{END_ITEM}\n\n"
            f"Repeat the block once per question. Do not include any {SPAN} block."
        )
        return f"TASK\n{task}\n\n{no_span_format}"
    return f"{shown}\n\nTASK\n{task}\n\n{RULES}\n\n{REPLY_FORMAT}"


@app.command()
def generate(
    corpus: Annotated[Path, typer.Argument(help="Corpus directory the index was built from.")],
    out: Annotated[Path, typer.Option(help="Where to write candidates.")] = Path(
        "evals/golden_candidates.yaml"
    ),
    lookups: Annotated[int, typer.Option()] = 30,
    multi_hop: Annotated[int, typer.Option()] = 12,
    ambiguous: Annotated[int, typer.Option()] = 9,
    no_answer: Annotated[int, typer.Option()] = 10,
    provider: Annotated[str, typer.Option(help="gemini (free) or openai.")] = "gemini",
    seed: Annotated[int, typer.Option(help="Document sampling seed, for reproducibility.")] = 7,
) -> None:
    """Draft candidates for a human to verify. Nothing here is trusted until it is read."""
    settings = get_settings()
    include_root = next((p for p in corpus.resolve().parents if (p / "docs_src").is_dir()), None)
    loader = CorpusLoader(corpus, include_root=include_root)
    documents = {d.relative_path: d for d in loader.iter_documents()}
    typer.echo(f"corpus: {len(documents)} documents")

    eligible = sorted(
        (
            d
            for d in documents.values()
            if len(d.text) >= MIN_DOCUMENT_CHARS
            and not d.relative_path.startswith(EXCLUDED_SOURCES)
        ),
        key=lambda d: d.relative_path,
    )
    typer.echo(f"eligible as question sources: {len(eligible)}")

    rng = random.Random(seed)
    model = CachedLanguageModel(
        _language_model(provider), settings.cache_dir / "completions.sqlite"
    )

    store = ChunkStore(settings.index_dir / "chunks.sqlite")
    embedder = CachedEmbedder(_embedder(), settings.cache_dir / "embeddings.sqlite")
    dense = DenseIndex.embedded(embedder, settings.index_dir / "chroma")

    accepted: list[GoldenQuestion] = []
    rejected: Counter[str] = Counter()

    def run(category: QuestionCategory, groups: list[list[Document]], per_group: int) -> None:
        for group in groups:
            if len(accepted) >= 1000:
                break
            reply = model.generate(_prompt_for(category, group, per_group))
            items = _parse_items(reply.text, rejected)
            if not items:
                rejected[f"{category.value}: call produced no usable items"] += 1
            for item in items:
                index = sum(1 for q in accepted if q.category is category) + 1
                question = _accept(item, category, documents, index, rejected)
                if question is not None:
                    accepted.append(question)
        typer.echo(
            f"  {category.value:10} accepted {sum(1 for q in accepted if q.category is category)}"
        )

    typer.echo("generating (cached; a re-run after a prompt change pays only for the change)")
    per_call = 3
    lookup_docs = rng.sample(eligible, min(len(eligible), -(-lookups // per_call)))
    run(QuestionCategory.LOOKUP, [[d] for d in lookup_docs], per_call)

    pair_sources = rng.sample(eligible, min(len(eligible), multi_hop))
    pairs = _pair_by_retrieval_overlap(pair_sources, dense, store)
    typer.echo(f"  paired {len(pairs)} documents by retrieval overlap")
    run(
        QuestionCategory.MULTI_HOP,
        [[documents[a], documents[b]] for a, b in pairs.items() if b in documents],
        1,
    )

    ambiguous_docs = rng.sample(eligible, min(len(eligible), -(-ambiguous // per_call)))
    run(QuestionCategory.AMBIGUOUS, [[d] for d in ambiguous_docs], per_call)

    # One call, no document: the prompt no longer depends on one.
    run(QuestionCategory.NO_ANSWER, [[eligible[0]]], no_answer)

    # A no_answer question the retriever answers confidently is probably answerable, so it
    # is flagged for the human rather than silently kept.
    suspicious = 0
    for question in accepted:
        if question.category is not QuestionCategory.NO_ANSWER:
            continue
        # Not the refusal threshold (0.30): a no_answer question is required to be
        # in-domain, so it scores near it by construction and every candidate would be
        # flagged. This is set in the range measured for genuinely answerable questions.
        hits = dense.search(question.question, k=1)
        if hits and hits[0][1] >= 0.55:
            question.notes = (
                f"CHECK: retrieval confidence {hits[0][1]:.2f} sits in the range measured "
                "for answerable questions, so the corpus may well answer this one."
            )
            suspicious += 1

    GoldenSet(corpus_ref="0.115.6", questions=accepted).save(out)

    typer.echo(f"\nwrote {len(accepted)} candidates to {out}")
    if suspicious:
        typer.echo(f"  {suspicious} no_answer candidate(s) flagged as possibly answerable")
    if rejected:
        typer.echo("rejected before you ever see them:")
        for reason, count in rejected.most_common():
            typer.echo(f"  {count:3}  {reason}")
    typer.echo(
        f"\ncache: {model.hits} hit / {model.misses} miss  "
        f"spend ${getattr(model.inner, 'estimated_cost_usd', 0.0):.4f}"
    )
    typer.echo("\nNext: read every candidate, fix or delete, then set verified: true.")

    model.close()
    embedder.close()
    store.close()


if __name__ == "__main__":
    sys.exit(app())
