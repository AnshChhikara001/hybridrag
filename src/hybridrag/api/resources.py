"""Everything the API needs, built once and held for the life of the process.

`scripts/ask.py` builds an equivalent set of objects on every single invocation, which is
correct for a CLI (one process, one question) and wrong for a service: reopening the chunk
store, the dense index and the embedding cache on every request would pay their setup cost
per question instead of once. `Resources` is that setup, done once in the app's `lifespan`
and shared across requests via `app.state`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hybridrag.api.schemas import DocumentSummary
from hybridrag.chunk_store import ChunkStore
from hybridrag.chunking import Chunker, FixedChunker
from hybridrag.config import Settings, get_settings
from hybridrag.embedding import Embedder, FastEmbedEmbedder
from hybridrag.embedding_cache import CachedEmbedder
from hybridrag.embedding_openai import OpenAIEmbedder
from hybridrag.generation import CachedLanguageModel, GeminiModel, LanguageModel, OpenAIModel
from hybridrag.indexing import (
    DenseIndex,
    SparseIndex,
    chroma_path,
    collection_name,
    sparse_path,
    store_path,
)
from hybridrag.loaders import DocumentStore
from hybridrag.models import ChunkingStrategy, Document
from hybridrag.retrieval import HybridRetriever
from hybridrag.tokenization import HuggingFaceTokenCounter

# The Tier-1 winner (D33): Recall@5 0.897 against structure's 0.759 and semantic's 0.828.
# The API serves one strategy rather than exposing chunking as a request parameter -- the
# three-way comparison is Phase 4's offline artefact, and the dashboard's brief-mandated
# toggle is hybrid-vs-dense-only (a retrieval mode), not a chunking strategy switch.
SERVED_STRATEGY = ChunkingStrategy.FIXED

VERIFIER_MODEL = "gpt-5-mini-2025-08-07"
VERIFIER_MAX_TOKENS = 1024

# Where a document POSTed to /v1/ingest is written before parsing. A sibling of the pinned
# FastAPI checkout (data/raw/fastapi, D19) rather than inside it: that directory is a
# pinned, re-fetchable external corpus, and mixing live uploads into it would mean a
# `fetch_corpus.sh` re-run could silently discard or shadow ingested content.
INGESTED_SUBDIR = "ingested"
INGESTED_MANIFEST = "ingested.jsonl"


def _embedder(settings: Settings) -> Embedder:
    """Whatever built the index must also embed the query, or the vectors do not compare."""
    if settings.openai_api_key is None:
        return FastEmbedEmbedder(settings.embedding_model)
    return OpenAIEmbedder(settings.openai_api_key)


def _language_model(settings: Settings) -> LanguageModel:
    if settings.generation_provider == "openai":
        if settings.openai_api_key is None:
            raise RuntimeError("HYBRIDRAG_GENERATION_PROVIDER=openai needs OPENAI_API_KEY.")
        return OpenAIModel(
            settings.openai_api_key,
            settings.generation_model,
            max_output_tokens=settings.answer_max_tokens,
        )
    if settings.gemini_api_key is None:
        raise RuntimeError("HYBRIDRAG_GENERATION_PROVIDER=gemini needs GEMINI_API_KEY.")
    return GeminiModel(
        settings.gemini_api_key,
        settings.generation_model,
        max_output_tokens=settings.answer_max_tokens,
    )


def build_chunker(settings: Settings) -> Chunker:
    """The served strategy's chunker, for ingesting one new document (D9's fixed strategy)."""
    tokenizer = HuggingFaceTokenCounter(settings.embedding_model)
    return FixedChunker(
        tokenizer,
        max_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )


def _document_summary(document: Document, chunk_counts: dict[str, int]) -> DocumentSummary:
    return DocumentSummary(
        relative_path=document.relative_path,
        title=document.title,
        source_format=document.source_format,
        content_hash=document.content_hash,
        sections=len(document.sections),
        chunks=chunk_counts.get(document.relative_path, 0),
    )


def _load_ingested_documents(path: Path) -> list[Document]:
    if not path.is_file():
        return []
    documents: dict[str, Document] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                document = Document.model_validate_json(line)
                # Last write for a relative_path wins: re-ingesting the same document
                # appends a new line rather than rewriting the file in place.
                documents[document.relative_path] = document
    return list(documents.values())


@dataclass
class Resources:
    """Constructed once in the app's lifespan; every request reads from this."""

    settings: Settings
    embedder: CachedEmbedder
    store: ChunkStore
    dense: DenseIndex
    retriever: HybridRetriever
    model: CachedLanguageModel
    verifier_model: CachedLanguageModel | None
    chunker: Chunker
    ingest_raw_dir: Path
    ingested_manifest_path: Path
    documents: list[DocumentSummary] = field(default_factory=list)

    def refresh_documents(self) -> None:
        """Recompute the `/v1/documents` listing from the processed store and the chunk store.

        Called once at startup and again after each successful ingest, rather than on every
        request: the corpus is small enough (~140 documents) that a full scan costs
        milliseconds, but there is no reason to pay it on a hot path that does not change
        between ingests.
        """
        chunk_counts: dict[str, int] = {}
        for chunk in self.store.iter_chunks(SERVED_STRATEGY):
            chunk_counts[chunk.relative_path] = chunk_counts.get(chunk.relative_path, 0) + 1

        documents: dict[str, Document] = {}
        doc_store = DocumentStore(self.settings.processed_dir)
        if doc_store.exists():
            for document in doc_store.load():
                documents[document.relative_path] = document
        for document in _load_ingested_documents(self.ingested_manifest_path):
            documents[document.relative_path] = document

        self.documents = sorted(
            (_document_summary(document, chunk_counts) for document in documents.values()),
            key=lambda summary: summary.relative_path,
        )

    def record_ingested_document(self, document: Document) -> None:
        """Append one ingested document to its manifest and refresh the cached listing."""
        self.ingested_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ingested_manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(document.model_dump_json() + "\n")
        self.refresh_documents()

    def close(self) -> None:
        self.model.close()
        if self.verifier_model is not None:
            self.verifier_model.close()
        self.embedder.close()
        self.store.close()


def build_resources(settings: Settings | None = None) -> Resources:
    settings = settings or get_settings()
    root = settings.index_dir
    bm25_path = sparse_path(root, SERVED_STRATEGY)
    if not bm25_path.is_file():
        raise RuntimeError(
            f"no {SERVED_STRATEGY.value} index at {root}. Build it first:\n"
            f"  uv run python scripts/build_index.py <corpus> --strategy {SERVED_STRATEGY.value}"
        )

    embedder = CachedEmbedder(_embedder(settings), settings.cache_dir / "embeddings.sqlite")
    store = ChunkStore(store_path(root))
    dense = DenseIndex.embedded(
        embedder, chroma_path(root), collection_name=collection_name(SERVED_STRATEGY)
    )
    sparse = SparseIndex.load(bm25_path)
    retriever = HybridRetriever({"dense": dense, "sparse": sparse}, store)

    model = CachedLanguageModel(
        _language_model(settings), settings.cache_dir / "completions.sqlite"
    )
    verifier_model = (
        CachedLanguageModel(
            OpenAIModel(
                settings.openai_api_key, VERIFIER_MODEL, max_output_tokens=VERIFIER_MAX_TOKENS
            ),
            settings.cache_dir / "completions.sqlite",
        )
        if settings.openai_api_key is not None
        else None
    )

    resources = Resources(
        settings=settings,
        embedder=embedder,
        store=store,
        dense=dense,
        retriever=retriever,
        model=model,
        verifier_model=verifier_model,
        chunker=build_chunker(settings),
        ingest_raw_dir=settings.raw_dir / INGESTED_SUBDIR,
        ingested_manifest_path=settings.processed_dir / INGESTED_MANIFEST,
    )
    resources.refresh_documents()
    return resources
