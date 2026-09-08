"""The FastAPI service: `POST /v1/ask`, `GET /v1/documents`, `POST /v1/ingest`.

Every route is a thin layer over library code that already exists and is already tested --
`Answerer` for generation, `HybridRetriever` for retrieval, the chunk/dense/sparse index
trio for storage. This module's job is wiring those into HTTP, once at startup, not
reimplementing any of them.

Route handlers are plain `def`, not `async def`, on purpose: everything they call
(sqlite, the embedding model, the language model) is blocking I/O, and FastAPI runs a sync
path operation in Starlette's threadpool automatically. Writing `async def` around blocking
calls would instead block the single event loop thread and serialise every request for
real, which is the opposite of what a threadpool is for.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from google.genai.errors import APIError

from hybridrag.api.resources import (
    SERVED_STRATEGY,
    VERIFIER_MAX_TOKENS,
    Resources,
    build_resources,
)
from hybridrag.api.schemas import (
    AskRequest,
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    RetrievalMode,
)
from hybridrag.embedding_openai import OpenAIEmbedder
from hybridrag.generation import (
    Answer,
    Answerer,
    CitationVerifier,
    CompletenessScorer,
    GenerationError,
)
from hybridrag.indexing import DedupScope, SparseIndex, deduplicate, sparse_path
from hybridrag.loaders import CorpusLoader, LoaderError
from hybridrag.models import Chunk

# Serialises the mutating half of /v1/ingest against itself. A single request already
# locks each store it touches individually (chunk_store.py, dense.py), but ingest is a
# sequence of several such calls plus a sparse-index file rewrite, and two concurrent
# ingests interleaving that sequence could each rebuild the sparse index from a different
# snapshot and have the second write clobber the first. Reads (/v1/ask, /v1/documents)
# are not blocked by this -- only another ingest is.
_ingest_lock = threading.Lock()

_ACCEPTED_INGEST_SUFFIXES = frozenset({".md", ".txt"})


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    resources = build_resources()
    app.state.resources = resources
    try:
        yield
    finally:
        resources.close()


app = FastAPI(
    title="HybridRAG",
    description=(
        "Hybrid dense+sparse retrieval over FastAPI's documentation, with grounded, "
        "cited answers. See docs/PROJECT_STATE.md for the measured evaluation behind "
        "every default here."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)


def get_resources(request: Request) -> Resources:
    resources: Resources = request.app.state.resources
    return resources


@app.get("/health", response_model=HealthResponse)
def health(resources: Resources = Depends(get_resources)) -> HealthResponse:
    return HealthResponse(
        strategy=SERVED_STRATEGY,
        documents_indexed=len(resources.documents),
        # Not len(resources.store): that counts chunks under every strategy the offline
        # comparison built (D33), and the API serves only SERVED_STRATEGY.
        chunks_indexed=len(resources.store.chunk_ids(SERVED_STRATEGY)),
    )


@app.get("/v1/documents", response_model=DocumentsResponse)
def list_documents(resources: Resources = Depends(get_resources)) -> DocumentsResponse:
    return DocumentsResponse(strategy=SERVED_STRATEGY, documents=resources.documents)


@app.post("/v1/ask", response_model=Answer)
def ask(payload: AskRequest, resources: Resources = Depends(get_resources)) -> Answer:
    retriever = resources.retriever
    if payload.mode is RetrievalMode.DENSE_ONLY:
        retriever = retriever.ablation("dense")

    checker = None
    if payload.verify:
        if resources.verifier_model is None:
            raise HTTPException(
                status_code=400,
                detail="verify=true needs OPENAI_API_KEY configured on the server.",
            )
        checker = resources.verifier_model

    answerer = Answerer(
        retriever,
        resources.model,
        k=payload.k,
        confidence_threshold=resources.settings.retrieval_confidence_threshold,
        verifier=CitationVerifier(checker, max_output_tokens=VERIFIER_MAX_TOKENS)
        if checker
        else None,
        completeness_scorer=CompletenessScorer(checker, max_output_tokens=VERIFIER_MAX_TOKENS)
        if checker
        else None,
    )
    try:
        return answerer.answer(payload.question)
    except (APIError, GenerationError) as error:
        # A provider outage or an exhausted-retries failure is not this service being
        # broken; 503 says so, matching ask.py's equivalent CLI-side message.
        raise HTTPException(
            status_code=503, detail=f"the model could not be reached: {error}"
        ) from error


def _rebuild_sparse_index(resources: Resources) -> None:
    """Full rebuild, not an incremental add: BM25's IDF is corpus-relative (D33), so
    inserting one document's terms changes the statistics for every existing chunk too.
    """
    chunks: Iterable[Chunk] = resources.store.iter_chunks(SERVED_STRATEGY)
    sparse = SparseIndex.build(chunks)
    sparse.save(sparse_path(resources.settings.index_dir, SERVED_STRATEGY))
    resources.retriever.indexes["sparse"] = sparse


@app.post("/v1/ingest", response_model=IngestResponse)
def ingest(file: UploadFile, resources: Resources = Depends(get_resources)) -> IngestResponse:
    if file.filename is None:
        raise HTTPException(status_code=422, detail="the upload needs a filename.")
    # Basename only: a filename is untrusted input, and joining a raw one to a directory
    # would let "../../x" write outside it.
    name = Path(file.filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in _ACCEPTED_INGEST_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"only {sorted(_ACCEPTED_INGEST_SUFFIXES)} are accepted; got '{suffix}'.",
        )

    raw_bytes = file.file.read()
    if not raw_bytes.strip():
        raise HTTPException(status_code=422, detail="the uploaded file is empty.")

    with _ingest_lock:
        resources.ingest_raw_dir.mkdir(parents=True, exist_ok=True)
        destination = resources.ingest_raw_dir / name
        destination.write_bytes(raw_bytes)

        loader = CorpusLoader(resources.ingest_raw_dir)
        try:
            document = loader.load(destination)
        except LoaderError as error:
            raise HTTPException(
                status_code=422, detail=f"could not parse '{name}': {error}"
            ) from error

        new_chunks = resources.chunker.chunk(document)
        if not new_chunks:
            raise HTTPException(status_code=422, detail=f"'{name}' produced no chunks.")

        embedder_inner = resources.embedder.inner
        cost_before = (
            embedder_inner.estimated_cost_usd if isinstance(embedder_inner, OpenAIEmbedder) else 0.0
        )

        report = deduplicate(
            new_chunks,
            resources.embedder,
            threshold=resources.settings.dedup_threshold,
            scope=DedupScope.DOCUMENT,
        )
        kept = report.kept

        chunks_removed = resources.store.delete_document(document.relative_path, SERVED_STRATEGY)
        resources.dense.delete_document(document.relative_path, SERVED_STRATEGY)

        resources.store.add(kept)
        resources.dense.add(kept)
        _rebuild_sparse_index(resources)
        resources.record_ingested_document(document)

        cost_after = (
            embedder_inner.estimated_cost_usd if isinstance(embedder_inner, OpenAIEmbedder) else 0.0
        )

    return IngestResponse(
        relative_path=document.relative_path,
        doc_id=document.doc_id,
        sections=len(document.sections),
        chunks_added=len(kept),
        chunks_removed=chunks_removed,
        duplicates_dropped=len(report.removed),
        cost_usd=round(cost_after - cost_before, 6),
    )
