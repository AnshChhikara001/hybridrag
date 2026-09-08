"""API endpoint tests.

Run entirely offline: no real corpus, no index files on disk, no API key, no network. A
`Resources` object is built directly from an in-memory chunk store, an in-memory Chroma
index, a small `SparseIndex`, a stub language model and the test suite's own
`WordTokenCounter`/`TopicEmbedder` -- exactly what `build_resources()` would produce from
real files, but fast and deterministic. `get_resources` is overridden via FastAPI's
`dependency_overrides` rather than exercising the real `lifespan`, so no test touches the
filesystem outside `tmp_path` or needs `data/index` to exist.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hybridrag.api.app import app, get_resources
from hybridrag.api.resources import Resources
from hybridrag.chunk_store import ChunkStore
from hybridrag.chunking import FixedChunker
from hybridrag.config import Settings
from hybridrag.embedding import Embedder
from hybridrag.embedding_cache import CachedEmbedder
from hybridrag.generation import GenerationError
from hybridrag.generation.base import Completion
from hybridrag.generation.cache import CachedLanguageModel
from hybridrag.generation.models import NO_ANSWER_TEXT
from hybridrag.indexing import DenseIndex, SparseIndex
from hybridrag.models import Chunk, ChunkingStrategy, Document, Section, SourceFormat
from hybridrag.retrieval import HybridRetriever
from hybridrag.tokenization import TokenCounter

MakeChunk = Callable[..., Chunk]


class StubModel:
    """A fixed reply, satisfying `LanguageModel` and remembering its calls."""

    def __init__(self, reply: str = "Alpha means the first topic [1].") -> None:
        self.model_name = "stub-model"
        self.reply = reply
        self.prompts: list[str] = []

    @property
    def fingerprint(self) -> str:
        return f"stub|{self.model_name}"

    def generate(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> Completion:
        self.prompts.append(prompt)
        return Completion(
            text=self.reply,
            model=self.model_name,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.0,
            latency_s=0.1,
        )


class RaisingModel:
    """Satisfies `LanguageModel`; every call fails, simulating a dead provider."""

    model_name = "raising-model"
    fingerprint = "raising|raising-model"

    def generate(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> Completion:
        raise GenerationError("simulated provider outage")


class BrokenIndex:
    """A `SearchIndex` that fails if searched -- proves an ablation excludes it."""

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        raise AssertionError("this index must not be searched under dense_only mode")


def _document(relative_path: str, title: str) -> Document:
    text = "Alpha content about the first topic."
    return Document(
        doc_id=Document.make_id(relative_path),
        relative_path=relative_path,
        source_format=SourceFormat.MARKDOWN,
        title=title,
        text=text,
        sections=[
            Section(
                section_id=Section.make_id(relative_path, ("Intro",)),
                relative_path=relative_path,
                heading_path=("Intro",),
                level=1,
                text=text,
                start_char=0,
                end_char=len(text),
            )
        ],
        content_hash="deadbeef",
    )


@pytest.fixture
def language_model() -> StubModel:
    return StubModel()


@pytest.fixture
def resources(
    tmp_path: Path,
    topic_embedder: Embedder,
    word_tokenizer: TokenCounter,
    make_chunk: MakeChunk,
    language_model: StubModel,
) -> Iterator[Resources]:
    settings = Settings(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        index_dir=tmp_path / "index",
        cache_dir=tmp_path / "cache",
    )

    embedder = CachedEmbedder(topic_embedder, tmp_path / "embeddings.sqlite")
    store = ChunkStore.in_memory()
    dense = DenseIndex.in_memory(embedder)
    strategy = ChunkingStrategy.FIXED

    chunk = make_chunk(
        0,
        "Alpha content about the first topic.",
        strategy=strategy,
        relative_path="alpha.md",
        heading_path=("Intro",),
    )
    store.add([chunk])
    dense.add([chunk])
    sparse = SparseIndex.build([chunk])
    retriever = HybridRetriever({"dense": dense, "sparse": sparse}, store)

    model = CachedLanguageModel(language_model, tmp_path / "completions.sqlite")

    built = Resources(
        settings=settings,
        embedder=embedder,
        store=store,
        dense=dense,
        retriever=retriever,
        model=model,
        verifier_model=None,
        chunker=FixedChunker(word_tokenizer, max_tokens=64, overlap_tokens=0),
        ingest_raw_dir=settings.raw_dir / "ingested",
        ingested_manifest_path=settings.processed_dir / "ingested.jsonl",
    )
    built.record_ingested_document(_document("alpha.md", "Alpha Doc"))
    yield built
    built.close()


@pytest.fixture(autouse=True)
def _override_resources(resources: Resources) -> Iterator[None]:
    app.dependency_overrides[get_resources] = lambda: resources
    yield
    app.dependency_overrides.pop(get_resources, None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealth:
    def test_reports_strategy_and_counts(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["strategy"] == "fixed"
        assert body["documents_indexed"] == 1
        assert body["chunks_indexed"] == 1

    def test_counts_only_the_served_strategy(
        self, client: TestClient, resources: Resources, make_chunk: MakeChunk
    ) -> None:
        """A chunk under a different strategy must not inflate the served count."""
        resources.store.add(
            [make_chunk(0, strategy=ChunkingStrategy.STRUCTURE, relative_path="other.md")]
        )
        assert client.get("/health").json()["chunks_indexed"] == 1


class TestDocuments:
    def test_lists_the_seeded_document(self, client: TestClient) -> None:
        response = client.get("/v1/documents")
        assert response.status_code == 200
        body = response.json()
        assert body["strategy"] == "fixed"
        assert body["documents"] == [
            {
                "relative_path": "alpha.md",
                "title": "Alpha Doc",
                "source_format": "markdown",
                "content_hash": "deadbeef",
                "sections": 1,
                "chunks": 1,
            }
        ]


class TestAsk:
    def test_returns_a_grounded_answer_with_citations(
        self, client: TestClient, language_model: StubModel
    ) -> None:
        response = client.post("/v1/ask", json={"question": "what is alpha"})
        assert response.status_code == 200
        body = response.json()
        assert body["answered"] is True
        assert body["text"] == language_model.reply
        assert [c["relative_path"] for c in body["citations"]] == ["alpha.md"]

    def test_rejects_an_empty_question(self, client: TestClient) -> None:
        response = client.post("/v1/ask", json={"question": ""})
        assert response.status_code == 422

    def test_low_confidence_refuses_before_generating(
        self, client: TestClient, language_model: StubModel
    ) -> None:
        response = client.post("/v1/ask", json={"question": "gamma delta unrelated topic"})
        body = response.json()
        assert body["answered"] is False
        assert NO_ANSWER_TEXT in body["text"]
        assert language_model.prompts == []

    def test_dense_only_mode_never_touches_the_sparse_index(
        self, client: TestClient, resources: Resources
    ) -> None:
        resources.retriever.indexes["sparse"] = BrokenIndex()
        response = client.post("/v1/ask", json={"question": "what is alpha", "mode": "dense_only"})
        assert response.status_code == 200
        assert response.json()["answered"] is True

    def test_hybrid_mode_does_touch_the_sparse_index(
        self, client: TestClient, resources: Resources
    ) -> None:
        """Sanity check for the ablation test above: hybrid mode must hit the broken index."""
        resources.retriever.indexes["sparse"] = BrokenIndex()
        with pytest.raises(AssertionError, match="must not be searched"):
            client.post("/v1/ask", json={"question": "what is alpha", "mode": "hybrid"})

    def test_verify_without_a_configured_verifier_is_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/ask", json={"question": "what is alpha", "verify": True})
        assert response.status_code == 400

    def test_a_dead_provider_maps_to_503(self, client: TestClient, resources: Resources) -> None:
        resources.model = CachedLanguageModel(
            RaisingModel(), resources.settings.cache_dir / "completions2.sqlite", read_only=True
        )
        response = client.post("/v1/ask", json={"question": "what is alpha"})
        assert response.status_code == 503


class TestIngest:
    def test_rejects_an_unsupported_extension(self, client: TestClient) -> None:
        response = client.post(
            "/v1/ingest", files={"file": ("doc.pdf", b"content", "application/pdf")}
        )
        assert response.status_code == 415

    def test_rejects_an_empty_file(self, client: TestClient) -> None:
        response = client.post("/v1/ingest", files={"file": ("doc.md", b"   ", "text/markdown")})
        assert response.status_code == 422

    def test_indexes_a_new_document_and_makes_it_answerable(self, client: TestClient) -> None:
        content = b"# Beta Doc\n\nBeta content about the second topic.\n"
        response = client.post("/v1/ingest", files={"file": ("beta.md", content, "text/markdown")})
        assert response.status_code == 200
        body = response.json()
        assert body["relative_path"] == "beta.md"
        assert body["chunks_added"] == 1
        assert body["chunks_removed"] == 0

        listed = {doc["relative_path"] for doc in client.get("/v1/documents").json()["documents"]}
        assert "beta.md" in listed

        ask_response = client.post("/v1/ask", json={"question": "beta"})
        cited = {c["relative_path"] for c in ask_response.json()["citations"]}
        assert "beta.md" in cited

    def test_reingesting_replaces_the_previous_version(self, client: TestClient) -> None:
        first = b"# Gamma\n\nGamma content, version one.\n"
        second = b"# Gamma\n\nGamma content, version two, now longer than before.\n"
        client.post("/v1/ingest", files={"file": ("gamma.md", first, "text/markdown")})
        response = client.post("/v1/ingest", files={"file": ("gamma.md", second, "text/markdown")})
        assert response.status_code == 200
        assert response.json()["chunks_removed"] == 1
