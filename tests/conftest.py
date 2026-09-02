"""Shared fixtures.

`make_pdf` builds a genuinely valid PDF with correct cross-reference offsets rather than
mocking `pypdf`. Mocking the parser would test only that our code calls a library; this
tests that a real PDF byte stream round-trips through extraction into pages we can cite.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from hybridrag.models import Chunk, ChunkingStrategy

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


def _build_pdf(pages: list[str]) -> bytes:
    """Assemble a minimal multi-page PDF, computing the xref table offsets."""
    count = len(pages)
    page_ids = [3 + i for i in range(count)]
    content_ids = [3 + count + i for i in range(count)]
    font_id = 3 + 2 * count

    objects: list[tuple[int, str]] = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>"),
        (
            2,
            f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {count} >>",
        ),
        (font_id, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    for page_id, content_id in zip(page_ids, content_ids, strict=True):
        objects.append(
            (
                page_id,
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>",
            )
        )
    for content_id, text in zip(content_ids, pages, strict=True):
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        objects.append((content_id, f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"))

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number, body in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")

    xref_offset = len(out)
    highest = max(offsets)
    out += f"xref\n0 {highest + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for number in range(1, highest + 1):
        out += f"{offsets[number]:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {highest + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[[list[str]], Path]:
    def _make(pages: list[str]) -> Path:
        target = tmp_path / "sample.pdf"
        target.write_bytes(_build_pdf(pages))
        return target

    return _make


@pytest.fixture
def fixture_corpus() -> Path:
    return FIXTURE_CORPUS


class WordTokenCounter:
    """Whitespace tokenizer satisfying `TokenCounter`.

    Chunker tests care about boundary placement, not sub-word vocabulary. A deterministic
    whitespace counter makes budgets readable in the test itself ("this text is 12 tokens")
    and keeps the suite free of a model download. The real tokenizer is exercised
    separately, against the real corpus.
    """

    def count(self, text: str) -> int:
        return len(text.split())

    def offsets(self, text: str) -> list[tuple[int, int]]:
        import re

        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


@pytest.fixture
def word_tokenizer() -> WordTokenCounter:
    return WordTokenCounter()


class TopicEmbedder:
    """Deterministic stand-in for a real embedder, satisfying `Embedder`.

    Each sentence is placed on whichever topic word it contains, so distance between
    consecutive sentences is 0 within a topic and 1 across a topic change. That makes the
    semantic chunker's expected boundaries readable in the test itself, and keeps the suite
    free of a model download and of a real model's judgement about what is similar.
    """

    model_name = "test-topic-embedder"
    topics = ("alpha", "beta", "gamma", "delta")

    @property
    def dimension(self) -> int:
        return len(self.topics)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        rows = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.lower()
            for column, topic in enumerate(self.topics):
                if topic in lowered:
                    rows[row, column] = 1.0
            if not rows[row].any():
                rows[row, 0] = 1.0  # text mentioning no topic sits with the first
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return (rows / np.where(norms == 0.0, 1.0, norms)).astype(np.float32, copy=False)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        row: NDArray[np.float32] = self.embed_documents([text])[0]
        return row


@pytest.fixture
def topic_embedder() -> TopicEmbedder:
    return TopicEmbedder()


def _make_chunk(
    index: int,
    text: str = "Query parameters are declared as function arguments.",
    *,
    doc_id: str = "doc-1",
    strategy: ChunkingStrategy = ChunkingStrategy.STRUCTURE,
    relative_path: str = "tutorial/query-params.md",
    heading_path: tuple[str, ...] = ("Tutorial", "Query Parameters"),
) -> Chunk:
    """A valid `Chunk` with as little ceremony as the model's validators allow."""
    return Chunk(
        chunk_id=Chunk.make_id(doc_id, strategy, index),
        doc_id=doc_id,
        relative_path=relative_path,
        section_ids=["section-a", "section-b"],
        heading_path=heading_path,
        text=text,
        chunk_index=index,
        strategy=strategy,
        token_count=max(1, len(text.split())),
        char_count=len(text),
        start_char=index * 1000,
        end_char=index * 1000 + len(text),
    )


@pytest.fixture
def make_chunk() -> Callable[..., Chunk]:
    """Chunk factory shared by the store and retrieval suites.

    Lives here rather than in one test module because `tests` is not an importable
    package: pytest reaches conftest fixtures without it, a cross-module import does not.
    """
    return _make_chunk
