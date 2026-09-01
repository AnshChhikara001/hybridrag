"""Embedding cache tests.

A cache that returns a wrong or stale vector is worse than no cache: retrieval degrades
with no error anywhere. These tests pin correctness first -- identical results to the
uncached path, no collisions across models or between queries and documents -- and only
then that it actually avoids work.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from hybridrag.embedding import Embedder
from hybridrag.embedding_cache import CachedEmbedder


class CountingEmbedder:
    """Records how many texts it was actually asked to embed."""

    def __init__(self, model_name: str = "fake-model", dimension: int = 4) -> None:
        self.model_name = model_name
        self._dimension = dimension
        self.documents_embedded = 0
        self.queries_embedded = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str, offset: float) -> NDArray[np.float32]:
        raw = np.array([len(text) + offset, sum(map(ord, text)) % 7, offset, 1.0], dtype=np.float32)
        return raw / np.linalg.norm(raw)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.documents_embedded += len(texts)
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        return np.vstack([self._vector(t, 0.0) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.queries_embedded += 1
        return self._vector(text, 3.0)


@pytest.fixture
def inner() -> CountingEmbedder:
    return CountingEmbedder()


@pytest.fixture
def cache(inner: CountingEmbedder, tmp_path: Path) -> CachedEmbedder:
    return CachedEmbedder(inner, tmp_path / "embeddings.sqlite")


class TestCorrectness:
    def test_satisfies_the_embedder_protocol(self, cache: CachedEmbedder) -> None:
        assert isinstance(cache, Embedder)

    def test_cached_results_match_the_uncached_ones(
        self, cache: CachedEmbedder, inner: CountingEmbedder
    ) -> None:
        texts = ["alpha", "beta", "gamma"]
        assert np.allclose(cache.embed_documents(texts), inner.embed_documents(texts))

    def test_a_second_call_returns_identical_vectors(self, cache: CachedEmbedder) -> None:
        first = cache.embed_documents(["alpha", "beta"])
        assert np.array_equal(first, cache.embed_documents(["alpha", "beta"]))

    def test_order_is_preserved(self, cache: CachedEmbedder) -> None:
        forward = cache.embed_documents(["alpha", "beta", "gamma"])
        assert np.array_equal(cache.embed_documents(["gamma"])[0], forward[2])

    def test_repeated_text_in_one_call_is_handled(self, cache: CachedEmbedder) -> None:
        vectors = cache.embed_documents(["alpha", "alpha", "beta"])
        assert vectors.shape == (3, 4)
        assert np.array_equal(vectors[0], vectors[1])

    def test_empty_input_returns_a_stackable_empty_matrix(self, cache: CachedEmbedder) -> None:
        assert cache.embed_documents([]).shape == (0, 4)

    def test_output_is_float32(self, cache: CachedEmbedder) -> None:
        assert cache.embed_documents(["alpha"]).dtype == np.float32


class TestNoCollisions:
    def test_a_query_and_a_document_with_the_same_text_differ(self, cache: CachedEmbedder) -> None:
        """bge prefixes queries with an instruction, so these are genuinely different."""
        assert not np.array_equal(cache.embed_query("alpha"), cache.embed_documents(["alpha"])[0])

    def test_two_models_do_not_share_cached_vectors(self, tmp_path: Path) -> None:
        """A shared key would silently serve one model's vectors to another."""
        path = tmp_path / "embeddings.sqlite"
        first = CachedEmbedder(CountingEmbedder("model-a"), path)
        second_inner = CountingEmbedder("model-b")
        second = CachedEmbedder(second_inner, path)
        first.embed_documents(["alpha"])
        second.embed_documents(["alpha"])
        assert second_inner.documents_embedded == 1, "model-b served model-a's vector"


class TestItActuallySavesWork:
    def test_repeat_texts_are_not_recomputed(
        self, cache: CachedEmbedder, inner: CountingEmbedder
    ) -> None:
        cache.embed_documents(["alpha", "beta"])
        cache.embed_documents(["alpha", "beta"])
        assert inner.documents_embedded == 2

    def test_only_the_new_texts_are_computed(
        self, cache: CachedEmbedder, inner: CountingEmbedder
    ) -> None:
        """The semantic sweep case: same sentences, one changed parameter."""
        cache.embed_documents(["alpha", "beta"])
        cache.embed_documents(["alpha", "beta", "gamma"])
        assert inner.documents_embedded == 3

    def test_duplicates_in_one_call_are_computed_once(
        self, cache: CachedEmbedder, inner: CountingEmbedder
    ) -> None:
        cache.embed_documents(["alpha"] * 5)
        assert inner.documents_embedded == 1

    def test_repeat_queries_are_not_recomputed(
        self, cache: CachedEmbedder, inner: CountingEmbedder
    ) -> None:
        """Evaluation runs the same golden questions repeatedly."""
        cache.embed_query("how do I use it")
        cache.embed_query("how do I use it")
        assert inner.queries_embedded == 1


class TestPersistence:
    def test_the_cache_survives_a_restart(self, tmp_path: Path) -> None:
        """The point of the whole thing: a 28-minute run is paid once, not per run."""
        path = tmp_path / "embeddings.sqlite"
        CachedEmbedder(CountingEmbedder(), path).embed_documents(["alpha", "beta"])

        fresh_inner = CountingEmbedder()
        CachedEmbedder(fresh_inner, path).embed_documents(["alpha", "beta"])
        assert fresh_inner.documents_embedded == 0

    def test_it_creates_missing_directories(self, tmp_path: Path) -> None:
        CachedEmbedder(CountingEmbedder(), tmp_path / "nested" / "deeper" / "e.sqlite")
        assert (tmp_path / "nested" / "deeper").is_dir()
