"""Embedder tests.

Two things are worth testing here and one is not. Worth testing: the contracts every caller
depends on -- unit-length vectors, correctly shaped output, and the query/document
asymmetry that a `bge` model needs to retrieve well. Not worth testing: that ONNX does
matrix multiplication correctly.

The instruction-prefix test substitutes a recording stand-in for the loaded model rather
than mocking `FastEmbedEmbedder` itself, so the code under test is the real method.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pytest

from hybridrag.embedding import BGE_QUERY_INSTRUCTION, Embedder, FastEmbedEmbedder


class RecordingModel:
    """Stands in for a loaded fastembed model, capturing what it was asked to embed."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.seen: list[str] = []

    def embed(self, documents: Sequence[str], **_: Any) -> Iterable[Any]:
        self.seen.extend(documents)
        # Deliberately un-normalised, so the caller's normalisation is what is measured.
        return [np.full(self.dimension, float(index + 2)) for index in range(len(documents))]


def embedder_with(model: RecordingModel) -> FastEmbedEmbedder:
    embedder = FastEmbedEmbedder()
    embedder._model = model  # type: ignore[assignment]  # duck-typed stand-in
    embedder._dimension = model.dimension
    return embedder


class TestContracts:
    def test_topic_embedder_satisfies_the_protocol(self, topic_embedder: Embedder) -> None:
        assert isinstance(topic_embedder, Embedder)

    def test_fastembed_embedder_satisfies_the_protocol(self) -> None:
        assert isinstance(FastEmbedEmbedder(), Embedder)

    def test_vectors_are_unit_length(self) -> None:
        """Cosine similarity is computed as a dot product everywhere downstream."""
        vectors = embedder_with(RecordingModel()).embed_documents(["a", "b", "c"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_output_is_float32(self) -> None:
        """Chroma and the dedup pass both expect float32; float64 doubles index size."""
        assert embedder_with(RecordingModel()).embed_documents(["a"]).dtype == np.float32

    def test_empty_input_returns_a_stackable_empty_matrix(self) -> None:
        """A bare (0,) array would fail to stack against real (n, d) results."""
        vectors = embedder_with(RecordingModel(dimension=4)).embed_documents([])
        assert vectors.shape == (0, 4)

    def test_query_is_prefixed_with_the_instruction(self) -> None:
        model = RecordingModel()
        embedder_with(model).embed_query("how do I add a dependency")
        assert model.seen == [BGE_QUERY_INSTRUCTION + "how do I add a dependency"]

    def test_documents_are_not_prefixed(self) -> None:
        """Applying the query instruction to documents costs retrieval quality."""
        model = RecordingModel()
        embedder_with(model).embed_documents(["a passage of documentation"])
        assert model.seen == ["a passage of documentation"]

    def test_query_returns_a_single_vector(self) -> None:
        assert embedder_with(RecordingModel(dimension=4)).embed_query("q").shape == (4,)


class TestFailsLoudly:
    def test_unknown_model_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding model"):
            _ = FastEmbedEmbedder("not-a-real/model").dimension

    def test_non_positive_batch_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            FastEmbedEmbedder(batch_size=0)

    def test_a_zero_vector_does_not_become_nan(self) -> None:
        """A degenerate embedding must not poison every later similarity computation."""

        class ZeroModel(RecordingModel):
            def embed(self, documents: Sequence[str], **_: Any) -> Iterable[Any]:
                return [np.zeros(self.dimension) for _ in documents]

        vectors = embedder_with(ZeroModel()).embed_documents(["x"])
        assert not np.isnan(vectors).any()


class TestTopicEmbedder:
    """The stand-in must behave like an embedder, or every test using it proves nothing."""

    def test_same_topic_sentences_are_identical(self, topic_embedder: Embedder) -> None:
        vectors = topic_embedder.embed_documents(["alpha one", "alpha two"])
        assert np.isclose(float(vectors[0] @ vectors[1]), 1.0)

    def test_different_topic_sentences_are_orthogonal(self, topic_embedder: Embedder) -> None:
        vectors = topic_embedder.embed_documents(["alpha one", "beta one"])
        assert np.isclose(float(vectors[0] @ vectors[1]), 0.0)


@pytest.mark.slow
class TestRealModel:
    """One end-to-end check against the actual model, since every contract above is faked."""

    def test_bge_small_produces_normalised_384_dimensional_vectors(self) -> None:
        embedder = FastEmbedEmbedder()
        assert embedder.dimension == 384
        vectors = embedder.embed_documents(["FastAPI dependency injection", "unrelated text"])
        assert vectors.shape == (2, 384)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_related_text_scores_higher_than_unrelated(self) -> None:
        embedder = FastEmbedEmbedder()
        query = embedder.embed_query("How do I declare a path parameter?")
        related, unrelated = embedder.embed_documents(
            [
                "Declare path parameters with the same syntax used by Python format strings.",
                "Install Docker Desktop and start the daemon.",
            ]
        )
        assert float(query @ related) > float(query @ unrelated)
