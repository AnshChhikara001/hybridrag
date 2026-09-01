"""Embedding, behind a protocol.

Three call sites need vectors -- the semantic chunker, the dense index, and query time --
and Phase 4 has to swap the model to compare local `bge-small-en-v1.5` against OpenAI
`text-embedding-3-small`. A protocol keeps that swap to one constructor argument instead of
a rewrite, and lets the chunker tests run without downloading a model.

Two contracts every implementation owes its callers:

* **Vectors are L2-normalised.** Cosine similarity then reduces to a dot product, which is
  what Chroma's cosine space, near-duplicate detection, and the semantic chunker's distance
  computation all assume. Enforced here rather than trusted, because a model that stops
  normalising would otherwise turn every similarity score subtly wrong with no error.
* **Documents and queries embed differently.** `bge` was trained with an instruction prefix
  on the query side only; omitting it costs retrieval quality, and applying it to documents
  costs it again. Keeping the asymmetry inside the embedder means no caller can get it
  wrong, and a model without an instruction (OpenAI's) just carries an empty one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # typing-only import; the runtime import is deferred to first use
    from fastembed import TextEmbedding

Vector = NDArray[np.float32]

# Prescribed by the BAAI/bge model card for retrieval; queries only, never documents.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    """Turns text into unit-length vectors."""

    model_name: str

    @property
    def dimension(self) -> int:
        """Vector width, needed to size the index before anything is embedded."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        """Embed corpus text. Returns shape (len(texts), dimension)."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a search query. Returns shape (dimension,)."""
        ...


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed -- the project default.

    fastembed runs the model through onnxruntime rather than PyTorch, which keeps the
    dependency around 60 MB instead of ~2 GB and needs no GPU. Everything is local, so
    chunking experiments that re-embed the corpus repeatedly cost nothing and need no key.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        batch_size: int = 256,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self._model: TextEmbedding | None = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = _declared_dimension(self.model_name)
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        if not texts:
            # Shape matters: callers stack these, and (0,) would not stack against (n, d).
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = np.asarray(list(self._loaded().embed(list(texts), batch_size=self.batch_size)))
        return _unit(vectors.astype(np.float32, copy=False))

    def embed_query(self, text: str) -> Vector:
        row: Vector = self.embed_documents([self.query_instruction + text])[0]
        return row

    def _loaded(self) -> TextEmbedding:
        """Load the model on first use.

        Deferred so that importing this module -- which the test suite and the CLI both do
        -- never triggers a 60 MB download or a multi-second ONNX session start.
        """
        if self._model is None:
            from fastembed import TextEmbedding

            model = TextEmbedding(self.model_name)
            self._model = model
            self._dimension = int(np.asarray(next(iter(model.embed(["probe"])))).shape[0])
        return self._model


def _unit(vectors: Vector) -> Vector:
    """L2-normalise each row, leaving all-zero rows alone rather than dividing by zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0.0, 1.0, norms)).astype(np.float32, copy=False)


def _declared_dimension(model_name: str) -> int:
    """Vector width from fastembed's model registry, avoiding a forward pass to learn it."""
    from fastembed import TextEmbedding

    for description in TextEmbedding.list_supported_models():
        if description["model"] == model_name:
            return int(description["dim"])
    raise ValueError(
        f"Unknown embedding model {model_name!r}. Supported models are listed by "
        "fastembed.TextEmbedding.list_supported_models()."
    )
