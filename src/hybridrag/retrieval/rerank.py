"""Cross-encoder reranking: a second, more expensive pass over a short candidate list.

RRF fusion (D20) reads only rank positions -- it never looks at the query and a candidate
together. A cross-encoder does exactly that: it scores each `(query, chunk)` pair jointly,
which is slower per item but sees interactions rank fusion structurally cannot. Run over
every indexed chunk it would be far too slow; run over the ~20 candidates fusion already
narrowed the field to, it is milliseconds.

**Model (D8): `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed, not `sentence-transformers`.**
fastembed is already this project's embedding dependency (`embedding.py`), and its rerank
module ships an ONNX build of the same cross-encoder weights `sentence-transformers` would
load through PyTorch. Adding `sentence-transformers` would mean a second inference runtime
and a `torch` install running to hundreds of MB to a few GB on the 8 GB machine this project
is measured on (see `PROJECT_STATE.md`'s environment section) -- for a model fastembed
already serves in ~80 MB over onnxruntime, which is already installed. Same weights, same
$0, one fewer heavy dependency.

**Why retrieve top-20 and keep top-5, not rerank everything fusion returns.** The
cross-encoder's whole value is looking at query and chunk *together*, which costs one
forward pass per candidate -- fusion's rank read costs nothing per candidate. Reranking a
fixed, narrow pool bounds that cost regardless of how deep evaluation asks the retriever to
search, and keeps the pipeline's latency independent of `k`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hybridrag.retrieval.hybrid import RetrievedChunk, Retriever

if TYPE_CHECKING:  # typing-only import; the runtime import is deferred to first use
    from fastembed.rerank.cross_encoder import TextCrossEncoder

# The ONNX build of cross-encoder/ms-marco-MiniLM-L-6-v2 that fastembed ships (D8).
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# Candidates fetched from the base retriever before reranking, regardless of the caller's
# `k`. Fixed rather than `max(k, ...)`, so the pipeline's shape -- and its cost -- does not
# change with how deep an evaluation run happens to ask.
DEFAULT_RERANK_DEPTH = 20


@runtime_checkable
class Reranker(Protocol):
    """Scores `(query, document)` pairs jointly, best-match-highest."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Local cross-encoder scoring via fastembed's ONNX runtime.

    Loading is deferred to first use, exactly like `FastEmbedEmbedder` -- importing this
    module, which the CLI and the test suite both do, must not trigger an 80 MB download or
    an ONNX session start on its own.
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        self.model_name = model_name
        self._model: TextCrossEncoder | None = None

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        return [float(value) for value in self._loaded().rerank(query, list(documents))]

    def _loaded(self) -> TextCrossEncoder:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(self.model_name)
        return self._model


class RerankingRetriever:
    """Wraps a retriever with a cross-encoder pass: fetch a fixed pool, keep the top `k`.

    The candidate pool size is `rerank_depth`, not `max(k, rerank_depth)` -- asking this
    retriever for more results than `rerank_depth` still only ever reranks `rerank_depth`
    candidates and returns at most that many, the same way any retriever returns fewer
    results than requested when it has no more to give (never padded).
    """

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        *,
        rerank_depth: int = DEFAULT_RERANK_DEPTH,
    ) -> None:
        if rerank_depth <= 0:
            raise ValueError(f"rerank_depth must be positive, got {rerank_depth}")
        self.base = base
        self.reranker = reranker
        self.rerank_depth = rerank_depth

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        candidates = self.base.retrieve(query, k=self.rerank_depth)
        if not candidates:
            return []
        scores = self.reranker.score(query, [result.chunk.text for result in candidates])
        # Ties break on chunk id, matching `reciprocal_rank_fusion`, so the same query
        # reranks identically on every run rather than depending on sort stability.
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-scores[i], candidates[i].chunk.chunk_id),
        )
        return [
            candidates[position].model_copy(update={"rank": rank, "rerank_score": scores[position]})
            for rank, position in enumerate(order[:k], start=1)
        ]
