"""Token counting, sharing the embedding model's own tokenizer.

Chunk budgets must be denominated in the same tokens the embedder will see. Counting with
a different tokenizer -- a generic BPE, or characters divided by four -- means a "512
token" chunk can silently exceed the model's 512-token input limit and be truncated at
embed time, dropping its tail from the index without any error.

The model's tokenizer truncates at its max length by default, which makes it report 512
for any longer text. Truncation is disabled here so counts are true lengths; that
behaviour was observed directly while sizing this corpus, where it capped every long
section at exactly 512.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, runtime_checkable

from tokenizers import Tokenizer


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens and maps them back to character positions."""

    def count(self, text: str) -> int:
        """Number of tokens in `text`."""
        ...

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Per-token (start, end) character spans, so token windows become char spans."""
        ...


class HuggingFaceTokenCounter:
    """Wraps the tokenizer belonging to a fastembed embedding model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._tokenizer = _load_tokenizer(model_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        return [
            (start, end)
            for start, end in self._tokenizer.encode(text, add_special_tokens=False).offsets
            if end > start  # drop zero-width tokens, which would produce empty spans
        ]


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str) -> Tokenizer:
    """Load and configure the tokenizer once per model.

    Prefers the tokenizer fastembed has already cached beside the model, so chunking and
    embedding are guaranteed to agree on what a token is. That attribute is a fastembed
    internal and not part of its typed API, so it is reached with `getattr` and falls back
    to the Hub if fastembed rearranges its internals.
    """
    from fastembed import TextEmbedding

    tokenizer: Tokenizer | None = getattr(TextEmbedding(model_name).model, "tokenizer", None)
    if tokenizer is None:  # pragma: no cover - depends on fastembed internals
        tokenizer = Tokenizer.from_pretrained(model_name)

    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer
