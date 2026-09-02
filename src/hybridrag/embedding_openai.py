"""OpenAI embeddings, as a second implementation of the `Embedder` protocol.

Exists for two reasons. Decision D3 wanted `text-embedding-3-small` measured against the
local model rather than assumed better. And embedding the corpus locally takes 28 minutes
on one core -- enough to make a laptop unusable -- while the same work here is a handful of
batched HTTP requests.

This module spends money, so it counts what it spends. Every response carries a token
count; those are accumulated and converted to dollars at the published rate, and an
optional `token_budget` makes a runaway loop raise instead of quietly draining a prepaid
balance. Wrapped in `CachedEmbedder`, repeat text is never paid for twice.

No instruction prefix here, unlike `bge`: OpenAI's embedding models are trained to take
queries and documents in the same form, so adding one would hurt rather than help.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tiktoken
from openai import OpenAI, omit
from pydantic import SecretStr

from hybridrag.embedding import Vector

# Published price per million input tokens. Kept here so cost reporting is derived from
# one number that is easy to check against the pricing page rather than scattered.
_PRICE_PER_1M_TOKENS = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}
_NATIVE_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
# The API caps a single request at 300k tokens. Batching well under that leaves room for
# the tokenizer disagreeing slightly with the server's own count.
_REQUEST_TOKEN_BUDGET = 200_000
_MAX_INPUTS_PER_REQUEST = 2048


class BudgetExceededError(RuntimeError):
    """Raised before a request that would push spending past the configured budget."""


class OpenAIEmbedder:
    """Hosted embeddings with explicit spend accounting."""

    def __init__(
        self,
        api_key: SecretStr | str,
        model_name: str = "text-embedding-3-small",
        *,
        dimensions: int | None = None,
        token_budget: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 5,
    ) -> None:
        if model_name not in _PRICE_PER_1M_TOKENS:
            raise ValueError(
                f"Unknown embedding model {model_name!r}. Known models and their prices "
                f"are listed in {__name__}._PRICE_PER_1M_TOKENS."
            )
        secret = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not secret:
            raise ValueError("An OpenAI API key is required. Set OPENAI_API_KEY in .env.")

        self.model_name = model_name
        self.dimensions = dimensions
        self.token_budget = token_budget
        self.tokens_used = 0
        self.requests_made = 0
        # The SDK retries with exponential backoff on rate limits and 5xx, which is exactly
        # the behaviour a bulk indexing run needs.
        self._client = OpenAI(api_key=secret, timeout=timeout, max_retries=max_retries)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def dimension(self) -> int:
        return self.dimensions or _NATIVE_DIMENSIONS[self.model_name]

    @property
    def estimated_cost_usd(self) -> float:
        """Dollars spent by this instance, from token counts the API itself reported."""
        return self.tokens_used / 1_000_000 * _PRICE_PER_1M_TOKENS[self.model_name]

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        rows = [self._request(batch) for batch in self._batches(list(texts))]
        return _unit(np.vstack(rows).astype(np.float32, copy=False))

    def embed_query(self, text: str) -> Vector:
        row: Vector = self.embed_documents([text])[0]
        return row

    def _batches(self, texts: list[str]) -> list[list[str]]:
        """Split by token count as well as item count.

        Batching on item count alone is how a bulk run dies partway through: 2048 short
        strings fit comfortably, 2048 full-size chunks do not.
        """
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for text in texts:
            tokens = len(self._encoding.encode(text))
            over_tokens = current and current_tokens + tokens > _REQUEST_TOKEN_BUDGET
            over_count = len(current) >= _MAX_INPUTS_PER_REQUEST
            if over_tokens or over_count:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += tokens
        if current:
            batches.append(current)
        return batches

    def _request(self, batch: list[str]) -> Vector:
        self._check_budget(sum(len(self._encoding.encode(text)) for text in batch))
        response = self._client.embeddings.create(
            model=self.model_name,
            input=batch,
            # `omit` is the SDK's "leave this out of the request" sentinel; passing None
            # would send an explicit null and be rejected.
            dimensions=self.dimensions if self.dimensions else omit,
        )
        # The server's own count, not our estimate, so the ledger matches the invoice.
        self.tokens_used += response.usage.total_tokens
        self.requests_made += 1
        return np.asarray([item.embedding for item in response.data], dtype=np.float32)

    def _check_budget(self, incoming: int) -> None:
        if self.token_budget is None:
            return
        if self.tokens_used + incoming > self.token_budget:
            raise BudgetExceededError(
                f"This request would bring usage to "
                f"{self.tokens_used + incoming:,} tokens, past the budget of "
                f"{self.token_budget:,} (~${self.estimated_cost_usd:.4f} spent so far). "
                "Raise token_budget deliberately if this is expected."
            )


def _unit(vectors: Vector) -> Vector:
    """L2-normalise, as the protocol promises. Idempotent on already-unit vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0.0, 1.0, norms)).astype(np.float32, copy=False)
