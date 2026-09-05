"""OpenAI embedder tests.

No network. A recording stand-in replaces the SDK client, so what is under test is our
batching, normalisation and spend accounting -- not OpenAI's service. The spend assertions
matter most: this is the only component in the project that can cost money, and a batching
bug that silently doubles requests is a budget bug, not just a performance one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import SecretStr

from hybridrag.embedding import Embedder
from hybridrag.embedding_openai import BudgetExceededError, OpenAIEmbedder


class FakeEmbeddings:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.batch_sizes: list[int] = []

    def create(self, *, model: str, input: list[str], **_: Any) -> Any:
        self.batch_sizes.append(len(input))
        # Deliberately un-normalised, so the caller's normalisation is what gets measured.
        data = [
            type("Item", (), {"embedding": [float(i + 2)] * self.dimension})()
            for i in range(len(input))
        ]
        usage = type("Usage", (), {"total_tokens": sum(len(t.split()) for t in input)})()
        return type("Response", (), {"data": data, "usage": usage})()


def embedder(dimension: int = 1536, **kwargs: Any) -> tuple[OpenAIEmbedder, FakeEmbeddings]:
    instance = OpenAIEmbedder(SecretStr("sk-test"), **kwargs)
    fake = FakeEmbeddings(instance.dimension if dimension == 1536 else dimension)
    instance._client = type("Client", (), {"embeddings": fake})()
    return instance, fake


class TestContracts:
    def test_satisfies_the_embedder_protocol(self) -> None:
        assert isinstance(OpenAIEmbedder(SecretStr("sk-test")), Embedder)

    def test_default_dimension_is_the_model_native_size(self) -> None:
        assert OpenAIEmbedder(SecretStr("sk-test")).dimension == 1536

    def test_shortened_dimensions_are_honoured(self) -> None:
        assert OpenAIEmbedder(SecretStr("sk-test"), dimensions=512).dimension == 512

    def test_vectors_are_unit_length(self) -> None:
        instance, _ = embedder()
        vectors = instance.embed_documents(["alpha", "beta"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_output_is_float32(self) -> None:
        instance, _ = embedder()
        assert instance.embed_documents(["alpha"]).dtype == np.float32

    def test_empty_input_makes_no_request(self) -> None:
        instance, fake = embedder()
        assert instance.embed_documents([]).shape == (0, 1536)
        assert fake.batch_sizes == []

    def test_query_returns_a_single_vector(self) -> None:
        instance, _ = embedder()
        assert instance.embed_query("q").shape == (1536,)


class TestSpendAccounting:
    def test_tokens_are_taken_from_the_api_response(self) -> None:
        """Counted from what the server reports, so the ledger matches the invoice."""
        instance, _ = embedder()
        instance.embed_documents(["one two three", "four five"])
        assert instance.tokens_used == 5

    def test_cost_is_derived_from_the_published_rate(self) -> None:
        instance, _ = embedder()
        instance.tokens_used = 1_000_000
        assert instance.estimated_cost_usd == pytest.approx(0.02)

    def test_spend_accumulates_across_calls(self) -> None:
        instance, _ = embedder()
        instance.embed_documents(["one two"])
        instance.embed_documents(["three four"])
        assert instance.tokens_used == 4

    def test_a_budget_stops_the_run_before_spending(self) -> None:
        """A runaway loop must raise, not quietly drain a prepaid balance."""
        instance, fake = embedder(token_budget=3)
        instance.embed_documents(["one two three"])
        with pytest.raises(BudgetExceededError, match="past the budget"):
            instance.embed_documents(["four five six seven"])
        assert len(fake.batch_sizes) == 1, "a request was sent after the budget was hit"

    def test_no_budget_means_no_limit(self) -> None:
        instance, _ = embedder()
        instance.embed_documents(["a b c d e"] * 10)
        assert instance.tokens_used == 50


class TestBatching:
    def test_small_inputs_go_in_one_request(self) -> None:
        instance, fake = embedder()
        instance.embed_documents(["alpha", "beta", "gamma"])
        assert fake.batch_sizes == [3]

    def test_batches_split_on_item_count(self) -> None:
        instance, fake = embedder()
        instance.embed_documents(["x"] * 5000)
        assert fake.batch_sizes == [2048, 2048, 904]

    def test_batches_split_on_token_count(self) -> None:
        """2048 short strings fit in a request; 2048 full chunks do not."""
        instance, fake = embedder()
        instance.embed_documents(["word " * 5000] * 100)
        assert len(fake.batch_sizes) > 1
        assert sum(fake.batch_sizes) == 100


class TestFailsLoudly:
    def test_unknown_model_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding model"):
            OpenAIEmbedder(SecretStr("sk-test"), "text-embedding-9-enormous")

    def test_a_missing_key_is_rejected(self) -> None:
        """Better than a confusing 401 from the first request of a long run."""
        with pytest.raises(ValueError, match="API key is required"):
            OpenAIEmbedder(SecretStr(""))

    def test_a_plain_string_key_is_accepted(self) -> None:
        assert OpenAIEmbedder("sk-test").dimension == 1536
