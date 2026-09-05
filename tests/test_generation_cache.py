"""Response cache behaviour.

Most of these are about the *key*. A cache that misses too often merely costs money; a
cache that hits when it should not returns an answer generated under different settings,
which during a parameter sweep is a wrong measurement that looks like a fast one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybridrag.generation import CachedLanguageModel, Completion, GenerationError


class CountingModel:
    """Returns a numbered reply, so a second real call is visible in the text itself."""

    def __init__(self, temperature: float = 0.0, max_output_tokens: int = 1024) -> None:
        self.model_name = "counting-model"
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.calls = 0

    @property
    def fingerprint(self) -> str:
        return f"counting|t={self.temperature}|max={self.max_output_tokens}"

    def generate(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> Completion:
        self.calls += 1
        return Completion(
            text=f"reply {self.calls}",
            model=self.model_name,
            input_tokens=100,
            output_tokens=20,
            thinking_tokens=3,
            cost_usd=0.0004,
            latency_s=1.5,
        )


class FailingModel:
    model_name = "failing-model"

    @property
    def fingerprint(self) -> str:
        return "failing"

    def generate(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> Completion:
        raise GenerationError("rate limited")


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "completions.sqlite"


def test_a_repeated_prompt_is_served_without_calling_the_model(cache_path: Path) -> None:
    inner = CountingModel()
    cache = CachedLanguageModel(inner, cache_path)

    first = cache.generate("what is a query parameter", system="be terse")
    second = cache.generate("what is a query parameter", system="be terse")

    assert inner.calls == 1
    assert first.text == second.text == "reply 1"
    assert (cache.hits, cache.misses) == (1, 1)


def test_a_hit_reports_zero_cost_because_nothing_was_spent_again(cache_path: Path) -> None:
    cache = CachedLanguageModel(CountingModel(), cache_path)
    live = cache.generate("q")
    hit = cache.generate("q")

    assert live.cached is False
    assert live.cost_usd == pytest.approx(0.0004)
    assert hit.cached is True
    assert hit.cost_usd == 0.0
    # Latency becomes lookup time, so a benchmark that includes hits understates itself.
    assert hit.latency_s < live.latency_s


def test_token_counts_survive_a_hit_unchanged(cache_path: Path) -> None:
    """They describe the prompt and response, not the call that fetched them."""
    cache = CachedLanguageModel(CountingModel(), cache_path)
    cache.generate("q")
    hit = cache.generate("q")

    assert (hit.input_tokens, hit.output_tokens, hit.thinking_tokens) == (100, 20, 3)


def test_a_different_prompt_is_a_different_entry(cache_path: Path) -> None:
    inner = CountingModel()
    cache = CachedLanguageModel(inner, cache_path)

    cache.generate("first question")
    cache.generate("second question")

    assert inner.calls == 2


def test_a_different_system_prompt_is_a_different_entry(cache_path: Path) -> None:
    """Prompt engineering changes the system prompt; the cache must notice."""
    inner = CountingModel()
    cache = CachedLanguageModel(inner, cache_path)

    cache.generate("q", system="rules v1")
    cache.generate("q", system="rules v2")

    assert inner.calls == 2


def test_a_different_output_cap_is_a_different_entry(cache_path: Path) -> None:
    """It decides where a long answer is truncated, so it changes the response."""
    inner = CountingModel()
    cache = CachedLanguageModel(inner, cache_path)

    cache.generate("q", max_output_tokens=256)
    cache.generate("q", max_output_tokens=1024)

    assert inner.calls == 2


def test_changing_temperature_invalidates_the_entry(cache_path: Path) -> None:
    """The failure this design exists to prevent: serving temperature-0 answers mid-sweep."""
    cold = CachedLanguageModel(CountingModel(temperature=0.0), cache_path)
    cold.generate("q")
    cold.close()

    warm_inner = CountingModel(temperature=0.7)
    warm = CachedLanguageModel(warm_inner, cache_path)
    warm.generate("q")

    assert warm_inner.calls == 1
    assert warm.hits == 0


def test_a_prompt_split_differently_does_not_collide(cache_path: Path) -> None:
    """NUL-separated fields, so 'ab' + '' and 'a' + 'b' cannot hash alike."""
    inner = CountingModel()
    cache = CachedLanguageModel(inner, cache_path)

    cache.generate("b", system="a")
    cache.generate("", system="ab")

    assert inner.calls == 2


def test_entries_survive_closing_and_reopening_the_file(cache_path: Path) -> None:
    """The point of persistence: an interrupted sweep resumes free."""
    first_inner = CountingModel()
    writer = CachedLanguageModel(first_inner, cache_path)
    writer.generate("q")
    writer.close()

    second_inner = CountingModel()
    reader = CachedLanguageModel(second_inner, cache_path)
    result = reader.generate("q")

    assert second_inner.calls == 0
    assert result.text == "reply 1"
    assert result.cached is True
    reader.close()


def test_a_failed_call_is_not_cached(cache_path: Path) -> None:
    """A rate limit must never be stored as though it were an answer."""
    cache = CachedLanguageModel(FailingModel(), cache_path)

    with pytest.raises(GenerationError):
        cache.generate("q")

    assert len(cache) == 0


def test_read_only_mode_never_serves_a_stored_entry(cache_path: Path) -> None:
    """For measuring true latency, or proving a prompt change actually changed answers."""
    warm = CachedLanguageModel(CountingModel(), cache_path)
    warm.generate("q")
    warm.close()

    inner = CountingModel()
    live = CachedLanguageModel(inner, cache_path, read_only=True)
    result = live.generate("q")

    assert inner.calls == 1
    assert result.cached is False
    assert live.hits == 0


def test_an_unreadable_row_is_treated_as_a_miss_not_an_error(cache_path: Path) -> None:
    """A cache is not a store of record: a stale entry costs one call, not the run."""
    inner = CountingModel()
    cache = CachedLanguageModel(inner, cache_path)
    cache.generate("q")
    cache._db.execute("UPDATE completions SET payload = '{\"nonsense\": true}'")
    cache._db.commit()

    result = cache.generate("q")

    assert inner.calls == 2
    assert result.text == "reply 2"


def test_the_wrapper_reports_the_inner_models_identity(cache_path: Path) -> None:
    """Callers read `model_name`, and caching must not change what answered."""
    inner = CountingModel()
    cache = CachedLanguageModel(inner, cache_path)

    assert cache.model_name == inner.model_name
    assert cache.fingerprint == inner.fingerprint
