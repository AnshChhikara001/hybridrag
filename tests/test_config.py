"""Tests that misconfiguration fails loudly at startup rather than silently later."""

import pytest
from pydantic import ValidationError

from hybridrag.config import Settings


class TestSettingsValidation:
    def test_defaults_are_valid(self) -> None:
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.chunk_tokens == 512
        assert s.chunk_overlap_tokens == 64
        assert s.dedup_threshold == 0.95

    def test_overlap_equal_to_chunk_size_is_rejected(self) -> None:
        """A non-advancing sliding window would loop forever; catch it at load time."""
        with pytest.raises(ValidationError, match="must be smaller than"):
            Settings(_env_file=None, chunk_tokens=512, chunk_overlap_tokens=512)  # type: ignore[call-arg]

    def test_overlap_larger_than_chunk_size_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be smaller than"):
            Settings(_env_file=None, chunk_tokens=256, chunk_overlap_tokens=300)  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad", [0.0, 101.0, 100.0])
    def test_semantic_percentile_bounds(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, semantic_percentile=bad)  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad", [0.0, 1.5])
    def test_dedup_threshold_bounds(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, dedup_threshold=bad)  # type: ignore[call-arg]


class TestSecrets:
    def test_api_keys_default_to_none(self) -> None:
        assert Settings(_env_file=None).gemini_api_key is None  # type: ignore[call-arg]

    def test_api_key_is_masked_in_repr(self) -> None:
        """A plain str would leak the key into every traceback carrying settings."""
        s = Settings(_env_file=None, GEMINI_API_KEY="super-secret-value")  # type: ignore[call-arg, arg-type]
        assert "super-secret-value" not in repr(s)
        assert s.gemini_api_key is not None
        assert s.gemini_api_key.get_secret_value() == "super-secret-value"
