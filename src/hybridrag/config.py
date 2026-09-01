"""Application configuration.

All tunable values live here and arrive from the environment or a local `.env`, so
application code never carries literals and nothing sensitive is ever committed. API keys
are typed `SecretStr`, which keeps them out of reprs, logs and tracebacks -- a plain `str`
leaks into every exception that happens to include the settings object.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, validated at load time.

    Bounds are deliberate: an out-of-range chunk size or a percentile of 100 fails loudly
    at startup rather than producing a silently degenerate index hours later.
    """

    model_config = SettingsConfigDict(
        env_prefix="HYBRIDRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/index")
    # Kept outside index_dir on purpose: the index is a build artefact that gets wiped and
    # rebuilt, while the cache is what makes rebuilding cheap. Deleting one must not
    # destroy the other.
    cache_dir: Path = Path("data/cache")

    # --- Chunking
    chunk_tokens: int = Field(default=512, ge=64, le=4096)
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    semantic_percentile: float = Field(
        default=95.0,
        gt=0.0,
        lt=100.0,
        description="Sentence-distance percentile above which a topic boundary is declared.",
    )

    # --- Indexing
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    dedup_threshold: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Cosine similarity at or above which a chunk is treated as a duplicate.",
    )

    # --- Credentials (never hardcoded, never logged)
    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")

    @model_validator(mode="after")
    def _overlap_fits_inside_chunk(self) -> Settings:
        """Overlap at or beyond the chunk size makes the sliding window fail to advance."""
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be smaller than "
                f"chunk_tokens ({self.chunk_tokens}); otherwise chunking cannot make progress."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once.

    Cached so configuration is loaded and validated a single time. Tests that need a
    different configuration construct `Settings(...)` directly rather than mutating this.
    """
    return Settings()
