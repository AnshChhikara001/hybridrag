"""The FastAPI service: `POST /v1/ask`, `GET /v1/documents`, `POST /v1/ingest`.

Run it with: uv run uvicorn hybridrag.api:app --reload
"""

from __future__ import annotations

from hybridrag.api.app import app

__all__ = ["app"]
