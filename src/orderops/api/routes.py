"""FastAPI routes for OrderOps Agent service."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from orderops import __version__
from orderops.config import Settings
from orderops.db import DatabaseManager

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response summarizing database and runtime status."""

    status: str
    version: str
    llm_provider: str
    llm_model: str
    database_path: str
    table_counts: dict[str, int]


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(request: Request) -> HealthResponse:
    """Return service readiness and SQLite table counts."""
    db: DatabaseManager = request.app.state.db
    settings: Settings = request.app.state.settings
    counts = db.get_table_counts()
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        database_path=settings.database_path,
        table_counts=counts,
    )
