"""FastAPI application entrypoint and lifespan wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orderops import __version__
from orderops.api.routes import router
from orderops.config import Settings, get_settings
from orderops.db import DatabaseManager


def create_app(
    settings: Settings | None = None,
    db: DatabaseManager | None = None,
) -> FastAPI:
    """Create and configure a FastAPI instance with isolated dependencies."""
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        managed_db = db is None
        active_db = db or DatabaseManager(active_settings.database_path)
        active_db.init_schema()
        app.state.settings = active_settings
        app.state.db = active_db
        try:
            yield
        finally:
            if managed_db:
                active_db.close()

    app = FastAPI(
        title="OrderOps Agent API",
        description=(
            "Bounded tool-calling AI agent for e-commerce order triage and refunds "
            "with deterministic financial guardrails and SSE trace streaming."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
