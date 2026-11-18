"""FastAPI application entrypoint and lifespan wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orderops import __version__
from orderops.agent.providers import AgentModelProvider, build_model_provider
from orderops.agent.runner import AgentRunner
from orderops.api.routes import router
from orderops.config import Settings, get_settings
from orderops.db import DatabaseManager
from orderops.tools import OrderOpsToolset, build_default_registry


def create_app(
    settings: Settings | None = None,
    db: DatabaseManager | None = None,
    provider: AgentModelProvider | None = None,
) -> FastAPI:
    """Create and configure a FastAPI instance with isolated dependencies."""
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        managed_db = db is None
        active_db = db or DatabaseManager(active_settings.database_path)
        active_db.init_schema()
        active_provider = provider or build_model_provider(active_settings)
        toolset = OrderOpsToolset(db=active_db, settings=active_settings)
        registry = build_default_registry(toolset)
        runner = AgentRunner(
            provider=active_provider,
            registry=registry,
            settings=active_settings,
        )
        app.state.settings = active_settings
        app.state.db = active_db
        app.state.toolset = toolset
        app.state.runner = runner
        try:
            yield
        finally:
            if managed_db:
                active_db.close()

    app = FastAPI(
        title="OrderOps Agent API",
        description=(
            "Bounded tool-calling AI agent for e-commerce order triage and refunds "
            "with deterministic financial guardrails, HITL supervisor approval queue, "
            "persistent multi-turn sessions, and SSE trace streaming."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
