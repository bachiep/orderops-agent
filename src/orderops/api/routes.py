"""FastAPI routes for OrderOps Agent service (REST, SSE streaming, and audit inspection)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from orderops import __version__
from orderops.agent.runner import AgentRunner
from orderops.agent.schemas import AgentRunRequest, TriageResolution
from orderops.config import Settings
from orderops.db import (
    DatabaseManager,
    list_demo_orders,
    list_recent_agent_runs,
    persist_agent_run,
)

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


@router.get("/v1/orders", tags=["orders"])
async def get_orders(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    """List seeded orders with masked customer metadata for console inspection."""
    db: DatabaseManager = request.app.state.db
    orders = list_demo_orders(db, limit=limit)
    return {"orders": orders, "total": len(orders)}


@router.get("/v1/runs", tags=["agent"])
async def get_runs(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """List persisted masked agent run audit entries."""
    db: DatabaseManager = request.app.state.db
    runs = list_recent_agent_runs(db, limit=limit)
    return {"runs": runs, "total": len(runs)}


@router.post("/v1/agent/runs", response_model=TriageResolution, tags=["agent"])
async def run_agent(payload: AgentRunRequest, request: Request) -> TriageResolution:
    """Run the bounded OrderOps triage agent synchronously and persist masked audit logs."""
    runner: AgentRunner = request.app.state.runner
    db: DatabaseManager = request.app.state.db
    resolution = await runner.run(payload)
    persist_agent_run(db, payload, resolution)
    return resolution


@router.post("/v1/agent/runs/stream", tags=["agent"])
async def stream_agent_run(payload: AgentRunRequest, request: Request) -> StreamingResponse:
    """Stream real-time SSE guardrail, tool_call, tool_result, delta, and completion events."""
    runner: AgentRunner = request.app.state.runner
    db: DatabaseManager = request.app.state.db

    async def _event_generator() -> AsyncIterator[str]:
        async for event in runner.run_stream(payload):
            if event.event == "run_completed":
                resolution = TriageResolution.model_validate(event.data)
                persist_agent_run(db, payload, resolution)
            serialized = json.dumps(event.data, ensure_ascii=False)
            yield f"event: {event.event}\ndata: {serialized}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
