"""FastAPI routes for OrderOps Agent service (Ops Console, REST, SSE, HITL, and Evals)."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from orderops import __version__
from orderops.agent.runner import AgentRunner
from orderops.agent.schemas import AgentRunRequest, TriageResolution
from orderops.config import Settings
from orderops.db import (
    DatabaseManager,
    RefundAlreadyDecidedError,
    RefundNotFoundError,
    decide_refund_request,
    get_session_history_for_llm,
    get_session_messages,
    list_demo_orders,
    list_recent_agent_runs,
    list_refund_requests,
    list_sessions,
    persist_agent_run,
)

router = APIRouter()

_UI_INDEX_PATH = Path(__file__).resolve().parent.parent / "ui" / "static" / "index.html"


class HealthResponse(BaseModel):
    """Health check response summarizing database and runtime status."""

    status: str
    version: str
    llm_provider: str
    llm_model: str
    database_path: str
    table_counts: dict[str, int]


class RefundDecisionRequest(BaseModel):
    """Supervisor HITL decision payload for a pending refund request."""

    decision: Literal["APPROVE", "REJECT"]
    reviewer_note: str = Field(..., min_length=1, max_length=500)
    reviewer_id: str | None = Field(default="supervisor-console", max_length=80)

    @field_validator("reviewer_note")
    @classmethod
    def validate_note_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("reviewer_note cannot be blank.")
        return cleaned


def _resolve_eval_results_dir(settings: Settings) -> Path | None:
    candidates = (
        Path(settings.eval_results_dir),
        Path(__file__).resolve().parents[3] / "evals" / "results",
        Path.cwd() / "evals" / "results",
    )
    for directory in candidates:
        if (directory / "eval_summary.json").is_file():
            return directory
    return None


def _load_session_history(db: DatabaseManager, session_id: str | None) -> list[dict[str, str]]:
    if not session_id or not session_id.strip():
        return []
    return get_session_history_for_llm(db, session_id.strip())


async def _execute_agent_sync(payload: AgentRunRequest, request: Request) -> TriageResolution:
    runner: AgentRunner = request.app.state.runner
    db: DatabaseManager = request.app.state.db
    history = _load_session_history(db, payload.session_id)
    resolution = await runner.run(payload, history=history)
    persist_agent_run(db, payload, resolution)
    return resolution


def _execute_agent_sse(payload: AgentRunRequest, request: Request) -> StreamingResponse:
    runner: AgentRunner = request.app.state.runner
    db: DatabaseManager = request.app.state.db
    history = _load_session_history(db, payload.session_id)

    async def _event_generator() -> AsyncIterator[str]:
        async for event in runner.run_stream(payload, history=history):
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


def _with_chat_session(payload: AgentRunRequest) -> AgentRunRequest:
    if payload.session_id and payload.session_id.strip():
        return payload.model_copy(update={"session_id": payload.session_id.strip()})
    return payload.model_copy(update={"session_id": f"sess_{uuid.uuid4().hex[:12]}"})


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ops_console_ui() -> HTMLResponse:
    """Serve the 3-tab Swiss Enterprise OrderOps Console UI."""
    if not _UI_INDEX_PATH.is_file():
        raise HTTPException(status_code=404, detail="Ops Console UI bundle not found.")
    return HTMLResponse(content=_UI_INDEX_PATH.read_text(encoding="utf-8"))


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
    return await _execute_agent_sync(payload, request)


@router.post("/v1/agent/runs/stream", tags=["agent"])
async def stream_agent_run(payload: AgentRunRequest, request: Request) -> StreamingResponse:
    """Stream real-time SSE guardrail, tool_call, tool_result, delta, and completion events."""
    return _execute_agent_sse(payload, request)


@router.post("/v1/chat", response_model=TriageResolution, tags=["sessions"])
async def chat_turn(payload: AgentRunRequest, request: Request) -> TriageResolution:
    """Multi-turn chat endpoint that auto-provisions a session_id if omitted."""
    return await _execute_agent_sync(_with_chat_session(payload), request)


@router.post("/v1/chat/stream", tags=["sessions"])
async def chat_turn_stream(payload: AgentRunRequest, request: Request) -> StreamingResponse:
    """Streaming SSE multi-turn chat endpoint that auto-provisions a session_id if omitted."""
    return _execute_agent_sse(_with_chat_session(payload), request)


@router.get("/v1/sessions", tags=["sessions"])
async def get_sessions(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """List recent multi-turn support sessions."""
    db: DatabaseManager = request.app.state.db
    sessions = list_sessions(db, limit=limit)
    return {"sessions": sessions, "total": len(sessions)}


@router.get("/v1/sessions/{session_id}", tags=["sessions"])
async def get_session_detail(session_id: str, request: Request) -> dict[str, Any]:
    """Return full ordered multi-turn session history and associated run IDs."""
    db: DatabaseManager = request.app.state.db
    return get_session_messages(db, session_id)


@router.get("/v1/refunds/pending", tags=["refunds"])
async def get_refunds_queue(
    request: Request,
    status: str = Query(
        default="PENDING_APPROVAL",
        description="Filter: PENDING_APPROVAL | APPROVED | AUTO_APPROVED | REJECTED | ALL",
    ),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """List refund requests with order metadata and supervisor review status counts."""
    db: DatabaseManager = request.app.state.db
    try:
        return list_refund_requests(db, status=status, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/v1/refunds/{refund_id}/decision", tags=["refunds"])
async def post_refund_decision(
    refund_id: str,
    payload: RefundDecisionRequest,
    request: Request,
) -> dict[str, Any]:
    """Atomically approve or reject a pending refund request inside BEGIN IMMEDIATE."""
    db: DatabaseManager = request.app.state.db
    try:
        return decide_refund_request(
            db,
            refund_id=refund_id,
            decision=payload.decision,
            reviewer_note=payload.reviewer_note,
            reviewer_id=payload.reviewer_id,
        )
    except RefundNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RefundAlreadyDecidedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/evals/latest", tags=["evals"])
async def get_latest_evals(request: Request) -> dict[str, Any]:
    """Return latest 30-scenario evaluation summary and per-scenario breakdown."""
    settings: Settings = request.app.state.settings
    results_dir = _resolve_eval_results_dir(settings)
    if results_dir is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run 'python -m evals.run_eval'.",
        )
    summary = json.loads((results_dir / "eval_summary.json").read_text(encoding="utf-8"))
    per_scenario_path = results_dir / "per_scenario_results.json"
    scenarios: list[dict[str, Any]] = (
        json.loads(per_scenario_path.read_text(encoding="utf-8"))
        if per_scenario_path.is_file()
        else []
    )
    return {
        "summary": summary,
        "scenarios": scenarios,
        "total_scenarios": len(scenarios),
        "passed_scenarios": sum(1 for item in scenarios if item.get("passed") is True),
    }
