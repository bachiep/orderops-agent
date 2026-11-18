"""FastAPI routes for OrderOps Agent service (Ops Console, REST, SSE, HITL, and Evals)."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from orderops import __version__
from orderops.agent.runner import AgentRunner
from orderops.agent.schemas import AgentRunRequest, TriageResolution
from orderops.config import Settings
from orderops.db import (
    DatabaseManager,
    IdempotencyConflictError,
    RefundAlreadyDecidedError,
    RefundNotFoundError,
    decide_refund_request,
    get_refund_request_detail,
    get_session_history_for_llm,
    get_session_messages,
    hash_canonical_payload,
    list_demo_orders,
    list_recent_agent_runs,
    list_refund_requests,
    list_sessions,
    lookup_idempotency_record,
    persist_agent_run,
    save_idempotency_record,
)
from orderops.tools.handlers import OrderOpsToolset
from orderops.tools.schemas import CreateRefundRequestInput, ReasonCode

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


class RefundSubmissionRequest(BaseModel):
    """Direct refund submission payload supporting Idempotency-Key and session linking."""

    order_id: str = Field(..., min_length=3, max_length=40)
    amount_vnd: int = Field(..., gt=0)
    reason: str = Field(..., min_length=3, max_length=500)
    reason_code: ReasonCode = Field(default="defective")
    session_id: str | None = Field(default=None, max_length=80)

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("order_id cannot be blank.")
        return cleaned

    @field_validator("reason")
    @classmethod
    def validate_reason_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 3:
            raise ValueError("reason must contain at least 3 non-whitespace characters.")
        return cleaned


def _normalize_idempotency_key(raw_key: str | None) -> str | None:
    if raw_key is None:
        return None
    cleaned = raw_key.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Idempotency-Key header cannot be blank.")
    return cleaned


def _problem_response(
    *,
    status_code: int,
    problem_type: str,
    title: str,
    detail: str,
    request: Request,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": problem_type,
        "title": title,
        "status": status_code,
        "detail": detail,
        "instance": str(request.url.path),
    }
    if extra:
        body.update(extra)
    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type="application/problem+json",
    )


def _agent_request_hash(payload: AgentRunRequest) -> str:
    return hash_canonical_payload(
        {
            "message": payload.message,
            "session_id": payload.session_id.strip() if payload.session_id else None,
            "customer_email": payload.customer_email.strip().lower()
            if payload.customer_email
            else None,
        }
    )


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


async def _execute_agent_sync(
    payload: AgentRunRequest,
    request: Request,
    response: Response,
    *,
    endpoint: str,
    raw_hash: str,
    idempotency_key: str | None = None,
) -> TriageResolution | JSONResponse:
    runner: AgentRunner = request.app.state.runner
    db: DatabaseManager = request.app.state.db
    clean_key = _normalize_idempotency_key(idempotency_key)
    if clean_key is not None:
        try:
            cached = lookup_idempotency_record(
                db,
                idempotency_key=clean_key,
                endpoint=endpoint,
                request_hash=raw_hash,
            )
        except IdempotencyConflictError as exc:
            return _problem_response(
                status_code=409,
                problem_type="https://orderops.local/problems/idempotency-conflict",
                title="Idempotency-Key Conflict",
                detail=str(exc),
                request=request,
            )
        if cached is not None:
            return JSONResponse(
                status_code=200,
                content=cached["response"],
                headers={"X-Idempotent-Replay": "true"},
            )

    history = _load_session_history(db, payload.session_id)
    resolution = await runner.run(payload, history=history)
    persist_agent_run(db, payload, resolution)
    if clean_key is not None:
        try:
            save_idempotency_record(
                db,
                idempotency_key=clean_key,
                endpoint=endpoint,
                request_hash=raw_hash,
                refund_id=resolution.refund_id,
                response_status=200,
                response_payload=resolution.model_dump(mode="json"),
            )
        except IdempotencyConflictError as exc:
            return _problem_response(
                status_code=409,
                problem_type="https://orderops.local/problems/idempotency-conflict",
                title="Idempotency-Key Conflict",
                detail=str(exc),
                request=request,
            )
        response.headers["X-Idempotent-Replay"] = "false"
    return resolution


def _execute_agent_sse(
    payload: AgentRunRequest,
    request: Request,
    *,
    endpoint: str,
    raw_hash: str,
    idempotency_key: str | None = None,
) -> Response:
    runner: AgentRunner = request.app.state.runner
    db: DatabaseManager = request.app.state.db
    clean_key = _normalize_idempotency_key(idempotency_key)
    if clean_key is not None:
        try:
            cached = lookup_idempotency_record(
                db,
                idempotency_key=clean_key,
                endpoint=endpoint,
                request_hash=raw_hash,
            )
        except IdempotencyConflictError as exc:
            return _problem_response(
                status_code=409,
                problem_type="https://orderops.local/problems/idempotency-conflict",
                title="Idempotency-Key Conflict",
                detail=str(exc),
                request=request,
            )
        if cached is not None:
            cached_data = dict(cached["response"])

            async def _replay_generator() -> AsyncIterator[str]:
                guard_payload = {
                    "layer": "input",
                    "passed": cached_data.get("resolution_status") != "BLOCKED_GUARDRAIL",
                    "flags": cached_data.get("guardrail_flags", []),
                    "idempotent_replay": True,
                }
                guard_json = json.dumps(guard_payload, ensure_ascii=False)
                yield f"event: guardrail_check\ndata: {guard_json}\n\n"
                summary = str(cached_data.get("summary") or "")
                if summary:
                    delta_json = json.dumps({"delta": summary}, ensure_ascii=False)
                    yield f"event: assistant_delta\ndata: {delta_json}\n\n"
                completed_json = json.dumps(cached_data, ensure_ascii=False)
                yield f"event: run_completed\ndata: {completed_json}\n\n"

            return StreamingResponse(
                _replay_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Idempotent-Replay": "true",
                },
            )

    history = _load_session_history(db, payload.session_id)

    async def _event_generator() -> AsyncIterator[str]:
        async for event in runner.run_stream(payload, history=history):
            if event.event == "run_completed":
                resolution = TriageResolution.model_validate(event.data)
                persist_agent_run(db, payload, resolution)
                if clean_key is not None:
                    save_idempotency_record(
                        db,
                        idempotency_key=clean_key,
                        endpoint=endpoint,
                        request_hash=raw_hash,
                        refund_id=resolution.refund_id,
                        response_status=200,
                        response_payload=resolution.model_dump(mode="json"),
                    )
            serialized = json.dumps(event.data, ensure_ascii=False)
            yield f"event: {event.event}\ndata: {serialized}\n\n"

    stream_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if clean_key is not None:
        stream_headers["X-Idempotent-Replay"] = "false"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers=stream_headers,
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
async def run_agent(
    payload: AgentRunRequest,
    request: Request,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128),
) -> TriageResolution | JSONResponse:
    """Run the bounded OrderOps triage agent synchronously and persist masked audit logs."""
    return await _execute_agent_sync(
        payload,
        request,
        response,
        endpoint="POST /v1/agent/runs",
        raw_hash=_agent_request_hash(payload),
        idempotency_key=idempotency_key,
    )


@router.post("/v1/agent/runs/stream", tags=["agent"])
async def stream_agent_run(
    payload: AgentRunRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128),
) -> Response:
    """Stream real-time SSE guardrail, tool_call, tool_result, delta, and completion events."""
    return _execute_agent_sse(
        payload,
        request,
        endpoint="POST /v1/agent/runs/stream",
        raw_hash=_agent_request_hash(payload),
        idempotency_key=idempotency_key,
    )


@router.post("/v1/chat", response_model=TriageResolution, tags=["sessions"])
async def chat_turn(
    payload: AgentRunRequest,
    request: Request,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128),
) -> TriageResolution | JSONResponse:
    """Multi-turn chat endpoint that auto-provisions a session_id if omitted."""
    raw_hash = _agent_request_hash(payload)
    return await _execute_agent_sync(
        _with_chat_session(payload),
        request,
        response,
        endpoint="POST /v1/chat",
        raw_hash=raw_hash,
        idempotency_key=idempotency_key,
    )


@router.post("/v1/chat/stream", tags=["sessions"])
async def chat_turn_stream(
    payload: AgentRunRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128),
) -> Response:
    """Streaming SSE multi-turn chat endpoint that auto-provisions a session_id if omitted."""
    raw_hash = _agent_request_hash(payload)
    return _execute_agent_sse(
        _with_chat_session(payload),
        request,
        endpoint="POST /v1/chat/stream",
        raw_hash=raw_hash,
        idempotency_key=idempotency_key,
    )


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


@router.post("/v1/refunds", tags=["refunds"])
async def submit_refund_request(
    payload: RefundSubmissionRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128),
) -> JSONResponse:
    """Submit a refund request with optional Idempotency-Key replay and 409 conflict checks."""
    db: DatabaseManager = request.app.state.db
    toolset: OrderOpsToolset = getattr(request.app.state, "toolset", None) or OrderOpsToolset(
        db=db, settings=request.app.state.settings
    )
    clean_key = _normalize_idempotency_key(idempotency_key)
    try:
        result = toolset.create_refund_request(
            CreateRefundRequestInput(
                order_id=payload.order_id,
                amount_vnd=payload.amount_vnd,
                reason=payload.reason,
                reason_code=payload.reason_code,
            ),
            session_id=payload.session_id,
            idempotency_key=clean_key,
            idempotency_endpoint="POST /v1/refunds",
        )
    except IdempotencyConflictError as exc:
        return _problem_response(
            status_code=409,
            problem_type="https://orderops.local/problems/idempotency-conflict",
            title="Idempotency-Key Conflict",
            detail=str(exc),
            request=request,
        )

    if not result.created or not result.refund_id:
        return _problem_response(
            status_code=422,
            problem_type="https://orderops.local/problems/policy-rejection",
            title="Refund Policy Rejection",
            detail=result.message,
            request=request,
            extra={"policy_violation": result.policy_violation, "order_id": result.order_id},
        )

    detail = get_refund_request_detail(db, result.refund_id)
    body: dict[str, Any] = {
        **(detail or {}),
        "refund_id": result.refund_id,
        "order_id": result.order_id,
        "amount_vnd": result.amount_vnd,
        "idempotency_key": clean_key,
        "idempotent_replay": result.idempotent_replay,
        "message": result.message,
    }
    headers = (
        {"X-Idempotent-Replay": "true" if result.idempotent_replay else "false"}
        if clean_key is not None
        else None
    )
    return JSONResponse(status_code=200, content=body, headers=headers)


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
