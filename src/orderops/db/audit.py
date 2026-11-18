"""SQLite audit, session history, and supervisor refund persistence helpers."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from orderops.db.database import DatabaseManager
from orderops.guardrails.pii import get_structured_logger, mask_email, redact_pii

if TYPE_CHECKING:
    from orderops.agent.schemas import AgentRunRequest, TriageResolution

_refund_logger = get_structured_logger("orderops.refunds")


class RefundNotFoundError(LookupError):
    """Raised when a target refund_id does not exist."""


class RefundAlreadyDecidedError(RuntimeError):
    """Raised when a refund request is no longer pending human approval."""


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def persist_agent_run(
    db: DatabaseManager,
    request: AgentRunRequest,
    resolution: TriageResolution,
) -> None:
    """Persist a completed agent run and optional multi-turn session messages with PII masked."""
    masked_message, _ = redact_pii(request.message, redact_email=True)
    masked_summary, _ = redact_pii(resolution.summary, redact_email=True)
    masked_email = mask_email(request.customer_email) if request.customer_email else None
    raw_trace = json.dumps(
        [step.model_dump(mode="json") for step in resolution.tool_trace],
        ensure_ascii=False,
    )
    masked_trace, _ = redact_pii(raw_trace, redact_email=True)
    created_at = _utc_now_iso()
    session_id = request.session_id.strip() if request.session_id else None

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO agent_runs (
                run_id, session_id, customer_email_masked, user_message_masked,
                intent, resolution_status, refund_id, requires_human_review,
                guardrail_flags_json, tool_trace_json, steps_taken,
                latency_ms, total_tokens, estimated_cost_usd, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                resolution.run_id,
                session_id,
                masked_email,
                masked_message,
                resolution.intent,
                resolution.resolution_status,
                resolution.refund_id,
                1 if resolution.requires_human_review else 0,
                json.dumps(resolution.guardrail_flags, ensure_ascii=False),
                masked_trace,
                len(resolution.tool_trace),
                resolution.latency_ms,
                resolution.token_usage.total_tokens,
                resolution.estimated_cost_usd,
                created_at,
            ),
        )

        if session_id:
            conn.executemany(
                """
                INSERT INTO session_messages (
                    session_id, run_id, role, content_masked,
                    intent, resolution_status, refund_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    (
                        session_id,
                        resolution.run_id,
                        "user",
                        masked_message,
                        None,
                        None,
                        None,
                        created_at,
                    ),
                    (
                        session_id,
                        resolution.run_id,
                        "assistant",
                        masked_summary,
                        resolution.intent,
                        resolution.resolution_status,
                        resolution.refund_id,
                        created_at,
                    ),
                ],
            )


def get_session_history_for_llm(
    db: DatabaseManager,
    session_id: str,
    *,
    limit: int = 12,
) -> list[dict[str, str]]:
    """Load recent safe conversation turns for passing into follow-up agent runs."""
    rows = db.fetch_all(
        """
        SELECT sm.role, sm.content_masked, ar.resolution_status AS run_status
        FROM session_messages sm
        LEFT JOIN agent_runs ar ON sm.run_id = ar.run_id
        WHERE sm.session_id = ?
        ORDER BY sm.id ASC;
        """,
        (session_id,),
    )
    safe_turns = [
        {"role": str(row["role"]), "content": str(row["content_masked"])}
        for row in rows
        if row["run_status"] != "BLOCKED_GUARDRAIL"
    ]
    return safe_turns[-limit:] if limit > 0 else safe_turns


def get_session_messages(db: DatabaseManager, session_id: str) -> dict[str, Any]:
    """Fetch full ordered multi-turn session messages and associated run IDs."""
    rows = db.fetch_all(
        """
        SELECT id, session_id, run_id, role, content_masked,
               intent, resolution_status, refund_id, created_at
        FROM session_messages
        WHERE session_id = ?
        ORDER BY created_at ASC, id ASC;
        """,
        (session_id,),
    )
    run_ids: list[str] = []
    seen_runs: set[str] = set()
    messages: list[dict[str, Any]] = []
    for row in rows:
        run_id = row["run_id"]
        if run_id and run_id not in seen_runs:
            seen_runs.add(run_id)
            run_ids.append(str(run_id))
        messages.append(
            {
                "id": int(row["id"]),
                "session_id": row["session_id"],
                "run_id": run_id,
                "role": row["role"],
                "content": row["content_masked"],
                "content_masked": row["content_masked"],
                "intent": row["intent"],
                "resolution_status": row["resolution_status"],
                "refund_id": row["refund_id"],
                "created_at": row["created_at"],
            }
        )
    return {
        "session_id": session_id,
        "messages": messages,
        "run_ids": run_ids,
        "total_messages": len(messages),
    }


def list_sessions(db: DatabaseManager, *, limit: int = 20) -> list[dict[str, Any]]:
    """List recent multi-turn sessions ordered by latest message activity."""
    rows = db.fetch_all(
        """
        SELECT session_id,
               COUNT(*) AS message_count,
               SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS turn_count,
               MAX(created_at) AS updated_at,
               MAX(id) AS last_msg_id
        FROM session_messages
        GROUP BY session_id
        ORDER BY last_msg_id DESC
        LIMIT ?;
        """,
        (limit,),
    )
    sessions: list[dict[str, Any]] = []
    for row in rows:
        last_assistant = db.fetch_one(
            """
            SELECT intent, resolution_status
            FROM session_messages
            WHERE session_id = ? AND role = 'assistant'
            ORDER BY id DESC
            LIMIT 1;
            """,
            (row["session_id"],),
        )
        sessions.append(
            {
                "session_id": row["session_id"],
                "message_count": int(row["message_count"]),
                "turn_count": int(row["turn_count"]),
                "last_intent": last_assistant["intent"] if last_assistant else None,
                "last_status": last_assistant["resolution_status"] if last_assistant else None,
                "updated_at": row["updated_at"],
            }
        )
    return sessions


def list_refund_requests(
    db: DatabaseManager,
    *,
    status: str = "PENDING_APPROVAL",
    limit: int = 50,
) -> dict[str, Any]:
    """List refund requests with order/customer metadata and queue status counts."""
    normalized = status.strip().upper()
    filter_map = {
        "PENDING_APPROVAL": "PENDING_HUMAN_APPROVAL",
        "PENDING_HUMAN_APPROVAL": "PENDING_HUMAN_APPROVAL",
        "PENDING": "PENDING_HUMAN_APPROVAL",
        "APPROVED": "APPROVED",
        "AUTO_APPROVED": "AUTO_APPROVED",
        "REJECTED": "REJECTED",
        "ALL": None,
    }
    if normalized not in filter_map:
        raise ValueError(
            "Unsupported status filter. Expected one of: "
            "PENDING_APPROVAL, APPROVED, AUTO_APPROVED, REJECTED, ALL."
        )

    db_status = filter_map[normalized]
    base_sql = """
        SELECT r.refund_id, r.order_id, r.customer_id, r.amount_vnd,
               r.reason_code, r.reason_detail, r.status, r.requires_human_review,
               r.created_at, r.reviewed_at, r.reviewer_note, r.reviewer_id,
               o.item_name, o.category, o.total_vnd AS order_total_vnd,
               o.status AS order_status,
               c.full_name AS customer_name, c.email AS customer_email,
               c.tier AS customer_tier
        FROM refund_requests r
        JOIN orders o ON r.order_id = o.order_id
        JOIN customers c ON r.customer_id = c.customer_id
    """
    order_by = """
        ORDER BY CASE WHEN r.status = 'PENDING_HUMAN_APPROVAL' THEN 0 ELSE 1 END,
                 r.created_at DESC, r.rowid DESC
        LIMIT ?;
    """
    if db_status is not None:
        rows = db.fetch_all(f"{base_sql} WHERE r.status = ? {order_by}", (db_status, limit))
    else:
        rows = db.fetch_all(f"{base_sql} {order_by}", (limit,))

    count_rows = db.fetch_all(
        "SELECT status, COUNT(*) AS cnt FROM refund_requests GROUP BY status;"
    )
    status_counts = {str(r["status"]): int(r["cnt"]) for r in count_rows}
    counts = {
        "pending_approval": status_counts.get("PENDING_HUMAN_APPROVAL", 0),
        "approved": status_counts.get("APPROVED", 0),
        "auto_approved": status_counts.get("AUTO_APPROVED", 0),
        "rejected": status_counts.get("REJECTED", 0),
        "all": sum(status_counts.values()),
    }

    refunds = [
        {
            "refund_id": row["refund_id"],
            "order_id": row["order_id"],
            "customer_id": row["customer_id"],
            "customer_name": row["customer_name"],
            "customer_tier": row["customer_tier"],
            "customer_email_masked": mask_email(row["customer_email"]),
            "item_name": row["item_name"],
            "category": row["category"],
            "order_total_vnd": int(row["order_total_vnd"]),
            "order_status": row["order_status"],
            "amount_vnd": int(row["amount_vnd"]),
            "reason_code": row["reason_code"],
            "reason": row["reason_detail"],
            "status": row["status"],
            "requires_human_review": bool(row["requires_human_review"]),
            "created_at": row["created_at"],
            "reviewed_at": row["reviewed_at"],
            "reviewer_note": row["reviewer_note"],
            "reviewer_id": row["reviewer_id"],
        }
        for row in rows
    ]
    return {"refunds": refunds, "total": len(refunds), "counts": counts}


def decide_refund_request(
    db: DatabaseManager,
    *,
    refund_id: str,
    decision: Literal["APPROVE", "REJECT"],
    reviewer_note: str,
    reviewer_id: str | None = None,
) -> dict[str, Any]:
    """Atomically approve or reject a pending refund request inside BEGIN IMMEDIATE."""
    safe_note, _ = redact_pii(reviewer_note.strip(), redact_email=True)
    raw_reviewer = (reviewer_id or "supervisor-console").strip() or "supervisor-console"
    safe_reviewer_id, _ = redact_pii(raw_reviewer, redact_email=True)
    reviewed_at = _utc_now_iso()
    new_status = "APPROVED" if decision == "APPROVE" else "REJECTED"

    with db.transaction() as conn:
        row = conn.execute(
            """
            SELECT r.refund_id, r.order_id, r.customer_id, r.amount_vnd,
                   r.reason_code, r.reason_detail, r.status, r.created_at,
                   o.status AS order_status, o.item_name, o.category,
                   o.total_vnd AS order_total_vnd,
                   c.full_name AS customer_name, c.email AS customer_email,
                   c.tier AS customer_tier
            FROM refund_requests r
            JOIN orders o ON r.order_id = o.order_id
            JOIN customers c ON r.customer_id = c.customer_id
            WHERE r.refund_id = ?;
            """,
            (refund_id,),
        ).fetchone()

        if row is None:
            raise RefundNotFoundError(f"Refund request '{refund_id}' not found.")

        previous_status = str(row["status"])
        if previous_status != "PENDING_HUMAN_APPROVAL":
            raise RefundAlreadyDecidedError(
                f"Refund request '{refund_id}' already has status '{previous_status}'."
            )

        next_order_status = (
            "REFUNDED"
            if decision == "APPROVE"
            else ("DELIVERED" if row["order_status"] == "REFUND_PENDING" else row["order_status"])
        )

        conn.execute(
            """
            UPDATE refund_requests
            SET status = ?,
                requires_human_review = 0,
                reviewed_at = ?,
                reviewer_note = ?,
                reviewer_id = ?
            WHERE refund_id = ? AND status = 'PENDING_HUMAN_APPROVAL';
            """,
            (new_status, reviewed_at, safe_note, safe_reviewer_id, refund_id),
        )
        conn.execute(
            "UPDATE orders SET status = ? WHERE order_id = ?;",
            (next_order_status, row["order_id"]),
        )
        audit_id = f"AUD-{uuid.uuid4().hex[:10].upper()}"
        conn.execute(
            """
            INSERT INTO refund_audit_logs (
                audit_id, refund_id, order_id, decision, previous_status,
                new_status, reviewer_id, reviewer_note_masked, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                audit_id,
                refund_id,
                row["order_id"],
                decision,
                previous_status,
                new_status,
                safe_reviewer_id,
                safe_note,
                reviewed_at,
            ),
        )

    _refund_logger.info(
        "Supervisor refund decision recorded",
        extra={
            "structured_data": {
                "audit_id": audit_id,
                "refund_id": refund_id,
                "order_id": row["order_id"],
                "decision": decision,
                "previous_status": previous_status,
                "new_status": new_status,
                "order_status": next_order_status,
                "reviewer_id": safe_reviewer_id,
                "reviewer_note_masked": safe_note,
            }
        },
    )

    return {
        "refund_id": row["refund_id"],
        "order_id": row["order_id"],
        "customer_id": row["customer_id"],
        "customer_name": row["customer_name"],
        "customer_tier": row["customer_tier"],
        "customer_email_masked": mask_email(row["customer_email"]),
        "item_name": row["item_name"],
        "category": row["category"],
        "order_total_vnd": int(row["order_total_vnd"]),
        "order_status": next_order_status,
        "amount_vnd": int(row["amount_vnd"]),
        "reason_code": row["reason_code"],
        "reason": row["reason_detail"],
        "decision": decision,
        "status": new_status,
        "requires_human_review": False,
        "created_at": row["created_at"],
        "reviewed_at": reviewed_at,
        "reviewer_note": safe_note,
        "reviewer_id": safe_reviewer_id,
        "audit_id": audit_id,
    }


def list_recent_agent_runs(db: DatabaseManager, *, limit: int = 20) -> list[dict[str, Any]]:
    """Fetch recent agent runs with decoded guardrail flags and trace summaries."""
    rows = db.fetch_all(
        """
        SELECT run_id, session_id, customer_email_masked, user_message_masked,
               intent, resolution_status, refund_id, requires_human_review,
               guardrail_flags_json, steps_taken, latency_ms, total_tokens,
               estimated_cost_usd, created_at
        FROM agent_runs
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?;
        """,
        (limit,),
    )
    results: list[dict[str, Any]] = []
    for row in rows:
        results.append(
            {
                "run_id": row["run_id"],
                "session_id": row["session_id"],
                "customer_email_masked": row["customer_email_masked"],
                "user_message_masked": row["user_message_masked"],
                "intent": row["intent"],
                "resolution_status": row["resolution_status"],
                "refund_id": row["refund_id"],
                "requires_human_review": bool(row["requires_human_review"]),
                "guardrail_flags": json.loads(row["guardrail_flags_json"]),
                "steps_taken": int(row["steps_taken"]),
                "latency_ms": float(row["latency_ms"]),
                "total_tokens": int(row["total_tokens"]),
                "estimated_cost_usd": float(row["estimated_cost_usd"]),
                "created_at": row["created_at"],
            }
        )
    return results


def list_demo_orders(db: DatabaseManager, *, limit: int = 50) -> list[dict[str, Any]]:
    """Fetch orders joined with masked customer info for console inspection."""
    rows = db.fetch_all(
        """
        SELECT o.order_id, o.item_name, o.category, o.total_vnd, o.status,
               o.ordered_at, o.delivered_at, o.tracking_code,
               c.full_name AS customer_name, c.email AS customer_email, c.tier AS customer_tier
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        ORDER BY o.order_id ASC
        LIMIT ?;
        """,
        (limit,),
    )
    return [
        {
            "order_id": row["order_id"],
            "item_name": row["item_name"],
            "category": row["category"],
            "total_vnd": int(row["total_vnd"]),
            "status": row["status"],
            "ordered_at": row["ordered_at"],
            "delivered_at": row["delivered_at"],
            "tracking_code": row["tracking_code"],
            "customer_name": row["customer_name"],
            "customer_email_masked": mask_email(row["customer_email"]),
            "customer_tier": row["customer_tier"],
        }
        for row in rows
    ]
