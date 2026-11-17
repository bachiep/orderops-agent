"""SQLite audit persistence helpers with mandatory PII redaction."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from orderops.agent.schemas import AgentRunRequest, TriageResolution
from orderops.db.database import DatabaseManager
from orderops.guardrails.pii import mask_email, redact_pii


def persist_agent_run(
    db: DatabaseManager,
    request: AgentRunRequest,
    resolution: TriageResolution,
) -> None:
    """Persist a completed agent run into agent_runs with all PII masked."""
    masked_message, _ = redact_pii(request.message, redact_email=True)
    masked_email = mask_email(request.customer_email) if request.customer_email else None
    raw_trace = json.dumps(
        [step.model_dump(mode="json") for step in resolution.tool_trace],
        ensure_ascii=False,
    )
    masked_trace, _ = redact_pii(raw_trace, redact_email=True)
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    db.execute(
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
            request.session_id,
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
