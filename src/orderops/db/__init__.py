"""Embedded SQLite persistence layer for OrderOps Agent."""

from __future__ import annotations

from typing import Any

from orderops.db.database import DatabaseManager, IdempotencyConflictError
from orderops.db.schema import SCHEMA_STATEMENTS

_AUDIT_EXPORTS = frozenset(
    {
        "RefundAlreadyDecidedError",
        "RefundNotFoundError",
        "decide_refund_request",
        "get_refund_request_detail",
        "get_session_history_for_llm",
        "get_session_messages",
        "hash_canonical_payload",
        "list_demo_orders",
        "list_recent_agent_runs",
        "list_refund_requests",
        "list_sessions",
        "lookup_idempotency_record",
        "persist_agent_run",
        "save_idempotency_record",
    }
)

__all__ = [
    "DatabaseManager",
    "IdempotencyConflictError",
    "SCHEMA_STATEMENTS",
    *sorted(_AUDIT_EXPORTS),
]


def __getattr__(name: str) -> Any:
    if name in _AUDIT_EXPORTS:
        from orderops.db import audit

        return getattr(audit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
