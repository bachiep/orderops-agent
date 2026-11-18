"""Embedded SQLite persistence layer for OrderOps Agent."""

from __future__ import annotations

from typing import Any

from orderops.db.database import DatabaseManager
from orderops.db.schema import SCHEMA_STATEMENTS

_AUDIT_EXPORTS = frozenset(
    {
        "RefundAlreadyDecidedError",
        "RefundNotFoundError",
        "decide_refund_request",
        "get_session_history_for_llm",
        "get_session_messages",
        "list_demo_orders",
        "list_recent_agent_runs",
        "list_refund_requests",
        "list_sessions",
        "persist_agent_run",
    }
)

__all__ = [
    "DatabaseManager",
    "SCHEMA_STATEMENTS",
    *sorted(_AUDIT_EXPORTS),
]


def __getattr__(name: str) -> Any:
    if name in _AUDIT_EXPORTS:
        from orderops.db import audit

        return getattr(audit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
