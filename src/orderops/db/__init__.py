"""Embedded SQLite persistence layer for OrderOps Agent."""

from __future__ import annotations

from typing import Any

from orderops.db.database import DatabaseManager
from orderops.db.schema import SCHEMA_STATEMENTS

__all__ = [
    "DatabaseManager",
    "SCHEMA_STATEMENTS",
    "list_demo_orders",
    "list_recent_agent_runs",
    "persist_agent_run",
]


def __getattr__(name: str) -> Any:
    if name in ("list_demo_orders", "list_recent_agent_runs", "persist_agent_run"):
        from orderops.db import audit

        return getattr(audit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
