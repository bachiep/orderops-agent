"""Thread-safe embedded SQLite connection and transaction manager."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from orderops.db.schema import SCHEMA_STATEMENTS


class DatabaseManager:
    """Manages SQLite schema initialization, parameterized queries, and atomic transactions."""

    def __init__(self, db_path: str | Path = "./data/orderops.db") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = self._open_connection()

    def _open_connection(self) -> sqlite3.Connection:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if self.db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def init_schema(self) -> None:
        """Create all tables and indexes if they do not exist."""
        with self._lock:
            for stmt in SCHEMA_STATEMENTS:
                self._conn.execute(stmt)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a unit of work inside an immediate SQLite transaction."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE;")
            try:
                yield self._conn
                self._conn.execute("COMMIT;")
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK;")
                raise

    def fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Execute a SELECT returning at most one row as a dict."""
        with self._lock:
            cursor = self._conn.execute(sql, params)
            row = cursor.fetchone()
            return dict(row) if row is not None else None

    def fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a SELECT returning all matching rows as dicts."""
        with self._lock:
            cursor = self._conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        """Execute a single write statement inside a transaction and return affected rows."""
        with self.transaction() as conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount

    def get_table_counts(self) -> dict[str, int]:
        """Return row counts across core domain tables."""
        tables = ("customers", "return_policies", "orders", "refund_requests", "agent_runs")
        counts: dict[str, int] = {}
        with self._lock:
            for table in tables:
                row = self._conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
                counts[table] = int(row["cnt"]) if row is not None else 0
        return counts

    def close(self) -> None:
        """Close the underlying SQLite connection cleanly."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> DatabaseManager:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
