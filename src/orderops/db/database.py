"""Thread-safe embedded SQLite connection and transaction manager."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from orderops.db.schema import SCHEMA_STATEMENTS


class IdempotencyConflictError(RuntimeError):
    """Raised when an Idempotency-Key is reused with a different request payload."""


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
        """Create all tables and indexes if they do not exist and migrate v1/v2 tables safely."""
        with self._lock:
            for stmt in SCHEMA_STATEMENTS:
                self._conn.execute(stmt)
            self._migrate_refund_requests_if_needed()
            self._migrate_refund_requests_v21_columns()

    def _migrate_refund_requests_if_needed(self) -> None:
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'refund_requests';"
        ).fetchone()
        if row is None:
            return
        ddl = str(row["sql"] or "")
        if "'APPROVED'" in ddl and "reviewed_at" in ddl:
            return

        self._conn.execute("PRAGMA foreign_keys = OFF;")
        try:
            self._conn.execute("BEGIN IMMEDIATE;")
            self._conn.execute(
                """
                CREATE TABLE refund_requests_v2 (
                    refund_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE RESTRICT,
                    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
                    session_id TEXT,
                    amount_vnd INTEGER NOT NULL CHECK (amount_vnd > 0),
                    reason_code TEXT NOT NULL,
                    reason_detail TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'AUTO_APPROVED', 'PENDING_HUMAN_APPROVAL', 'APPROVED', 'REJECTED'
                        )
                    ),
                    requires_human_review INTEGER NOT NULL CHECK (requires_human_review IN (0, 1)),
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewer_note TEXT,
                    reviewer_id TEXT
                );
                """
            )
            self._conn.execute(
                """
                INSERT INTO refund_requests_v2 (
                    refund_id, order_id, customer_id, amount_vnd,
                    reason_code, reason_detail, status, requires_human_review, created_at
                )
                SELECT refund_id, order_id, customer_id, amount_vnd,
                       reason_code, reason_detail, status, requires_human_review, created_at
                FROM refund_requests;
                """
            )
            self._conn.execute("DROP TABLE refund_requests;")
            self._conn.execute("ALTER TABLE refund_requests_v2 RENAME TO refund_requests;")
            self._conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_refund_order
                ON refund_requests(order_id)
                WHERE status IN ('AUTO_APPROVED', 'PENDING_HUMAN_APPROVAL', 'APPROVED');
                """
            )
            self._conn.execute("COMMIT;")
        except Exception:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK;")
            raise
        finally:
            self._conn.execute("PRAGMA foreign_keys = ON;")

    def _migrate_refund_requests_v21_columns(self) -> None:
        cols = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(refund_requests);").fetchall()
        }
        if "session_id" not in cols:
            self._conn.execute("BEGIN IMMEDIATE;")
            try:
                self._conn.execute("ALTER TABLE refund_requests ADD COLUMN session_id TEXT;")
                self._conn.execute("COMMIT;")
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK;")
                raise
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_refund_requests_session_id
            ON refund_requests(session_id);
            """
        )

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
        tables = (
            "customers",
            "return_policies",
            "orders",
            "refund_requests",
            "agent_runs",
            "session_messages",
            "refund_audit_logs",
            "idempotency_records",
        )
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
