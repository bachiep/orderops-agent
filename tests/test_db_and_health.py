"""Smoke tests for SQLite schema, idempotent seeding, and FastAPI /health."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from orderops.api.main import create_app
from orderops.config import Settings
from orderops.db import DatabaseManager
from scripts.seed_demo_db import seed_demo_database


def test_seed_demo_database_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "orderops_test.db"
    with DatabaseManager(db_path) as db:
        first_counts = seed_demo_database(db)
        second_counts = seed_demo_database(db)

        assert first_counts == second_counts
        assert first_counts["customers"] == 5
        assert first_counts["return_policies"] == 11
        assert first_counts["orders"] == 12
        assert first_counts["refund_requests"] == 1


def test_sqlite_enforces_foreign_keys_and_unique_active_refund(tmp_path: Path) -> None:
    db_path = tmp_path / "constraints.db"
    with DatabaseManager(db_path) as db:
        seed_demo_database(db)

        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """
                INSERT INTO orders (
                    order_id, customer_id, item_name, category, quantity,
                    total_vnd, status, ordered_at, delivered_at, tracking_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "ORD-INVALID",
                    "CUST-MISSING",
                    "Ghost Item",
                    "electronics",
                    1,
                    100_000,
                    "DELIVERED",
                    "2026-11-01T00:00:00+00:00",
                    "2026-11-03T00:00:00+00:00",
                    "TRK-0",
                ),
            )

        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """
                INSERT INTO refund_requests (
                    refund_id, order_id, customer_id, amount_vnd,
                    reason_code, reason_detail, status, requires_human_review, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "RFD-DUP",
                    "ORD-1007",
                    "CUST-004",
                    480_000,
                    "wrong_item",
                    "Duplicate active refund test",
                    "PENDING_HUMAN_APPROVAL",
                    1,
                    "2026-11-10T00:00:00+00:00",
                ),
            )


async def test_health_endpoint_returns_table_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "api_health.db"
    with DatabaseManager(db_path) as db:
        seed_demo_database(db)
        settings = Settings(database_path=str(db_path), llm_provider="local")
        app = create_app(settings=settings, db=db)

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/health")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["llm_provider"] == "local"
        assert payload["table_counts"]["orders"] == 12
        assert payload["table_counts"]["customers"] == 5


@pytest.mark.parametrize(
    "module_name",
    [
        "orderops.db",
        "orderops.db.audit",
        "orderops.tools",
        "orderops.guardrails",
        "orderops.agent",
        "orderops.api.main",
        "orderops.ui.app",
    ],
)
def test_cold_module_imports_have_no_circular_dependencies(module_name: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Failed importing {module_name}: {proc.stderr}"


def test_seed_demo_db_cli_runs_cleanly_in_fresh_subprocess(tmp_path: Path) -> None:
    db_path = tmp_path / "cli_seed.db"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.seed_demo_db", "--db-path", str(db_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Seeded SQLite database" in proc.stdout
    assert db_path.exists()
