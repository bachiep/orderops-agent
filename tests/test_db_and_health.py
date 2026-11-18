"""Smoke tests for SQLite schema, idempotent seeding, and FastAPI /health."""

from __future__ import annotations

import json
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
        assert first_counts["orders"] == 13
        assert first_counts["refund_requests"] == 2

        pending_seed = db.fetch_one(
            "SELECT * FROM refund_requests WHERE refund_id = 'RFD-SEED-002';"
        )
        assert pending_seed is not None
        assert pending_seed["order_id"] == "ORD-1013"
        assert int(pending_seed["amount_vnd"]) == 650_000
        assert pending_seed["status"] == "PENDING_HUMAN_APPROVAL"


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
        assert payload["table_counts"]["orders"] == 13
        assert payload["table_counts"]["refund_requests"] == 2
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


def test_v1_refund_requests_schema_migrates_safely_on_init_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_v1.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT NOT NULL,
            tier TEXT NOT NULL DEFAULT 'STANDARD',
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
            item_name TEXT NOT NULL,
            category TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            total_vnd INTEGER NOT NULL,
            status TEXT NOT NULL,
            ordered_at TEXT NOT NULL,
            delivered_at TEXT,
            tracking_code TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE refund_requests (
            refund_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE RESTRICT,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
            amount_vnd INTEGER NOT NULL,
            reason_code TEXT NOT NULL,
            reason_detail TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('AUTO_APPROVED', 'PENDING_HUMAN_APPROVAL', 'REJECTED')
            ),
            requires_human_review INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        INSERT INTO customers VALUES (
            'C1', 'Test User', 'c1@example.com', '0901234567',
            'VIP', '2026-11-01T00:00:00+00:00'
        );
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'O1', 'C1', 'Headphones', 'electronics', 1, 650000,
            'REFUND_PENDING', '2026-11-01T00:00:00+00:00',
            '2026-11-02T00:00:00+00:00', 'TRK-1'
        );
        """
    )
    conn.execute(
        """
        INSERT INTO refund_requests VALUES (
            'R1', 'O1', 'C1', 650000, 'defective', 'Static audio',
            'PENDING_HUMAN_APPROVAL', 1, '2026-11-03T00:00:00+00:00'
        );
        """
    )
    conn.commit()
    conn.close()

    with DatabaseManager(db_path) as db:
        db.init_schema()
        db.execute(
            """
            UPDATE refund_requests
            SET status = 'APPROVED', reviewed_at = '2026-11-04T00:00:00+00:00',
                reviewer_note = 'Verified', reviewer_id = 'sup-1'
            WHERE refund_id = 'R1';
            """
        )
        migrated = db.fetch_one(
            "SELECT status, reviewer_id FROM refund_requests WHERE refund_id = 'R1';"
        )
        assert migrated == {"status": "APPROVED", "reviewer_id": "sup-1"}


def test_colab_qlora_notebook_is_valid_nbformat4() -> None:
    nb_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "colab_qlora_agent_tool_calling_gpu.ipynb"
    )
    assert nb_path.is_file()
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert len(payload["cells"]) >= 10
    full_source = "\n".join("".join(cell["source"]) for cell in payload["cells"])
    assert "Qwen/Qwen2.5-3B-Instruct" in full_source
    assert "BitsAndBytesConfig" in full_source
    assert "LoraConfig" in full_source
    assert "SFTTrainer" in full_source
    assert "TriageResolutionSchema" in full_source
