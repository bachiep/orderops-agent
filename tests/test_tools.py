"""Unit tests for the 5 Pydantic v2 tools and ToolRegistry validation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from orderops.config import Settings
from orderops.db import DatabaseManager
from orderops.tools import OrderOpsToolset, ToolRegistry, build_default_registry
from scripts.seed_demo_db import seed_demo_database


def _setup_registry(tmp_path: Path | None = None) -> tuple[DatabaseManager, ToolRegistry]:
    fixed_now = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)
    db_target = str(tmp_path / "tools.db") if tmp_path else ":memory:"
    db = DatabaseManager(db_target)
    seed_demo_database(db, now=fixed_now)
    toolset = OrderOpsToolset(db=db, settings=Settings(), now_provider=lambda: fixed_now)
    return db, build_default_registry(toolset)


def test_registry_exports_all_five_openai_tool_schemas() -> None:
    db, registry = _setup_registry()
    try:
        assert registry.list_names() == [
            "get_order_details",
            "list_customer_orders",
            "lookup_return_policy",
            "evaluate_refund_eligibility",
            "create_refund_request",
        ]
        schemas = registry.get_openai_tools_schema()
        assert len(schemas) == 5
        assert all(s["type"] == "function" and "parameters" in s["function"] for s in schemas)
    finally:
        db.close()


def test_read_only_order_and_policy_tools() -> None:
    db, registry = _setup_registry()
    try:
        order_res = registry.execute("get_order_details", {"order_id": "ord-1001"})
        assert order_res.ok is True
        assert order_res.output["found"] is True
        assert order_res.output["total_vnd"] == 320_000
        assert order_res.output["days_since_delivery"] == 3

        list_res = registry.execute(
            "list_customer_orders",
            {"customer_email": "minh.anh@example.com", "limit": 5},
        )
        assert list_res.ok is True
        assert list_res.output["found_customer"] is True
        assert list_res.output["total_returned"] == 2
        assert list_res.output["customer_email_masked"] == "mi***@example.com"

        policy_res = registry.execute(
            "lookup_return_policy",
            {"category": "perishable_food", "issue_type": "defective"},
        )
        assert policy_res.ok is True
        assert policy_res.output["found"] is True
        assert policy_res.output["is_returnable"] is False
    finally:
        db.close()


def test_evaluate_refund_eligibility_boundary_and_rejections() -> None:
    db, registry = _setup_registry()
    try:
        # 14-day boundary (ORD-1009) is still eligible
        ok_14d = registry.execute(
            "evaluate_refund_eligibility",
            {"order_id": "ORD-1009", "reason_code": "changed_mind"},
        )
        assert ok_14d.output["eligible"] is True
        assert ok_14d.output["days_since_delivery"] == 14

        # 15-day boundary (ORD-1012) is expired
        exp_15d = registry.execute(
            "evaluate_refund_eligibility",
            {"order_id": "ORD-1012", "reason_code": "defective"},
        )
        assert exp_15d.output["eligible"] is False
        assert "EXPIRED_RETURN_WINDOW" in exp_15d.output["reason_codes"]

        # Undelivered order (ORD-1004)
        transit = registry.execute(
            "evaluate_refund_eligibility",
            {"order_id": "ORD-1004", "reason_code": "defective"},
        )
        assert transit.output["eligible"] is False
        assert "ORDER_STATUS_IN_TRANSIT" in transit.output["reason_codes"]

        # Cosmetics changed_mind vs damaged_in_transit (ORD-1010)
        cosm_mind = registry.execute(
            "evaluate_refund_eligibility",
            {"order_id": "ORD-1010", "reason_code": "changed_mind"},
        )
        assert cosm_mind.output["eligible"] is False
        assert "NON_RETURNABLE_CATEGORY" in cosm_mind.output["reason_codes"]

        cosm_dmg = registry.execute(
            "evaluate_refund_eligibility",
            {"order_id": "ORD-1010", "reason_code": "damaged_in_transit"},
        )
        assert cosm_dmg.output["eligible"] is True
        assert cosm_dmg.output["requires_human_review_if_full"] is True
    finally:
        db.close()


def test_create_refund_request_auto_approved_and_high_value_escalation(tmp_path: Path) -> None:
    db, registry = _setup_registry(tmp_path)
    try:
        # Exact 500,000 VND boundary -> AUTO_APPROVED
        auto_res = registry.execute(
            "create_refund_request",
            {
                "order_id": "ORD-1011",
                "amount_vnd": 500_000,
                "reason": "Lid seal is broken out of the box",
                "reason_code": "defective",
            },
        )
        assert auto_res.ok is True
        assert auto_res.output["created"] is True
        assert auto_res.output["status"] == "AUTO_APPROVED"
        assert auto_res.output["requires_human_review"] is False

        # > 500,000 VND -> PENDING_HUMAN_APPROVAL
        high_res = registry.execute(
            "create_refund_request",
            {
                "order_id": "ORD-1002",
                "amount_vnd": 1_850_000,
                "reason": "Left earcup emits static noise",
                "reason_code": "defective",
            },
        )
        assert high_res.ok is True
        assert high_res.output["created"] is True
        assert high_res.output["status"] == "PENDING_HUMAN_APPROVAL"
        assert high_res.output["requires_human_review"] is True

        # Second attempt on ORD-1011 must be rejected as already refunded
        dup_res = registry.execute(
            "create_refund_request",
            {
                "order_id": "ORD-1011",
                "amount_vnd": 100_000,
                "reason": "Duplicate refund attempt",
                "reason_code": "defective",
            },
        )
        assert dup_res.output["created"] is False
        assert dup_res.output["status"] == "REJECTED"
    finally:
        db.close()


def test_registry_handles_invalid_args_and_unknown_tools_gracefully() -> None:
    db, registry = _setup_registry()
    try:
        bad_amount = registry.execute(
            "create_refund_request",
            {"order_id": "ORD-1001", "amount_vnd": -50, "reason": "bad"},
        )
        assert bad_amount.ok is False
        assert bad_amount.error is not None
        assert bad_amount.error.startswith("VALIDATION_ERROR:")

        unknown = registry.execute("drop_database", {})
        assert unknown.ok is False
        assert unknown.error == "UNKNOWN_TOOL:drop_database"
    finally:
        db.close()
