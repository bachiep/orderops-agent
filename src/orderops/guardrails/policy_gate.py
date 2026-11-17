"""Layer 2 Deterministic Financial Policy Gate backed by SQLite state."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from orderops.config import Settings
from orderops.db import DatabaseManager
from orderops.tools.schemas import (
    CreateRefundOutput,
    CreateRefundRequestInput,
    EvaluateRefundEligibilityInput,
    LookupReturnPolicyInput,
    RefundEligibilityOutput,
    ReturnPolicyOutput,
)


def _parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _days_elapsed(delivered_at: str | None, ref_now: datetime) -> int | None:
    if not delivered_at:
        return None
    delivered_dt = _parse_iso(delivered_at)
    seconds = (ref_now - delivered_dt).total_seconds()
    return max(0, int(seconds // 86400))


class FinancialPolicyGate:
    """Hard deterministic business and financial policy invariants for refunds."""

    def __init__(
        self,
        db: DatabaseManager,
        settings: Settings | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or Settings()
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    def _now(self) -> datetime:
        return self._now_provider().astimezone(UTC)

    def find_policy_row(self, category: str, issue_type: str) -> dict[str, Any] | None:
        """Fetch exact category+issue policy or fallback to category+any."""
        exact = self.db.fetch_one(
            "SELECT * FROM return_policies WHERE category = ? AND issue_type = ?;",
            (category, issue_type),
        )
        if exact is not None:
            return exact
        return self.db.fetch_one(
            "SELECT * FROM return_policies WHERE category = ? AND issue_type = 'any';",
            (category,),
        )

    def lookup_policy(self, payload: LookupReturnPolicyInput) -> ReturnPolicyOutput:
        """Look up return policy metadata for a product category and issue type."""
        row = self.find_policy_row(payload.category, payload.issue_type)
        if row is None:
            return ReturnPolicyOutput(
                found=False,
                category=payload.category,
                issue_type=payload.issue_type,
                is_returnable=False,
                return_window_days=0,
                restocking_fee_pct=0,
                requires_photo_evidence=False,
                notes=f"No return policy found for category '{payload.category}'.",
            )
        return ReturnPolicyOutput(
            found=True,
            policy_id=row["policy_id"],
            category=row["category"],
            issue_type=row["issue_type"],
            is_returnable=bool(row["is_returnable"]),
            return_window_days=int(row["return_window_days"]),
            restocking_fee_pct=int(row["restocking_fee_pct"]),
            requires_photo_evidence=bool(row["requires_photo_evidence"]),
            notes=row["notes"],
        )

    def evaluate_eligibility(
        self,
        payload: EvaluateRefundEligibilityInput,
    ) -> RefundEligibilityOutput:
        """Deterministically evaluate refund eligibility from DB order and policy state."""
        order = self.db.fetch_one("SELECT * FROM orders WHERE order_id = ?;", (payload.order_id,))
        if order is None:
            return RefundEligibilityOutput(
                order_id=payload.order_id,
                eligible=False,
                reason_codes=["ORDER_NOT_FOUND"],
                policy_notes=f"Order {payload.order_id} does not exist.",
            )

        active_refund = self.db.fetch_one(
            """
            SELECT refund_id, status FROM refund_requests
            WHERE order_id = ? AND status IN ('AUTO_APPROVED', 'PENDING_HUMAN_APPROVAL')
            LIMIT 1;
            """,
            (payload.order_id,),
        )

        policy = self.find_policy_row(order["category"], payload.reason_code)
        days_since = _days_elapsed(order["delivered_at"], self._now())
        max_window = (
            int(policy["return_window_days"])
            if policy
            else self.settings.return_window_days
        )
        effective_window = min(max_window, self.settings.return_window_days)
        restocking_pct = int(policy["restocking_fee_pct"]) if policy else 0
        total_vnd = int(order["total_vnd"])
        max_refundable = total_vnd * (100 - restocking_pct) // 100

        reason_codes: list[str] = []
        if order["status"] in ("REFUNDED", "REFUND_PENDING") or active_refund is not None:
            reason_codes.append("ALREADY_REFUNDED_OR_PENDING")
        elif order["status"] != "DELIVERED":
            reason_codes.append(f"ORDER_STATUS_{order['status']}")

        if policy is None or not bool(policy["is_returnable"]):
            reason_codes.append("NON_RETURNABLE_CATEGORY")

        if order["status"] == "DELIVERED":
            if days_since is None:
                reason_codes.append("MISSING_DELIVERY_TIMESTAMP")
            elif days_since > effective_window:
                reason_codes.append("EXPIRED_RETURN_WINDOW")

        eligible = len(reason_codes) == 0
        if eligible:
            reason_codes = ["ELIGIBLE"]

        requires_review = eligible and (max_refundable > self.settings.auto_refund_limit_vnd)
        notes = (
            policy["notes"]
            if policy is not None
            else f"No returnable policy found for category '{order['category']}'."
        )
        return RefundEligibilityOutput(
            order_id=payload.order_id,
            eligible=eligible,
            reason_codes=reason_codes,
            order_status=order["status"],
            category=order["category"],
            days_since_delivery=days_since,
            return_window_days=effective_window if policy and policy["is_returnable"] else 0,
            max_refundable_vnd=max_refundable if eligible else 0,
            restocking_fee_pct=restocking_pct,
            requires_human_review_if_full=requires_review,
            policy_notes=notes,
        )

    def execute_refund(self, payload: CreateRefundRequestInput) -> CreateRefundOutput:
        """Unconditionally enforce financial policy invariants before mutating SQLite."""
        eligibility = self.evaluate_eligibility(
            EvaluateRefundEligibilityInput(
                order_id=payload.order_id,
                reason_code=payload.reason_code,
            )
        )
        if not eligibility.eligible:
            violation = ", ".join(eligibility.reason_codes)
            return CreateRefundOutput(
                created=False,
                refund_id=None,
                order_id=payload.order_id,
                amount_vnd=payload.amount_vnd,
                status="REJECTED",
                requires_human_review=False,
                policy_violation=violation,
                message=f"Refund rejected by policy gate ({violation}): {eligibility.policy_notes}",
            )

        if payload.amount_vnd > eligibility.max_refundable_vnd:
            return CreateRefundOutput(
                created=False,
                refund_id=None,
                order_id=payload.order_id,
                amount_vnd=payload.amount_vnd,
                status="REJECTED",
                requires_human_review=False,
                policy_violation="AMOUNT_EXCEEDS_MAX_REFUNDABLE",
                message=(
                    f"Requested amount {payload.amount_vnd:,} VND exceeds maximum refundable "
                    f"{eligibility.max_refundable_vnd:,} VND."
                ),
            )

        requires_human = payload.amount_vnd > self.settings.auto_refund_limit_vnd
        refund_status = "PENDING_HUMAN_APPROVAL" if requires_human else "AUTO_APPROVED"
        next_order_status = "REFUND_PENDING" if requires_human else "REFUNDED"
        refund_id = f"RFD-{uuid.uuid4().hex[:8].upper()}"
        created_at = self._now().replace(microsecond=0).isoformat()

        with self.db.transaction() as conn:
            order_row = conn.execute(
                "SELECT customer_id, status FROM orders WHERE order_id = ?;",
                (payload.order_id,),
            ).fetchone()
            if order_row is None or order_row["status"] != "DELIVERED":
                return CreateRefundOutput(
                    created=False,
                    refund_id=None,
                    order_id=payload.order_id,
                    amount_vnd=payload.amount_vnd,
                    status="REJECTED",
                    requires_human_review=False,
                    policy_violation="CONCURRENT_STATE_CHANGE",
                    message="Order is no longer eligible for refund.",
                )

            conn.execute(
                """
                INSERT INTO refund_requests (
                    refund_id, order_id, customer_id, amount_vnd,
                    reason_code, reason_detail, status, requires_human_review, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    refund_id,
                    payload.order_id,
                    order_row["customer_id"],
                    payload.amount_vnd,
                    payload.reason_code,
                    payload.reason,
                    refund_status,
                    1 if requires_human else 0,
                    created_at,
                ),
            )
            conn.execute(
                "UPDATE orders SET status = ? WHERE order_id = ?;",
                (next_order_status, payload.order_id),
            )

        return CreateRefundOutput(
            created=True,
            refund_id=refund_id,
            order_id=payload.order_id,
            amount_vnd=payload.amount_vnd,
            status=refund_status,
            requires_human_review=requires_human,
            policy_violation=None,
            message=(
                f"Refund {refund_id} created for {payload.amount_vnd:,} VND with status "
                f"{refund_status}."
            ),
        )
