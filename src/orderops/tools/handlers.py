"""Business tool implementations backed by embedded SQLite and FinancialPolicyGate."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from orderops.config import Settings
from orderops.db import DatabaseManager
from orderops.guardrails.pii import mask_email
from orderops.guardrails.policy_gate import FinancialPolicyGate, _days_elapsed
from orderops.tools.schemas import (
    CreateRefundOutput,
    CreateRefundRequestInput,
    CustomerOrdersOutput,
    EvaluateRefundEligibilityInput,
    GetOrderDetailsInput,
    ListCustomerOrdersInput,
    LookupReturnPolicyInput,
    OrderDetailsOutput,
    OrderSummary,
    RefundEligibilityOutput,
    ReturnPolicyOutput,
)


class OrderOpsToolset:
    """Implements the 5 OrderOps read and mutation tools against SQLite."""

    def __init__(
        self,
        db: DatabaseManager,
        settings: Settings | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or Settings()
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        self.policy_gate = FinancialPolicyGate(
            db=self.db,
            settings=self.settings,
            now_provider=self._now_provider,
        )

    def _now(self) -> datetime:
        return self._now_provider().astimezone(UTC)

    def get_order_details(self, payload: GetOrderDetailsInput) -> OrderDetailsOutput:
        """Fetch full details for a single order including active refund state."""
        row = self.db.fetch_one(
            """
            SELECT o.*, c.full_name AS customer_name, c.tier AS customer_tier
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.order_id = ?;
            """,
            (payload.order_id,),
        )
        if row is None:
            return OrderDetailsOutput(
                found=False,
                order_id=payload.order_id,
                error=f"Order {payload.order_id} not found.",
            )

        refund = self.db.fetch_one(
            """
            SELECT refund_id, status
            FROM refund_requests
            WHERE order_id = ?
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            (payload.order_id,),
        )
        days_since = _days_elapsed(row["delivered_at"], self._now())
        return OrderDetailsOutput(
            found=True,
            order_id=row["order_id"],
            customer_id=row["customer_id"],
            customer_name=row["customer_name"],
            customer_tier=row["customer_tier"],
            item_name=row["item_name"],
            category=row["category"],
            quantity=int(row["quantity"]),
            total_vnd=int(row["total_vnd"]),
            status=row["status"],
            ordered_at=row["ordered_at"],
            delivered_at=row["delivered_at"],
            days_since_delivery=days_since,
            tracking_code=row["tracking_code"],
            existing_refund_id=refund["refund_id"] if refund else None,
            existing_refund_status=refund["status"] if refund else None,
        )

    def list_customer_orders(self, payload: ListCustomerOrdersInput) -> CustomerOrdersOutput:
        """List recent orders belonging to a customer email (email masked in output)."""
        masked = mask_email(payload.customer_email)
        customer = self.db.fetch_one(
            "SELECT customer_id, full_name FROM customers WHERE LOWER(email) = ?;",
            (payload.customer_email,),
        )
        if customer is None:
            return CustomerOrdersOutput(
                found_customer=False,
                customer_email_masked=masked,
                error="No customer found matching the provided email.",
            )

        rows = self.db.fetch_all(
            """
            SELECT order_id, item_name, category, total_vnd, status, ordered_at, delivered_at
            FROM orders
            WHERE customer_id = ?
            ORDER BY ordered_at DESC
            LIMIT ?;
            """,
            (customer["customer_id"], payload.limit),
        )
        summaries = [
            OrderSummary(
                order_id=r["order_id"],
                item_name=r["item_name"],
                category=r["category"],
                total_vnd=int(r["total_vnd"]),
                status=r["status"],
                ordered_at=r["ordered_at"],
                delivered_at=r["delivered_at"],
            )
            for r in rows
        ]
        return CustomerOrdersOutput(
            found_customer=True,
            customer_email_masked=masked,
            customer_name=customer["full_name"],
            orders=summaries,
            total_returned=len(summaries),
        )

    def lookup_return_policy(self, payload: LookupReturnPolicyInput) -> ReturnPolicyOutput:
        """Look up return eligibility rules via the deterministic policy gate."""
        return self.policy_gate.lookup_policy(payload)

    def evaluate_refund_eligibility(
        self,
        payload: EvaluateRefundEligibilityInput,
    ) -> RefundEligibilityOutput:
        """Deterministically evaluate refund eligibility against DB state and policy rules."""
        return self.policy_gate.evaluate_eligibility(payload)

    def create_refund_request(
        self,
        payload: CreateRefundRequestInput,
        *,
        session_id: str | None = None,
        idempotency_key: str | None = None,
        idempotency_endpoint: str = "service:create_refund_request",
    ) -> CreateRefundOutput:
        """Create a refund request only after enforcing deterministic financial policy gates."""
        return self.policy_gate.execute_refund(
            payload,
            session_id=session_id,
            idempotency_key=idempotency_key,
            idempotency_endpoint=idempotency_endpoint,
        )
