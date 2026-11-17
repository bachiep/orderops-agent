"""Pydantic v2 input and output schemas for the 5 OrderOps tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ReasonCode = Literal["defective", "wrong_item", "damaged_in_transit", "changed_mind"]


def _normalize_id(value: str) -> str:
    cleaned = value.strip().upper()
    if not cleaned:
        raise ValueError("Identifier cannot be blank.")
    return cleaned


class GetOrderDetailsInput(BaseModel):
    """Input schema for fetching a single order by order_id."""

    order_id: str = Field(..., min_length=3, max_length=40, description="Order ID, e.g. ORD-1001")

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, v: str) -> str:
        return _normalize_id(v)


class OrderDetailsOutput(BaseModel):
    """Structured details for a single customer order."""

    found: bool
    order_id: str
    customer_id: str | None = None
    customer_name: str | None = None
    customer_tier: str | None = None
    item_name: str | None = None
    category: str | None = None
    quantity: int | None = None
    total_vnd: int | None = None
    status: str | None = None
    ordered_at: str | None = None
    delivered_at: str | None = None
    days_since_delivery: int | None = None
    tracking_code: str | None = None
    existing_refund_id: str | None = None
    existing_refund_status: str | None = None
    error: str | None = None


class ListCustomerOrdersInput(BaseModel):
    """Input schema for listing recent orders belonging to a customer email."""

    customer_email: str = Field(
        ...,
        min_length=5,
        max_length=120,
        description="Customer email address, e.g. minh.anh@example.com",
    )
    limit: int = Field(default=10, ge=1, le=50, description="Maximum orders to return")

    @field_validator("customer_email")
    @classmethod
    def validate_email_shape(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if "@" not in cleaned or "." not in cleaned.split("@", 1)[-1]:
            raise ValueError("customer_email must be a valid email address.")
        return cleaned


class OrderSummary(BaseModel):
    """Compact order record used in customer order listings."""

    order_id: str
    item_name: str
    category: str
    total_vnd: int
    status: str
    ordered_at: str
    delivered_at: str | None = None


class CustomerOrdersOutput(BaseModel):
    """Structured listing of orders for a customer."""

    found_customer: bool
    customer_email_masked: str
    customer_name: str | None = None
    orders: list[OrderSummary] = Field(default_factory=list)
    total_returned: int = 0
    error: str | None = None


class LookupReturnPolicyInput(BaseModel):
    """Input schema for looking up category return policy rules."""

    category: str = Field(
        ...,
        min_length=2,
        max_length=60,
        description="Product category such as electronics, apparel, perishable_food, cosmetics",
    )
    issue_type: str = Field(
        default="defective",
        min_length=2,
        max_length=40,
        description="Issue reason: defective, wrong_item, damaged_in_transit, changed_mind",
    )

    @field_validator("category", "issue_type")
    @classmethod
    def normalize_slug(cls, v: str) -> str:
        cleaned = v.strip().lower().replace(" ", "_")
        if not cleaned:
            raise ValueError("Value cannot be blank.")
        return cleaned


class ReturnPolicyOutput(BaseModel):
    """Structured return policy record for a category and issue type."""

    found: bool
    policy_id: str | None = None
    category: str
    issue_type: str
    is_returnable: bool = False
    return_window_days: int = 0
    restocking_fee_pct: int = 0
    requires_photo_evidence: bool = False
    notes: str


class EvaluateRefundEligibilityInput(BaseModel):
    """Input schema for deterministic refund eligibility verification."""

    order_id: str = Field(..., min_length=3, max_length=40)
    reason_code: ReasonCode = Field(default="defective")

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, v: str) -> str:
        return _normalize_id(v)


class RefundEligibilityOutput(BaseModel):
    """Deterministic eligibility assessment before initiating a refund."""

    order_id: str
    eligible: bool
    reason_codes: list[str]
    order_status: str | None = None
    category: str | None = None
    days_since_delivery: int | None = None
    return_window_days: int | None = None
    max_refundable_vnd: int = 0
    restocking_fee_pct: int = 0
    requires_human_review_if_full: bool = False
    policy_notes: str


class CreateRefundRequestInput(BaseModel):
    """Input schema for creating a refund request (mutating tool)."""

    order_id: str = Field(..., min_length=3, max_length=40)
    amount_vnd: int = Field(..., gt=0, description="Requested refund amount in VND")
    reason: str = Field(..., min_length=3, max_length=500, description="Customer refund reason")
    reason_code: ReasonCode = Field(default="defective")

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, v: str) -> str:
        return _normalize_id(v)

    @field_validator("reason")
    @classmethod
    def clean_reason(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 3:
            raise ValueError("reason must contain at least 3 non-whitespace characters.")
        return cleaned


class CreateRefundOutput(BaseModel):
    """Result of executing a refund request through deterministic financial gates."""

    created: bool
    refund_id: str | None = None
    order_id: str
    amount_vnd: int
    status: Literal["AUTO_APPROVED", "PENDING_HUMAN_APPROVAL", "REJECTED"]
    requires_human_review: bool
    policy_violation: str | None = None
    message: str
