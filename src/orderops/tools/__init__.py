"""Typed Pydantic v2 tools and execution registry for OrderOps Agent."""

from orderops.tools.handlers import OrderOpsToolset
from orderops.tools.registry import ToolExecutionResult, ToolRegistry, build_default_registry
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
    ReasonCode,
    RefundEligibilityOutput,
    ReturnPolicyOutput,
)

__all__ = [
    "CreateRefundOutput",
    "CreateRefundRequestInput",
    "CustomerOrdersOutput",
    "EvaluateRefundEligibilityInput",
    "GetOrderDetailsInput",
    "ListCustomerOrdersInput",
    "LookupReturnPolicyInput",
    "OrderDetailsOutput",
    "OrderOpsToolset",
    "OrderSummary",
    "ReasonCode",
    "RefundEligibilityOutput",
    "ReturnPolicyOutput",
    "ToolExecutionResult",
    "ToolRegistry",
    "build_default_registry",
]
