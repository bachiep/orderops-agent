"""Typed tool registry, OpenAI/Gemini schema export, and safe execution wrapper."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from orderops.tools.handlers import OrderOpsToolset
from orderops.tools.schemas import (
    CreateRefundRequestInput,
    EvaluateRefundEligibilityInput,
    GetOrderDetailsInput,
    ListCustomerOrdersInput,
    LookupReturnPolicyInput,
)


@dataclass(frozen=True)
class RegisteredTool:
    """Metadata and callable binding for a single Pydantic-typed tool."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any], BaseModel]
    is_mutating: bool = False


class ToolExecutionResult(BaseModel):
    """Normalized outcome returned to the agent loop after executing a tool."""

    tool_name: str
    ok: bool
    duration_ms: float
    output: dict[str, Any]
    error: str | None = None


class ToolRegistry:
    """Validates tool arguments with Pydantic v2 and safely invokes registered handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        """Register a tool by unique name."""
        self._tools[tool.name] = tool

    def list_names(self) -> list[str]:
        """Return registered tool names in registration order."""
        return list(self._tools.keys())

    def get_openai_tools_schema(self) -> list[dict[str, Any]]:
        """Export OpenAI/Gemini-compatible function declarations from Pydantic schemas."""
        definitions: list[dict[str, Any]] = []
        for tool in self._tools.values():
            raw_schema = tool.input_model.model_json_schema()
            raw_schema.pop("title", None)
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": raw_schema,
                    },
                }
            )
        return definitions

    def execute(self, tool_name: str, raw_arguments: dict[str, Any] | str) -> ToolExecutionResult:
        """Parse, validate, and execute a tool without throwing unhandled exceptions."""
        started = time.perf_counter()
        tool = self._tools.get(tool_name)
        if tool is None:
            elapsed = (time.perf_counter() - started) * 1000.0
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                duration_ms=round(elapsed, 2),
                output={"error": f"Unknown tool '{tool_name}'."},
                error=f"UNKNOWN_TOOL:{tool_name}",
            )

        try:
            args_dict = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
            )
            validated = tool.input_model.model_validate(args_dict)
            result_model = tool.handler(validated)
            output_dict = result_model.model_dump(mode="json")
            elapsed = (time.perf_counter() - started) * 1000.0
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=True,
                duration_ms=round(elapsed, 2),
                output=output_dict,
                error=None,
            )
        except (json.JSONDecodeError, TypeError, ValidationError) as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                duration_ms=round(elapsed, 2),
                output={"error": f"Invalid arguments for '{tool_name}': {exc}"},
                error=f"VALIDATION_ERROR:{exc}",
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            elapsed = (time.perf_counter() - started) * 1000.0
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                duration_ms=round(elapsed, 2),
                output={"error": f"Tool execution error in '{tool_name}': {exc}"},
                error=f"EXECUTION_ERROR:{exc}",
            )


def build_default_registry(toolset: OrderOpsToolset) -> ToolRegistry:
    """Register all 5 core OrderOps tools."""
    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            name="get_order_details",
            description=(
                "Read-only lookup of a single order by order_id (status, delivered_at, "
                "days_since_delivery, total_vnd, tracking_code, and existing refund status)."
            ),
            input_model=GetOrderDetailsInput,
            handler=toolset.get_order_details,
            is_mutating=False,
        )
    )
    registry.register(
        RegisteredTool(
            name="list_customer_orders",
            description="Read-only lookup listing recent orders for a customer email address.",
            input_model=ListCustomerOrdersInput,
            handler=toolset.list_customer_orders,
            is_mutating=False,
        )
    )
    registry.register(
        RegisteredTool(
            name="lookup_return_policy",
            description=(
                "Read-only lookup of returnability, return window days, and restocking fees "
                "for a product category and issue type."
            ),
            input_model=LookupReturnPolicyInput,
            handler=toolset.lookup_return_policy,
            is_mutating=False,
        )
    )
    registry.register(
        RegisteredTool(
            name="evaluate_refund_eligibility",
            description=(
                "Deterministic eligibility check verifying order status, delivery 14-day window, "
                "category policy, duplicate refunds, and maximum refundable VND."
            ),
            input_model=EvaluateRefundEligibilityInput,
            handler=toolset.evaluate_refund_eligibility,
            is_mutating=False,
        )
    )
    registry.register(
        RegisteredTool(
            name="create_refund_request",
            description=(
                "Mutating tool that creates a refund request in SQLite after enforcing "
                "deterministic financial rules (<= 500,000 VND auto-approved; > 500,000 VND "
                "queued as PENDING_HUMAN_APPROVAL)."
            ),
            input_model=CreateRefundRequestInput,
            handler=toolset.create_refund_request,
            is_mutating=True,
        )
    )
    return registry
