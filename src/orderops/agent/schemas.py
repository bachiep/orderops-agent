"""Pydantic v2 request, trace, streaming, and final output schemas for the agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

IntentType = Literal[
    "ORDER_STATUS",
    "LIST_ORDERS",
    "POLICY_INQUIRY",
    "REFUND_REQUEST",
    "BLOCKED_ADVERSARIAL",
    "UNKNOWN",
]

ResolutionStatus = Literal[
    "RESOLVED_INFO",
    "AUTO_APPROVED",
    "PENDING_HUMAN_APPROVAL",
    "REJECTED_POLICY",
    "BLOCKED_GUARDRAIL",
    "FAILED_MAX_STEPS",
    "FAILED_BUDGET",
]

StreamEventType = Literal[
    "guardrail_check",
    "tool_call",
    "tool_result",
    "assistant_delta",
    "run_completed",
]


class AgentRunRequest(BaseModel):
    """Input payload for executing a bounded OrderOps triage run."""

    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=80)
    customer_email: str | None = Field(default=None, max_length=120)


class ToolTraceStep(BaseModel):
    """Recorded execution trace of a single tool invocation."""

    step: int
    tool_name: str
    arguments: dict[str, Any]
    ok: bool
    duration_ms: float
    result_summary: dict[str, Any]
    error: str | None = None


class TokenUsage(BaseModel):
    """Token consumption breakdown across all turns in a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TriageResolution(BaseModel):
    """Validated final structured output of an OrderOps Agent run."""

    run_id: str
    intent: IntentType
    summary: str
    tool_trace: list[ToolTraceStep] = Field(default_factory=list)
    resolution_status: ResolutionStatus
    refund_id: str | None = None
    requires_human_review: bool = False
    guardrail_flags: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: float = 0.0


class AgentStreamEvent(BaseModel):
    """Single Server-Sent Event frame emitted during agent execution."""

    event: StreamEventType
    data: dict[str, Any]
