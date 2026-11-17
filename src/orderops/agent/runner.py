"""Bounded tool-calling state loop for OrderOps Agent."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import ValidationError

from orderops.agent.providers import AgentModelProvider
from orderops.agent.schemas import (
    AgentRunRequest,
    AgentStreamEvent,
    IntentType,
    ResolutionStatus,
    TokenUsage,
    ToolTraceStep,
    TriageResolution,
)
from orderops.config import Settings
from orderops.guardrails.input_guard import evaluate_input_guardrail
from orderops.guardrails.output_guard import reconcile_triage_resolution
from orderops.guardrails.pii import get_structured_logger
from orderops.prompts.triage_agent_v1 import SYSTEM_PROMPT, estimate_cost_usd
from orderops.tools.registry import ToolRegistry

logger = get_structured_logger("orderops.agent")


def _summarize_tool_output(output: dict[str, Any]) -> dict[str, Any]:
    """Keep tool_trace concise and free of oversized nested lists."""
    summary: dict[str, Any] = {}
    for key, value in output.items():
        if key == "orders" and isinstance(value, list):
            summary["orders_count"] = len(value)
            summary["order_ids"] = [o.get("order_id") for o in value[:5] if isinstance(o, dict)]
        else:
            summary[key] = value
    return summary


def _parse_model_resolution(
    raw_content: str | None,
    *,
    run_id: str,
    tool_trace: list[ToolTraceStep],
    guardrail_flags: list[str],
    latency_ms: float,
    token_usage: TokenUsage,
    estimated_cost_usd: float,
) -> TriageResolution:
    """Parse final assistant output into TriageResolution with resilient fallback."""
    intent: IntentType = "UNKNOWN"
    resolution_status: ResolutionStatus = "RESOLVED_INFO"
    refund_id: str | None = None
    requires_human_review = False
    summary = (raw_content or "").strip()

    if raw_content:
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                intent = parsed.get("intent", intent)
                summary = str(parsed.get("summary") or summary)
                resolution_status = parsed.get("resolution_status", resolution_status)
                refund_id = parsed.get("refund_id")
                requires_human_review = bool(parsed.get("requires_human_review", False))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if not summary:
        summary = "Request processed."

    try:
        return TriageResolution(
            run_id=run_id,
            intent=intent,
            summary=summary,
            tool_trace=tool_trace,
            resolution_status=resolution_status,
            refund_id=refund_id,
            requires_human_review=requires_human_review,
            guardrail_flags=guardrail_flags,
            latency_ms=round(latency_ms, 2),
            token_usage=token_usage,
            estimated_cost_usd=estimated_cost_usd,
        )
    except ValidationError:
        return TriageResolution(
            run_id=run_id,
            intent="UNKNOWN",
            summary=summary,
            tool_trace=tool_trace,
            resolution_status="RESOLVED_INFO",
            refund_id=None,
            requires_human_review=False,
            guardrail_flags=guardrail_flags + ["SCHEMA_COERCED"],
            latency_ms=round(latency_ms, 2),
            token_usage=token_usage,
            estimated_cost_usd=estimated_cost_usd,
        )


def _chunk_text(text: str, chunk_size: int = 48) -> list[str]:
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


class AgentRunner:
    """Explicit bounded tool-calling loop with step limit, token budget, and duplicate breaker."""

    def __init__(
        self,
        provider: AgentModelProvider,
        registry: ToolRegistry,
        settings: Settings | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.settings = settings or Settings()

    async def run(self, request: AgentRunRequest) -> TriageResolution:
        """Execute a run to completion and return the validated TriageResolution."""
        final_resolution: TriageResolution | None = None
        async for event in self.run_stream(request):
            if event.event == "run_completed":
                final_resolution = TriageResolution.model_validate(event.data)
        if final_resolution is None:  # pragma: no cover - invariant
            raise RuntimeError("Agent stream ended without a run_completed event.")
        return final_resolution

    async def run_stream(self, request: AgentRunRequest) -> AsyncIterator[AgentStreamEvent]:
        """Stream structured execution events from guardrail checks through completion."""
        started = time.perf_counter()
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        user_text = request.message.strip()
        if request.customer_email and request.customer_email not in user_text:
            user_text = f"{user_text}\n[customer_email: {request.customer_email}]"

        input_guard = evaluate_input_guardrail(
            user_text,
            max_input_chars=self.settings.max_input_chars,
        )
        logger.info(
            "Agent run started",
            extra={
                "structured_data": {
                    "run_id": run_id,
                    "guard_allowed": input_guard.allowed,
                    "flags": input_guard.flags,
                    "preview": input_guard.masked_preview,
                }
            },
        )
        yield AgentStreamEvent(
            event="guardrail_check",
            data={
                "layer": "input",
                "passed": input_guard.allowed,
                "flags": input_guard.flags,
                "masked_preview": input_guard.masked_preview,
            },
        )
        if not input_guard.allowed:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            blocked = TriageResolution(
                run_id=run_id,
                intent="BLOCKED_ADVERSARIAL",
                summary=input_guard.reason or "Request blocked by input guardrail.",
                tool_trace=[],
                resolution_status="BLOCKED_GUARDRAIL",
                refund_id=None,
                requires_human_review=False,
                guardrail_flags=input_guard.flags,
                latency_ms=round(elapsed_ms, 2),
                token_usage=TokenUsage(),
                estimated_cost_usd=0.0,
            )
            for delta in _chunk_text(blocked.summary):
                yield AgentStreamEvent(event="assistant_delta", data={"delta": delta})
            yield AgentStreamEvent(
                event="run_completed",
                data=blocked.model_dump(mode="json"),
            )
            return

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        tool_schemas = self.registry.get_openai_tools_schema()
        tool_trace: list[ToolTraceStep] = []
        guardrail_flags: list[str] = list(input_guard.flags)
        failed_call_counts: dict[tuple[str, str], int] = {}
        prompt_tokens = 0
        completion_tokens = 0
        model_name = self.settings.llm_model

        for step in range(1, self.settings.max_steps + 1):
            if (prompt_tokens + completion_tokens) >= self.settings.max_token_budget:
                guardrail_flags.append("TOKEN_BUDGET_EXCEEDED")
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                resolution = TriageResolution(
                    run_id=run_id,
                    intent="UNKNOWN",
                    summary="Agent stopped because the per-run token budget was exceeded.",
                    tool_trace=tool_trace,
                    resolution_status="FAILED_BUDGET",
                    guardrail_flags=guardrail_flags,
                    latency_ms=round(elapsed_ms, 2),
                    token_usage=usage,
                    estimated_cost_usd=estimate_cost_usd(
                        prompt_tokens, completion_tokens, model_name
                    ),
                )
                yield AgentStreamEvent(
                    event="run_completed",
                    data=resolution.model_dump(mode="json"),
                )
                return

            model_resp = await self.provider.complete(messages=messages, tools=tool_schemas)
            model_name = model_resp.model_name or model_name
            prompt_tokens += model_resp.prompt_tokens
            completion_tokens += model_resp.completion_tokens

            if not model_resp.tool_calls:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                cost = estimate_cost_usd(prompt_tokens, completion_tokens, model_name)
                raw_resolution = _parse_model_resolution(
                    model_resp.content,
                    run_id=run_id,
                    tool_trace=tool_trace,
                    guardrail_flags=guardrail_flags,
                    latency_ms=elapsed_ms,
                    token_usage=usage,
                    estimated_cost_usd=cost,
                )
                resolution = reconcile_triage_resolution(raw_resolution, tool_trace)
                for delta in _chunk_text(resolution.summary):
                    yield AgentStreamEvent(event="assistant_delta", data={"delta": delta})
                yield AgentStreamEvent(
                    event="run_completed",
                    data=resolution.model_dump(mode="json"),
                )
                return

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": model_resp.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, sort_keys=True),
                        },
                    }
                    for tc in model_resp.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in model_resp.tool_calls:
                canonical_key = (tc.name, json.dumps(tc.arguments, sort_keys=True))
                if failed_call_counts.get(canonical_key, 0) >= 1:
                    guardrail_flags.append("DUPLICATE_FAILING_TOOL_BREAKER")
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    usage = TokenUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                    resolution = TriageResolution(
                        run_id=run_id,
                        intent="UNKNOWN",
                        summary=(
                            f"Stopped loop after repeated failing call to '{tc.name}' "
                            "with identical arguments."
                        ),
                        tool_trace=tool_trace,
                        resolution_status="FAILED_MAX_STEPS",
                        guardrail_flags=guardrail_flags,
                        latency_ms=round(elapsed_ms, 2),
                        token_usage=usage,
                        estimated_cost_usd=estimate_cost_usd(
                            prompt_tokens, completion_tokens, model_name
                        ),
                    )
                    yield AgentStreamEvent(
                        event="run_completed",
                        data=resolution.model_dump(mode="json"),
                    )
                    return

                yield AgentStreamEvent(
                    event="tool_call",
                    data={"step": step, "tool_name": tc.name, "arguments": tc.arguments},
                )
                exec_res = self.registry.execute(tc.name, tc.arguments)
                is_domain_reject = (
                    tc.name == "create_refund_request"
                    and exec_res.output.get("created") is False
                )
                if not exec_res.ok or is_domain_reject:
                    failed_call_counts[canonical_key] = failed_call_counts.get(canonical_key, 0) + 1

                trace_step = ToolTraceStep(
                    step=step,
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    ok=exec_res.ok,
                    duration_ms=exec_res.duration_ms,
                    result_summary=_summarize_tool_output(exec_res.output),
                    error=exec_res.error,
                )
                tool_trace.append(trace_step)

                yield AgentStreamEvent(
                    event="tool_result",
                    data={
                        "step": step,
                        "tool_name": tc.name,
                        "ok": exec_res.ok,
                        "duration_ms": exec_res.duration_ms,
                        "result": exec_res.output,
                        "error": exec_res.error,
                    },
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": json.dumps(exec_res.output, ensure_ascii=False),
                    }
                )

        guardrail_flags.append("MAX_STEPS_REACHED")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        resolution = TriageResolution(
            run_id=run_id,
            intent="UNKNOWN",
            summary=f"Agent halted safely after reaching max_steps={self.settings.max_steps}.",
            tool_trace=tool_trace,
            resolution_status="FAILED_MAX_STEPS",
            guardrail_flags=guardrail_flags,
            latency_ms=round(elapsed_ms, 2),
            token_usage=usage,
            estimated_cost_usd=estimate_cost_usd(prompt_tokens, completion_tokens, model_name),
        )
        yield AgentStreamEvent(event="run_completed", data=resolution.model_dump(mode="json"))
