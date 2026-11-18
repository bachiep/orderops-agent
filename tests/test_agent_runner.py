"""Unit tests for bounded AgentRunner loop, providers, retries, and breakers."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx

from orderops.agent import (
    AgentRunner,
    AgentRunRequest,
    GeminiProvider,
    LocalPlannerLLMProvider,
    ModelResponse,
    ModelToolCall,
    OpenAICompatibleProvider,
    ScriptedMockLLMProvider,
)
from orderops.config import Settings
from orderops.db import DatabaseManager
from orderops.tools import OrderOpsToolset, build_default_registry
from scripts.seed_demo_db import seed_demo_database


def _make_runner(
    provider,
    *,
    max_steps: int = 5,
    max_token_budget: int = 4096,
) -> tuple[DatabaseManager, AgentRunner]:
    fixed_now = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)
    db = DatabaseManager(":memory:")
    seed_demo_database(db, now=fixed_now)
    settings = Settings(max_steps=max_steps, max_token_budget=max_token_budget)
    toolset = OrderOpsToolset(db=db, settings=settings, now_provider=lambda: fixed_now)
    registry = build_default_registry(toolset)
    return db, AgentRunner(provider=provider, registry=registry, settings=settings)


async def test_local_planner_executes_multi_step_auto_refund() -> None:
    db, runner = _make_runner(LocalPlannerLLMProvider())
    try:
        result = await runner.run(
            AgentRunRequest(message="Please refund my defective order ORD-1001.")
        )
        assert result.intent == "REFUND_REQUEST"
        assert result.resolution_status == "AUTO_APPROVED"
        assert result.refund_id is not None and result.refund_id.startswith("RFD-")
        assert result.requires_human_review is False
        tool_names = [step.tool_name for step in result.tool_trace]
        assert tool_names == [
            "get_order_details",
            "evaluate_refund_eligibility",
            "create_refund_request",
        ]
    finally:
        db.close()


async def test_runner_stops_at_max_steps() -> None:
    infinite_tool_provider = ScriptedMockLLMProvider(
        lambda msgs: ModelResponse(
            tool_calls=[
                ModelToolCall(
                    id=f"c_{len(msgs)}",
                    name="get_order_details",
                    arguments={"order_id": "ORD-1001"},
                )
            ],
            prompt_tokens=20,
            completion_tokens=10,
            model_name="scripted-mock",
        )
    )
    db, runner = _make_runner(infinite_tool_provider, max_steps=3)
    try:
        result = await runner.run(AgentRunRequest(message="Check ORD-1001"))
        assert result.resolution_status == "FAILED_MAX_STEPS"
        assert "MAX_STEPS_REACHED" in result.guardrail_flags
        assert len(result.tool_trace) == 3
    finally:
        db.close()


async def test_runner_duplicate_failing_tool_breaker_halts_loop() -> None:
    repeating_bad_call = ScriptedMockLLMProvider(
        [
            ModelResponse(
                tool_calls=[
                    ModelToolCall(
                        id="c1",
                        name="create_refund_request",
                        arguments={"order_id": "ORD-1001", "amount_vnd": -100, "reason": "bad"},
                    )
                ],
                prompt_tokens=15,
                completion_tokens=10,
            ),
            ModelResponse(
                tool_calls=[
                    ModelToolCall(
                        id="c2",
                        name="create_refund_request",
                        arguments={"order_id": "ORD-1001", "amount_vnd": -100, "reason": "bad"},
                    )
                ],
                prompt_tokens=15,
                completion_tokens=10,
            ),
        ]
    )
    db, runner = _make_runner(repeating_bad_call, max_steps=5)
    try:
        result = await runner.run(AgentRunRequest(message="Refund ORD-1001"))
        assert result.resolution_status == "FAILED_MAX_STEPS"
        assert "DUPLICATE_FAILING_TOOL_BREAKER" in result.guardrail_flags
        assert len(result.tool_trace) == 1
    finally:
        db.close()


async def test_runner_enforces_token_budget() -> None:
    heavy_provider = ScriptedMockLLMProvider(
        [
            ModelResponse(
                tool_calls=[
                    ModelToolCall(
                        id="c1",
                        name="get_order_details",
                        arguments={"order_id": "ORD-1001"},
                    )
                ],
                prompt_tokens=250,
                completion_tokens=60,
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "intent": "ORDER_STATUS",
                        "summary": "Done",
                        "resolution_status": "RESOLVED_INFO",
                    }
                ),
                prompt_tokens=50,
                completion_tokens=20,
            ),
        ]
    )
    db, runner = _make_runner(heavy_provider, max_token_budget=300)
    try:
        result = await runner.run(AgentRunRequest(message="Status of ORD-1001"))
        assert result.resolution_status == "FAILED_BUDGET"
        assert "TOKEN_BUDGET_EXCEEDED" in result.guardrail_flags
    finally:
        db.close()


async def test_openai_and_gemini_providers_retry_transient_503() -> None:
    attempts = {"openai": 0, "gemini": 0}

    def openai_handler(request: httpx.Request) -> httpx.Response:
        attempts["openai"] += 1
        if attempts["openai"] == 1:
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_order_details",
                                        "arguments": '{"order_id":"ORD-1001"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 11},
            },
        )

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        attempts["gemini"] += 1
        if attempts["gemini"] == 1:
            return httpx.Response(429, json={"error": "rate limit"})
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_order_details",
                                        "args": {"order_id": "ORD-1002"},
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 9},
            },
        )

    settings = Settings(llm_max_retries=2, llm_api_key="test-key", gemini_api_key="test-gem")
    oa = OpenAICompatibleProvider(settings, transport=httpx.MockTransport(openai_handler))
    oa_resp = await oa.complete([{"role": "user", "content": "check"}], [])
    assert attempts["openai"] == 2
    assert oa_resp.tool_calls[0].name == "get_order_details"
    assert oa_resp.tool_calls[0].arguments == {"order_id": "ORD-1001"}

    gem = GeminiProvider(settings, transport=httpx.MockTransport(gemini_handler))
    gem_resp = await gem.complete([{"role": "user", "content": "check"}], [])
    assert attempts["gemini"] == 2
    assert gem_resp.tool_calls[0].arguments == {"order_id": "ORD-1002"}


async def test_local_planner_resolves_multi_turn_followup_from_history() -> None:
    db, runner = _make_runner(LocalPlannerLLMProvider())
    try:
        history = [
            {"role": "user", "content": "Check status of order ORD-1002."},
            {
                "role": "assistant",
                "content": (
                    "Order ORD-1002 (Wireless Noise-Canceling Headphones, 1,850,000 VND) "
                    "is currently DELIVERED."
                ),
            },
        ]
        res = await runner.run(
            AgentRunRequest(
                message="Please refund it because the left earcup is defective.",
                session_id="sess-followup-1",
            ),
            history=history,
        )
        assert res.session_id == "sess-followup-1"
        assert res.intent == "REFUND_REQUEST"
        assert res.resolution_status == "PENDING_HUMAN_APPROVAL"
        assert res.requires_human_review is True
        assert [step.tool_name for step in res.tool_trace] == [
            "get_order_details",
            "evaluate_refund_eligibility",
            "create_refund_request",
        ]
    finally:
        db.close()


async def test_local_planner_policy_inquiry_in_followup_does_not_hijack_prior_order() -> None:
    db, runner = _make_runner(LocalPlannerLLMProvider())
    try:
        history = [
            {"role": "user", "content": "Check status of order ORD-1002."},
            {"role": "assistant", "content": "Order ORD-1002 is DELIVERED."},
        ]
        res = await runner.run(
            AgentRunRequest(
                message="What is the return policy for perishable_food items?",
                session_id="sess-followup-2",
            ),
            history=history,
        )
        assert res.intent == "POLICY_INQUIRY"
        assert res.resolution_status == "RESOLVED_INFO"
        assert [step.tool_name for step in res.tool_trace] == ["lookup_return_policy"]
    finally:
        db.close()
