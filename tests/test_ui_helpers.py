"""Unit tests for Streamlit Trace Console SSE parsing and scenario helpers."""

from __future__ import annotations

import json

from orderops.ui.app import (
    DEFAULT_SAMPLE_SCENARIOS,
    format_resolution_badge,
    parse_sse_line_events,
)


def test_default_sample_scenarios_cover_core_and_adversarial_cases() -> None:
    assert len(DEFAULT_SAMPLE_SCENARIOS) == 6
    prompts = [p for _, p in DEFAULT_SAMPLE_SCENARIOS]
    assert any("ORD-1001" in p for p in prompts)
    assert any("ORD-1002" in p for p in prompts)
    assert any("Ignore previous instructions" in p for p in prompts)


def test_parse_sse_line_events_handles_full_orderops_stream() -> None:
    lines = iter(
        [
            "event: guardrail_check",
            'data: {"layer": "input", "passed": true, "flags": []}',
            "",
            "event: tool_call",
            "data: "
            + json.dumps(
                {"step": 1, "tool_name": "get_order_details", "arguments": {"order_id": "ORD-1001"}}
            ),
            "",
            "event: tool_result",
            'data: {"step": 1, "tool_name": "get_order_details", "ok": true, "duration_ms": 0.4}',
            "",
            "event: assistant_delta",
            'data: {"delta": "Refund approved."}',
            "",
            "event: run_completed",
            "data: " + json.dumps({"resolution_status": "AUTO_APPROVED", "refund_id": "RFD-123"}),
            "",
        ]
    )
    events = list(parse_sse_line_events(lines))
    assert [name for name, _ in events] == [
        "guardrail_check",
        "tool_call",
        "tool_result",
        "assistant_delta",
        "run_completed",
    ]
    assert events[-1][1]["resolution_status"] == "AUTO_APPROVED"
    assert format_resolution_badge("PENDING_HUMAN_APPROVAL") == "PENDING_HUMAN_APPROVAL"
