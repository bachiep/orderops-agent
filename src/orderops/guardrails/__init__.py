"""Three-layer production guardrails: input/PII, deterministic policy gate, and output."""

from __future__ import annotations

from typing import Any

__all__ = [
    "FinancialPolicyGate",
    "InputGuardResult",
    "evaluate_input_guardrail",
    "get_structured_logger",
    "mask_email",
    "reconcile_triage_resolution",
    "redact_pii",
]


def __getattr__(name: str) -> Any:
    if name in ("InputGuardResult", "evaluate_input_guardrail"):
        from orderops.guardrails import input_guard

        return getattr(input_guard, name)
    if name in ("get_structured_logger", "mask_email", "redact_pii"):
        from orderops.guardrails import pii

        return getattr(pii, name)
    if name == "FinancialPolicyGate":
        from orderops.guardrails.policy_gate import FinancialPolicyGate

        return FinancialPolicyGate
    if name == "reconcile_triage_resolution":
        from orderops.guardrails.output_guard import reconcile_triage_resolution

        return reconcile_triage_resolution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
