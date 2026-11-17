"""Layer 1 Input Guardrail: length checks, prompt-injection detection, and PII masking."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from orderops.guardrails.pii import redact_pii

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "INSTRUCTION_OVERRIDE",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+"
            r"(?:all\s+|previous\s+|prior\s+|above\s+|your\s+|system\s+|"
            r"return\s+|refund\s+|company\s+|safety\s+)*"
            r"(?:instructions?|rules?|policies|policy|guidelines|prompts?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "POLICY_BYPASS_EN",
        re.compile(
            r"\b(?:bypass|skip|disable|override|circumvent)\s+"
            r"(?:the\s+|all\s+|any\s+|return\s+|refund\s+)*"
            r"(?:14[- ]?day|return\s+window|eligibility|policy|policies|guardrails?|"
            r"approval|human\s+review|limits?|checks?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "FORCE_AUTO_APPROVE_EN",
        re.compile(
            r"\b(?:force|mark|set|approve)\b.{0,80}\bauto[_ -]?approv(?:e|ed|al)\b|"
            r"\b(?:without|no)\s+human\s+(?:approval|review)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ROLE_HIJACK_OR_PROMPT_LEAK",
        re.compile(
            r"\b(?:system\s+override|developer\s+mode|jailbreak|dan\s+mode|"
            r"pretend\s+you\s+are\s+(?:an?\s+)?admin(?:istrator)?|"
            r"(?:reveal|print|dump|show)\s+(?:your\s+)?system\s+prompt)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "POLICY_BYPASS_VI",
        re.compile(
            r"(?:bỏ\s+qua|phớt\s+lờ|lách)\s+"
            r"(?:quy\s+định|chính\s+sách|kiểm\s+tra|giới\s+hạn|điều\s+kiện|hạn\s+mức|14\s*ngày)|"
            r"(?:không\s+cần|miễn)\s+(?:duyệt\s+tay|kiểm\s+tra|xét\s+duyệt)|"
            r"(?:ép|cưỡng\s+bức)\s+(?:duyệt|phê\s+duyệt|hoàn\s+tiền)|"
            r"vượt\s+(?:quyền|giới\s+hạn|hạn\s+mức)|"
            r"(?:tiết\s+lộ|in\s+ra|xem)\s+system\s+prompt",
            re.IGNORECASE,
        ),
    ),
)


class InputGuardResult(BaseModel):
    """Outcome of evaluating Layer 1 input guardrails."""

    allowed: bool
    flags: list[str] = Field(default_factory=list)
    reason: str | None = None
    masked_preview: str


def evaluate_input_guardrail(message: str, *, max_input_chars: int = 1500) -> InputGuardResult:
    """Validate input length, detect prompt-injection/policy-override, and redact PII."""
    stripped = message.strip()
    masked_preview, pii_flags = redact_pii(stripped)

    if not stripped:
        return InputGuardResult(
            allowed=False,
            flags=["EMPTY_INPUT"],
            reason="Input message cannot be empty.",
            masked_preview="",
        )

    if len(stripped) > max_input_chars:
        return InputGuardResult(
            allowed=False,
            flags=["INPUT_TOO_LONG", *pii_flags],
            reason=f"Input exceeds maximum allowed length of {max_input_chars} characters.",
            masked_preview=masked_preview[:200],
        )

    matched_rules: list[str] = []
    for label, pattern in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            matched_rules.append(label)

    if matched_rules:
        return InputGuardResult(
            allowed=False,
            flags=["PROMPT_INJECTION_DETECTED", *matched_rules, *pii_flags],
            reason=(
                "Request blocked by input safety guardrail: policy-override or prompt-injection "
                "instructions are not permitted. Refunds must pass deterministic policy checks."
            ),
            masked_preview=masked_preview[:240],
        )

    return InputGuardResult(
        allowed=True,
        flags=pii_flags,
        reason=None,
        masked_preview=masked_preview[:240],
    )
