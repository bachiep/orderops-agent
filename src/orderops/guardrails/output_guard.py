"""Layer 3 Output Guardrail: reconcile TriageResolution against verified tool trace."""

from __future__ import annotations

from orderops.agent.schemas import ToolTraceStep, TriageResolution
from orderops.guardrails.pii import redact_pii


def reconcile_triage_resolution(
    resolution: TriageResolution,
    tool_trace: list[ToolTraceStep],
) -> TriageResolution:
    """Ensure final output strictly matches verified DB tool outcomes and redacts PII."""
    flags = list(resolution.guardrail_flags)
    safe_summary, pii_flags = redact_pii(resolution.summary, redact_email=True)
    for pf in pii_flags:
        if pf not in flags:
            flags.append(pf)

    refund_created_step = next(
        (
            step
            for step in reversed(tool_trace)
            if step.tool_name == "create_refund_request"
            and step.ok
            and step.result_summary.get("created") is True
        ),
        None,
    )

    refund_rejected_step = next(
        (
            step
            for step in reversed(tool_trace)
            if step.tool_name == "create_refund_request"
            and step.result_summary.get("created") is False
        ),
        None,
    )

    eligibility_rejected_step = next(
        (
            step
            for step in reversed(tool_trace)
            if step.tool_name == "evaluate_refund_eligibility"
            and step.ok
            and step.result_summary.get("eligible") is False
        ),
        None,
    )

    next_status = resolution.resolution_status
    next_refund_id = resolution.refund_id
    next_human_review = resolution.requires_human_review

    if refund_created_step is not None:
        verified_status = str(refund_created_step.result_summary["status"])
        verified_refund_id = str(refund_created_step.result_summary["refund_id"])
        verified_human = bool(refund_created_step.result_summary["requires_human_review"])
        if (
            next_status != verified_status
            or next_refund_id != verified_refund_id
            or next_human_review != verified_human
        ):
            flags.append("OUTPUT_RECONCILED_WITH_TOOL_TRACE")
        next_status = verified_status  # type: ignore[assignment]
        next_refund_id = verified_refund_id
        next_human_review = verified_human
    elif refund_rejected_step is not None or eligibility_rejected_step is not None:
        if next_status in ("AUTO_APPROVED", "PENDING_HUMAN_APPROVAL") or (
            next_refund_id is not None and resolution.intent == "REFUND_REQUEST"
        ):
            flags.append("HALLUCINATED_APPROVAL_BLOCKED")
            next_refund_id = None
        if next_status not in ("FAILED_MAX_STEPS", "FAILED_BUDGET", "BLOCKED_GUARDRAIL"):
            next_status = "REJECTED_POLICY"
        next_human_review = False
    elif resolution.intent == "REFUND_REQUEST" and next_status in (
        "AUTO_APPROVED",
        "PENDING_HUMAN_APPROVAL",
    ):
        # LLM cannot claim a refund succeeded when create_refund_request never succeeded.
        flags.append("UNVERIFIED_REFUND_CLAIM_BLOCKED")
        next_status = "REJECTED_POLICY"
        next_refund_id = None
        next_human_review = False

    return resolution.model_copy(
        update={
            "summary": safe_summary,
            "resolution_status": next_status,
            "refund_id": next_refund_id,
            "requires_human_review": next_human_review,
            "guardrail_flags": flags,
        }
    )
