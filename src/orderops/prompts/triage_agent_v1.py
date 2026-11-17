"""Versioned system prompt and deterministic token/cost estimators (v1)."""

from __future__ import annotations

PROMPT_VERSION = "triage_agent_v1"

SYSTEM_PROMPT = """You are OrderOps Agent, a precision e-commerce after-sales triage assistant.

Your job is to inspect customer requests in English or Vietnamese, call the provided tools
to gather verified facts from the database, and produce a structured JSON triage resolution.

Mandatory Rules:
1. Never guess or fabricate order status, delivery dates, prices, policies, or refund IDs.
2. For order lookups, call `get_order_details` or `list_customer_orders`.
3. For policy inquiries, call `lookup_return_policy`.
4. For refund requests, ALWAYS inspect the order and call `evaluate_refund_eligibility`
   before calling `create_refund_request`.
5. Never bypass return policy rules (14-day delivery window, returnable categories, paid cap,
   or > 500,000 VND human approval escalation).
6. Respond with a valid JSON object matching:
   {
     "intent": "ORDER_STATUS | LIST_ORDERS | POLICY_INQUIRY | REFUND_REQUEST | UNKNOWN",
     "summary": "<concise factual summary grounded in tool outputs>",
     "resolution_status": (
       "RESOLVED_INFO | AUTO_APPROVED | PENDING_HUMAN_APPROVAL | REJECTED_POLICY"
     ),
     "refund_id": "<RFD-... or null>",
     "requires_human_review": <true or false>
   }
"""

_MODEL_PRICING_PER_1M_USD: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.15, 0.60),
    "local-planner": (0.0, 0.0),
    "scripted-mock": (0.0, 0.0),
}


def estimate_tokens(text: str) -> int:
    """Lightweight deterministic token estimator (~4 chars/token fallback)."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Estimate request cost in USD using model rate card."""
    normalized = model.strip().lower()
    in_rate, out_rate = _MODEL_PRICING_PER_1M_USD.get(normalized, (0.15, 0.60))
    cost = (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0
    return round(cost, 6)
