"""Pluggable LLM providers: LocalPlanner, ScriptedMock, OpenAI-compatible, and Gemini."""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, Field

from orderops.config import Settings
from orderops.prompts.triage_agent_v1 import estimate_tokens

_ORDER_ID_RE = re.compile(r"\b(ORD-\d+)\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_AMOUNT_RE = re.compile(
    r"(?<!limit\s)(?<!ngưỡng\s)(?:refund|hoàn(?:\s+tiền)?|partial(?:\s+refund)?|amount|số\s+tiền)?"
    r"[^\d\n]{0,16}([\d][,\d.]{3,11})\s*(?:vnd|vnđ|đ|d\b)",
    re.IGNORECASE,
)


class ModelToolCall(BaseModel):
    """Normalized function/tool call emitted by a model provider."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Normalized model turn output (either tool calls or final assistant content)."""

    content: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_name: str = "local-planner"


class AgentModelProvider(ABC):
    """Abstract model provider supporting OpenAI-style tool calling."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Run one completion turn given conversation history and tool definitions."""


class ScriptedMockLLMProvider(AgentModelProvider):
    """Deterministic scripted provider for unit tests and edge-case simulation."""

    def __init__(
        self,
        responses: Sequence[ModelResponse] | Callable[[list[dict[str, Any]]], ModelResponse],
    ) -> None:
        self._responses = list(responses) if not callable(responses) else None
        self._callback = responses if callable(responses) else None
        self._cursor = 0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        if self._callback is not None:
            return self._callback(messages)
        assert self._responses is not None
        if not self._responses:
            raise RuntimeError("ScriptedMockLLMProvider has no scripted responses.")
        idx = min(self._cursor, len(self._responses) - 1)
        self._cursor += 1
        return self._responses[idx]


def _extract_category(text: str) -> str:
    lower = text.lower()
    mapping = (
        ("perishable_food", ("perishable", "food", "coffee", "cà phê", "thực phẩm", "đồ ăn")),
        ("digital_voucher", ("voucher", "e-gift", "gift card", "gift code", "phiếu quà")),
        ("cosmetics", ("cosmetic", "serum", "skincare", "mỹ phẩm", "kem dưỡng")),
        ("apparel", ("apparel", "shirt", "overshirt", "hoodie", "clothing", "quần áo", "áo")),
        (
            "home_appliances",
            ("home_appliances", "appliance", "lamp", "purifier", "mug", "gia dụng", "đèn"),
        ),
        (
            "electronics",
            ("electronic", "headphone", "keyboard", "keycap", "mouse", "điện tử", "tai nghe"),
        ),
    )
    for category, keywords in mapping:
        if any(kw in lower for kw in keywords):
            return category
    return "electronics"


def _extract_reason_code(text: str) -> str:
    lower = text.lower()
    if any(
        k in lower
        for k in ("changed mind", "change of mind", "change my mind", "đổi ý", "không thích")
    ):
        return "changed_mind"
    if any(
        k in lower
        for k in ("damaged in transit", "crushed", "shipping damage", "vỡ", "móp", "khi vận chuyển")
    ):
        return "damaged_in_transit"
    if any(
        k in lower
        for k in ("wrong item", "wrong size", "wrong color", "giao nhầm", "sai mẫu", "nhầm size")
    ):
        return "wrong_item"
    return "defective"


def _extract_explicit_amount(text: str) -> int | None:
    for match in _AMOUNT_RE.finditer(text):
        digits = re.sub(r"[^\d]", "", match.group(1))
        if digits:
            value = int(digits)
            if value >= 1000:
                return value
    return None


def _collect_tool_history(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    history: list[tuple[str, dict[str, Any]]] = []
    for msg in messages:
        if msg.get("role") == "tool":
            name = str(msg.get("name", ""))
            raw_content = msg.get("content", "{}")
            try:
                parsed = (
                    json.loads(raw_content)
                    if isinstance(raw_content, str)
                    else dict(raw_content)
                )
            except json.JSONDecodeError:
                parsed = {}
            history.append((name, parsed))
    return history


class LocalPlannerLLMProvider(AgentModelProvider):
    """Zero-cost deterministic planner emitting authentic multi-step tool calls."""

    model_name = "local-planner"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        last_user_idx = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default=-1,
        )
        latest_user = str(messages[last_user_idx]["content"]) if last_user_idx >= 0 else ""
        prior_messages = (
            [m for m in messages[:last_user_idx] if m.get("role") in ("user", "assistant")]
            if last_user_idx > 0
            else []
        )
        prompt_tokens = sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)

        tool_history = _collect_tool_history(messages)
        explicit_order_match = _ORDER_ID_RE.search(latest_user)
        explicit_order_id = (
            explicit_order_match.group(1).upper() if explicit_order_match else None
        )
        prior_order_id: str | None = None
        for prior_msg in reversed(prior_messages):
            match = _ORDER_ID_RE.search(str(prior_msg.get("content", "")))
            if match:
                prior_order_id = match.group(1).upper()
                break

        email_match = _EMAIL_RE.search(latest_user)
        email = email_match.group(0).lower() if email_match else None

        lower = latest_user.lower()
        is_refund = any(
            kw in lower
            for kw in ("refund", "return", "hoàn tiền", "trả hàng", "đổi trả", "money back")
        )
        policy_keywords = (
            "policy",
            "chính sách",
            "quy định",
            "returnable",
            "restocking",
            "điều kiện đổi",
        )
        action_keywords = ("refund", "hoàn tiền", "trả hàng", "process", "tạo")
        is_policy = any(kw in lower for kw in policy_keywords) and not (
            explicit_order_id and any(k in lower for k in action_keywords)
        )

        order_id = explicit_order_id or (None if email is not None else prior_order_id)

        # Resolve order_id from list_customer_orders if customer only gave email
        listed_res = next(
            (out for name, out in tool_history if name == "list_customer_orders"),
            None,
        )
        if order_id is None and listed_res and listed_res.get("orders"):
            order_id = str(listed_res["orders"][0]["order_id"])

        def _respond_tool(call_id: str, name: str, args: dict[str, Any]) -> ModelResponse:
            call = ModelToolCall(id=call_id, name=name, arguments=args)
            comp = estimate_tokens(json.dumps({"name": name, "arguments": args}))
            return ModelResponse(
                content=None,
                tool_calls=[call],
                prompt_tokens=prompt_tokens,
                completion_tokens=comp,
                model_name=self.model_name,
            )

        def _respond_json(payload: dict[str, Any]) -> ModelResponse:
            content = json.dumps(payload, ensure_ascii=False)
            return ModelResponse(
                content=content,
                tool_calls=[],
                prompt_tokens=prompt_tokens,
                completion_tokens=estimate_tokens(content),
                model_name=self.model_name,
            )

        if (is_policy and not is_refund) or (is_policy and explicit_order_id is None):
            policy_res = next((out for n, out in tool_history if n == "lookup_return_policy"), None)
            if policy_res is None:
                return _respond_tool(
                    "call_policy_1",
                    "lookup_return_policy",
                    {
                        "category": _extract_category(latest_user),
                        "issue_type": _extract_reason_code(latest_user),
                    },
                )
            cat = policy_res.get("category", "unknown")
            if policy_res.get("is_returnable"):
                summary = (
                    f"Return policy for {cat} ({policy_res.get('issue_type')}): returnable within "
                    f"{policy_res.get('return_window_days')} days, restocking fee "
                    f"{policy_res.get('restocking_fee_pct')}%. {policy_res.get('notes', '')}"
                )
            else:
                summary = (
                    f"Category {cat} ({policy_res.get('issue_type')}) is NOT returnable: "
                    f"{policy_res.get('notes', '')}"
                )
            return _respond_json(
                {
                    "intent": "POLICY_INQUIRY",
                    "summary": summary.strip(),
                    "resolution_status": "RESOLVED_INFO",
                    "refund_id": None,
                    "requires_human_review": False,
                }
            )

        if email is not None and order_id is None and listed_res is None:
            return _respond_tool(
                "call_list_1",
                "list_customer_orders",
                {"customer_email": email, "limit": 10},
            )

        if not is_refund and email is not None and listed_res is not None:
            orders = listed_res.get("orders", [])
            if not orders:
                masked = listed_res.get("customer_email_masked", "customer")
                summary = f"No orders found for {masked}."
            else:
                items = ", ".join(f"{o['order_id']} ({o['status']})" for o in orders)
                summary = (
                    f"Found {len(orders)} order(s) for {listed_res.get('customer_name')}: {items}."
                )
            return _respond_json(
                {
                    "intent": "LIST_ORDERS",
                    "summary": summary,
                    "resolution_status": "RESOLVED_INFO",
                    "refund_id": None,
                    "requires_human_review": False,
                }
            )

        if order_id is not None:
            order_res = next((out for n, out in tool_history if n == "get_order_details"), None)
            if order_res is None:
                return _respond_tool("call_order_1", "get_order_details", {"order_id": order_id})

            if not order_res.get("found"):
                return _respond_json(
                    {
                        "intent": "REFUND_REQUEST" if is_refund else "ORDER_STATUS",
                        "summary": f"Order {order_id} was not found in the system.",
                        "resolution_status": "REJECTED_POLICY" if is_refund else "RESOLVED_INFO",
                        "refund_id": None,
                        "requires_human_review": False,
                    }
                )

            if not is_refund:
                summary = (
                    f"Order {order_id} ({order_res.get('item_name')}, "
                    f"{order_res.get('total_vnd', 0):,} VND) is currently "
                    f"{order_res.get('status')}. "
                    f"Tracking code: {order_res.get('tracking_code') or 'N/A'}."
                )
                return _respond_json(
                    {
                        "intent": "ORDER_STATUS",
                        "summary": summary,
                        "resolution_status": "RESOLVED_INFO",
                        "refund_id": order_res.get("existing_refund_id"),
                        "requires_human_review": False,
                    }
                )

            reason_code = _extract_reason_code(latest_user)
            if reason_code == "defective":
                for prior_msg in reversed(prior_messages):
                    if prior_msg.get("role") == "user":
                        prior_reason = _extract_reason_code(str(prior_msg.get("content", "")))
                        if prior_reason != "defective":
                            reason_code = prior_reason
                            break
            elig_res = next(
                (out for n, out in tool_history if n == "evaluate_refund_eligibility"),
                None,
            )
            if elig_res is None:
                return _respond_tool(
                    "call_elig_1",
                    "evaluate_refund_eligibility",
                    {"order_id": order_id, "reason_code": reason_code},
                )

            if not elig_res.get("eligible"):
                codes = ", ".join(elig_res.get("reason_codes", []))
                return _respond_json(
                    {
                        "intent": "REFUND_REQUEST",
                        "summary": (
                            f"Refund request for {order_id} rejected ({codes}). "
                            f"{elig_res.get('policy_notes', '')}"
                        ).strip(),
                        "resolution_status": "REJECTED_POLICY",
                        "refund_id": order_res.get("existing_refund_id"),
                        "requires_human_review": False,
                    }
                )

            refund_res = next(
                (out for n, out in tool_history if n == "create_refund_request"),
                None,
            )
            if refund_res is None:
                requested = _extract_explicit_amount(latest_user)
                amount = requested if requested is not None else int(elig_res["max_refundable_vnd"])
                return _respond_tool(
                    "call_refund_1",
                    "create_refund_request",
                    {
                        "order_id": order_id,
                        "amount_vnd": amount,
                        "reason": latest_user[:240],
                        "reason_code": reason_code,
                    },
                )

            if not refund_res.get("created"):
                msg = str(refund_res.get("message", "Refund rejected by policy gate."))
                return _respond_json(
                    {
                        "intent": "REFUND_REQUEST",
                        "summary": msg,
                        "resolution_status": "REJECTED_POLICY",
                        "refund_id": None,
                        "requires_human_review": False,
                    }
                )

            status = str(refund_res["status"])
            return _respond_json(
                {
                    "intent": "REFUND_REQUEST",
                    "summary": str(refund_res.get("message", f"Refund created for {order_id}.")),
                    "resolution_status": status,
                    "refund_id": refund_res.get("refund_id"),
                    "requires_human_review": bool(refund_res.get("requires_human_review", False)),
                }
            )

        return _respond_json(
            {
                "intent": "UNKNOWN",
                "summary": (
                    "Please provide a valid order ID (e.g., ORD-1001) or customer email "
                    "so I can check order status, return policy, or process a refund."
                ),
                "resolution_status": "RESOLVED_INFO",
                "refund_id": None,
                "requires_human_review": False,
            }
        )


async def _post_json_with_retry(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    max_retries: int,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Execute POST JSON with bounded exponential backoff on transient HTTP/network errors."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds, transport=transport) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    await asyncio.sleep(0.1 * (2**attempt))
                    continue
                resp.raise_for_status()
                return dict(resp.json())
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            await asyncio.sleep(0.1 * (2**attempt))
    raise RuntimeError(f"HTTP call failed after retries: {last_error}")


class OpenAICompatibleProvider(AgentModelProvider):
    """Async httpx client for OpenAI, Groq, or Ollama chat/completions tool-calling APIs."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        base_url = (self.settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.0,
        }
        data = await _post_json_with_retry(
            f"{base_url}/chat/completions",
            headers=headers,
            payload=payload,
            timeout_seconds=self.settings.llm_timeout_seconds,
            max_retries=self.settings.llm_max_retries,
            transport=self.transport,
        )
        choice = data["choices"][0]["message"]
        usage = data.get("usage", {})
        parsed_calls: list[ModelToolCall] = []
        for idx, tc in enumerate(choice.get("tool_calls") or []):
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            parsed_calls.append(
                ModelToolCall(
                    id=str(tc.get("id") or f"call_{idx}"),
                    name=str(fn["name"]),
                    arguments=args,
                )
            )
        return ModelResponse(
            content=choice.get("content"),
            tool_calls=parsed_calls,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            model_name=self.settings.llm_model,
        )


class GeminiProvider(AgentModelProvider):
    """Async httpx client for Google Gemini generateContent with functionDeclarations."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        model = self.settings.llm_model or "gemini-2.0-flash"
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.settings.gemini_api_key}"
        )
        system_texts = [m["content"] for m in messages if m.get("role") == "system"]
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                contents.append({"role": "user", "parts": [{"text": str(msg.get("content", ""))}]})
            elif role == "assistant":
                parts: list[dict[str, Any]] = []
                if msg.get("content"):
                    parts.append({"text": str(msg["content"])})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    parts.append({"functionCall": {"name": fn["name"], "args": args}})
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                raw = msg.get("content", "{}")
                resp_obj = json.loads(raw) if isinstance(raw, str) else dict(raw)
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": str(msg.get("name", "tool")),
                                    "response": resp_obj,
                                }
                            }
                        ],
                    }
                )

        declarations = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            }
            for t in tools
        ]
        payload: dict[str, Any] = {
            "contents": contents,
            "tools": [{"functionDeclarations": declarations}],
            "generationConfig": {"temperature": 0.0},
        }
        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}

        data = await _post_json_with_retry(
            url,
            headers={"Content-Type": "application/json"},
            payload=payload,
            timeout_seconds=self.settings.llm_timeout_seconds,
            max_retries=self.settings.llm_max_retries,
            transport=self.transport,
        )
        candidate = data["candidates"][0]["content"]
        parts = candidate.get("parts", [])
        text_chunks: list[str] = []
        parsed_calls: list[ModelToolCall] = []
        for idx, part in enumerate(parts):
            if "text" in part:
                text_chunks.append(str(part["text"]))
            if "functionCall" in part:
                fc = part["functionCall"]
                parsed_calls.append(
                    ModelToolCall(
                        id=f"gemini_call_{idx}",
                        name=str(fc["name"]),
                        arguments=dict(fc.get("args", {})),
                    )
                )
        usage = data.get("usageMetadata", {})
        return ModelResponse(
            content="\n".join(text_chunks) if text_chunks else None,
            tool_calls=parsed_calls,
            prompt_tokens=int(usage.get("promptTokenCount", 0)),
            completion_tokens=int(usage.get("candidatesTokenCount", 0)),
            model_name=model,
        )


def build_model_provider(settings: Settings) -> AgentModelProvider:
    """Instantiate the configured model provider from Settings."""
    provider_name = settings.llm_provider.strip().lower()
    if provider_name in ("openai", "ollama"):
        return OpenAICompatibleProvider(settings)
    if provider_name == "gemini":
        return GeminiProvider(settings)
    return LocalPlannerLLMProvider()
