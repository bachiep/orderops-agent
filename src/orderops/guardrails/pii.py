"""PII redaction utilities and structured logging filter."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

_EMAIL_RE = re.compile(r"\b([\w.+-]+)@([\w-]+\.[\w.-]+)\b")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,18}\d(?!\d)")
_PHONE_VN_RE = re.compile(r"(?<!\d)(?:\+84|0)(?:3|5|7|8|9)\d{1,2}[\s.-]?\d{3}[\s.-]?\d{3,4}(?!\d)")
_CCCD_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")


def mask_email(email: str) -> str:
    """Mask the local part of an email address while preserving domain for triage."""
    local, _, domain = email.strip().partition("@")
    if not domain:
        return "***"
    prefix = local[:2] if len(local) > 2 else local[:1]
    return f"{prefix}***@{domain}"


def redact_pii(text: str, *, redact_email: bool = True) -> tuple[str, list[str]]:
    """Redact payment cards, phones, citizen IDs, and optionally emails from text."""
    flags: list[str] = []
    redacted = text

    def _replace_card(match: re.Match[str]) -> str:
        digits = re.sub(r"[^\d]", "", match.group(0))
        if 13 <= len(digits) <= 19:
            if "CARD_REDACTED" not in flags:
                flags.append("CARD_REDACTED")
            return f"[REDACTED_CARD_{digits[-4:]}]"
        return match.group(0)

    redacted = _CARD_RE.sub(_replace_card, redacted)

    def _replace_phone(match: re.Match[str]) -> str:
        digits = re.sub(r"[^\d]", "", match.group(0))
        if 9 <= len(digits) <= 12:
            if "PHONE_REDACTED" not in flags:
                flags.append("PHONE_REDACTED")
            return f"{digits[:3]}****{digits[-3:]}"
        return match.group(0)

    redacted = _PHONE_VN_RE.sub(_replace_phone, redacted)

    if _CCCD_RE.search(redacted):
        redacted = _CCCD_RE.sub("[REDACTED_ID]", redacted)
        flags.append("ID_REDACTED")

    if redact_email and _EMAIL_RE.search(redacted):
        redacted = _EMAIL_RE.sub(lambda m: mask_email(m.group(0)), redacted)
        flags.append("EMAIL_REDACTED")

    return redacted, flags


class _PIIRedactingJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON with PII redacted."""

    def format(self, record: logging.LogRecord) -> str:
        raw_message = record.getMessage()
        safe_message, _ = redact_pii(raw_message)
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": safe_message,
        }
        extra_data = getattr(record, "structured_data", None)
        if isinstance(extra_data, dict):
            serialized = json.dumps(extra_data, ensure_ascii=False, default=str)
            safe_serialized, _ = redact_pii(serialized)
            payload["data"] = json.loads(safe_serialized)
        return json.dumps(payload, ensure_ascii=False)


def get_structured_logger(name: str = "orderops") -> logging.Logger:
    """Return a logger configured with PII-redacting JSON formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_PIIRedactingJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
