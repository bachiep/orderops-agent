"""Integration tests for FastAPI REST, SSE streaming, and masked audit persistence."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from orderops.api.main import create_app
from orderops.config import Settings
from orderops.db import DatabaseManager
from scripts.seed_demo_db import seed_demo_database


def _parse_sse_text(raw: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    current_event = "message"
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("event:"):
            current_event = stripped.removeprefix("event:").strip()
        elif stripped.startswith("data:"):
            payload = json.loads(stripped.removeprefix("data:").strip())
            events.append((current_event, payload))
    return events


async def test_post_agent_runs_rest_auto_approved_and_persists_masked_audit(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rest_api.db"
    with DatabaseManager(db_path) as db:
        seed_demo_database(db)
        settings = Settings(database_path=str(db_path), llm_provider="local")
        app = create_app(settings=settings, db=db)

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                resp = await client.post(
                    "/v1/agent/runs",
                    json={
                        "message": (
                            "Please refund defective ORD-1001. Reach me at "
                            "minh.anh@example.com or 0901234567."
                        ),
                        "session_id": "sess-1",
                        "customer_email": "minh.anh@example.com",
                    },
                )
                assert resp.status_code == 200
                body = resp.json()
                assert body["intent"] == "REFUND_REQUEST"
                assert body["resolution_status"] == "AUTO_APPROVED"
                assert body["refund_id"].startswith("RFD-")
                assert body["requires_human_review"] is False
                assert len(body["tool_trace"]) == 3

                runs_resp = await client.get("/v1/runs")
                assert runs_resp.status_code == 200
                runs_data = runs_resp.json()["runs"]
                assert len(runs_data) == 1
                assert runs_data[0]["run_id"] == body["run_id"]
                assert "0901234567" not in runs_data[0]["user_message_masked"]
                assert "minh.anh@example.com" not in runs_data[0]["user_message_masked"]
                assert runs_data[0]["customer_email_masked"] == "mi***@example.com"


async def test_post_agent_runs_stream_emits_ordered_sse_events_and_escalation(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sse_api.db"
    with DatabaseManager(db_path) as db:
        seed_demo_database(db)
        settings = Settings(database_path=str(db_path), llm_provider="local")
        app = create_app(settings=settings, db=db)

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                resp = await client.post(
                    "/v1/agent/runs/stream",
                    json={"message": "Request refund for defective headphones ORD-1002"},
                )
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]

                events = _parse_sse_text(resp.text)
                event_names = [name for name, _ in events]
                assert event_names[0] == "guardrail_check"
                assert "tool_call" in event_names
                assert "tool_result" in event_names
                assert "assistant_delta" in event_names
                assert event_names[-1] == "run_completed"

                completed_data = events[-1][1]
                assert completed_data["resolution_status"] == "PENDING_HUMAN_APPROVAL"
                assert completed_data["requires_human_review"] is True

                orders_resp = await client.get("/v1/orders")
                assert orders_resp.status_code == 200
                assert orders_resp.json()["total"] == 12
