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
                assert orders_resp.json()["total"] == 13


async def test_ops_console_html_served_at_root(tmp_path: Path) -> None:
    db_path = tmp_path / "ui_console.db"
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
                resp = await client.get("/")
                assert resp.status_code == 200
                assert "text/html" in resp.headers["content-type"]
                html = resp.text
                assert "Agent Support Desk" in html
                assert "Supervisor Approval Queue" in html
                assert "Safety &amp; Tool Eval Scorecard" in html
                assert "OpenAPI /docs" in html
                assert "PENDING APPROVAL" in html
                assert "AUTO APPROVED" in html
                assert "BLOCKED BY POLICY" in html


async def test_multi_turn_session_persistence_and_chat_endpoints(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions_api.db"
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
                turn1 = await client.post(
                    "/v1/chat",
                    json={"message": "Check the status of order ORD-1008."},
                )
                assert turn1.status_code == 200
                t1_body = turn1.json()
                session_id = t1_body["session_id"]
                assert session_id and session_id.startswith("sess_")
                assert t1_body["intent"] == "ORDER_STATUS"
                assert t1_body["resolution_status"] == "RESOLVED_INFO"

                turn2 = await client.post(
                    "/v1/agent/runs",
                    json={
                        "session_id": session_id,
                        "message": (
                            "Please refund it because keys stopped working. Call 0901234567."
                        ),
                    },
                )
                assert turn2.status_code == 200
                t2_body = turn2.json()
                assert t2_body["session_id"] == session_id
                assert t2_body["intent"] == "REFUND_REQUEST"
                assert t2_body["resolution_status"] == "PENDING_HUMAN_APPROVAL"
                assert t2_body["refund_id"].startswith("RFD-")

                sess_resp = await client.get(f"/v1/sessions/{session_id}")
                assert sess_resp.status_code == 200
                sess_body = sess_resp.json()
                assert sess_body["total_messages"] == 4
                assert [m["role"] for m in sess_body["messages"]] == [
                    "user",
                    "assistant",
                    "user",
                    "assistant",
                ]
                assert len(sess_body["run_ids"]) == 2
                assert "0901234567" not in sess_body["messages"][2]["content_masked"]

                list_resp = await client.get("/v1/sessions")
                assert list_resp.status_code == 200
                assert any(s["session_id"] == session_id for s in list_resp.json()["sessions"])


async def test_hitl_supervisor_queue_approve_reject_and_conflicts(tmp_path: Path) -> None:
    db_path = tmp_path / "hitl_api.db"
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
                pending_resp = await client.get("/v1/refunds/pending")
                assert pending_resp.status_code == 200
                pending_body = pending_resp.json()
                assert pending_body["total"] == 1
                assert pending_body["refunds"][0]["refund_id"] == "RFD-SEED-002"
                assert pending_body["refunds"][0]["amount_vnd"] == 650_000
                assert pending_body["refunds"][0]["customer_name"] == "Pham Gia Huy"

                # Missing refund -> 404
                missing_resp = await client.post(
                    "/v1/refunds/RFD-UNKNOWN/decision",
                    json={"decision": "APPROVE", "reviewer_note": "Checking missing"},
                )
                assert missing_resp.status_code == 404

                # Approve seeded pending refund -> APPROVED & order REFUNDED, PII redacted
                approve_resp = await client.post(
                    "/v1/refunds/RFD-SEED-002/decision",
                    json={
                        "decision": "APPROVE",
                        "reviewer_note": "Verified return package with 0901234567",
                        "reviewer_id": "sup.linh",
                    },
                )
                assert approve_resp.status_code == 200
                approved = approve_resp.json()
                assert approved["status"] == "APPROVED"
                assert approved["order_status"] == "REFUNDED"
                assert "0901234567" not in approved["reviewer_note"]
                assert approved["reviewer_id"] == "sup.linh"

                # Duplicate decision -> 409 Conflict
                conflict_resp = await client.post(
                    "/v1/refunds/RFD-SEED-002/decision",
                    json={"decision": "REJECT", "reviewer_note": "Too late"},
                )
                assert conflict_resp.status_code == 409

                # Create another >500k pending refund on ORD-1002 and reject it
                run_resp = await client.post(
                    "/v1/agent/runs",
                    json={"message": "Request refund for defective headphones ORD-1002"},
                )
                assert run_resp.status_code == 200
                new_refund_id = run_resp.json()["refund_id"]

                reject_resp = await client.post(
                    f"/v1/refunds/{new_refund_id}/decision",
                    json={"decision": "REJECT", "reviewer_note": "Serial mismatch on inspection"},
                )
                assert reject_resp.status_code == 200
                rejected = reject_resp.json()
                assert rejected["status"] == "REJECTED"
                assert rejected["order_status"] == "DELIVERED"

                rejected_list = await client.get("/v1/refunds/pending?status=REJECTED")
                assert rejected_list.status_code == 200
                assert any(r["refund_id"] == new_refund_id for r in rejected_list.json()["refunds"])

                bad_filter = await client.get("/v1/refunds/pending?status=INVALID")
                assert bad_filter.status_code == 422


async def test_latest_evals_endpoint_returns_30_scenarios(tmp_path: Path) -> None:
    db_path = tmp_path / "evals_api.db"
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
                resp = await client.get("/v1/evals/latest")
                assert resp.status_code == 200
                body = resp.json()
                assert body["total_scenarios"] == 30
                assert body["passed_scenarios"] == 30
                assert body["summary"]["task_success_rate"] == 100.0
                assert len(body["scenarios"]) == 30
                sample = body["scenarios"][0]
                for field in (
                    "id",
                    "category",
                    "prompt",
                    "expected_decision",
                    "actual_decision",
                    "tools_called",
                    "guardrail_blocked",
                    "passed",
                    "latency_ms",
                ):
                    assert field in sample
