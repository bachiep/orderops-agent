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
                assert "v2.1.0" in html
                assert "desk-chat-card" in html
                assert "idempotency-key-input" in html
                assert "Run Agent Stream" in html
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


async def test_hitl_decision_appends_closed_loop_notification_to_session_messages(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "hitl_closed_loop.db"
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
                chat_resp = await client.post(
                    "/v1/chat",
                    json={"message": "Request refund for defective headphones ORD-1002"},
                )
                assert chat_resp.status_code == 200
                chat_body = chat_resp.json()
                session_id = chat_body["session_id"]
                refund_id = chat_body["refund_id"]
                assert chat_body["resolution_status"] == "PENDING_HUMAN_APPROVAL"

                decision_resp = await client.post(
                    f"/v1/refunds/{refund_id}/decision",
                    json={
                        "decision": "APPROVE",
                        "reviewer_note": "Approved after QA check; contact 0901234567",
                        "reviewer_id": "sup.mai",
                    },
                )
                assert decision_resp.status_code == 200
                decided = decision_resp.json()
                assert decided["session_id"] == session_id
                assert decided["session_notified"] is True

                sess_resp = await client.get(f"/v1/sessions/{session_id}")
                assert sess_resp.status_code == 200
                messages = sess_resp.json()["messages"]
                assert len(messages) == 3
                last_msg = messages[-1]
                assert last_msg["role"] == "assistant"
                assert last_msg["intent"] == "HITL_DECISION"
                assert last_msg["resolution_status"] == "APPROVED"
                assert last_msg["refund_id"] == refund_id
                assert "[Supervisor Update]" in last_msg["content"]
                assert refund_id in last_msg["content"]
                assert "ORD-1002" in last_msg["content"]
                assert "APPROVED" in last_msg["content"]
                assert "0901234567" not in last_msg["content"]


async def test_post_refunds_idempotency_replay_and_409_conflict_and_hitl_reject_notification(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "refund_idempotency.db"
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
                payload = {
                    "order_id": "ORD-1008",
                    "amount_vnd": 1_290_000,
                    "reason": "Mechanical switches double-typing",
                    "reason_code": "defective",
                    "session_id": "sess-direct-1008",
                }
                first = await client.post(
                    "/v1/refunds",
                    json=payload,
                    headers={"Idempotency-Key": "ik-ord1008-001"},
                )
                assert first.status_code == 200
                assert first.headers["X-Idempotent-Replay"] == "false"
                first_body = first.json()
                assert first_body["status"] == "PENDING_HUMAN_APPROVAL"
                assert first_body["session_id"] == "sess-direct-1008"
                assert first_body["idempotent_replay"] is False
                refund_id = first_body["refund_id"]

                # Exact retry returns the existing RefundRequest without duplicating HITL tickets
                replay = await client.post(
                    "/v1/refunds",
                    json=payload,
                    headers={"Idempotency-Key": "ik-ord1008-001"},
                )
                assert replay.status_code == 200
                assert replay.headers["X-Idempotent-Replay"] == "true"
                replay_body = replay.json()
                assert replay_body["refund_id"] == refund_id
                assert replay_body["idempotent_replay"] is True

                row_count = db.fetch_one(
                    "SELECT COUNT(*) AS cnt FROM refund_requests WHERE order_id = 'ORD-1008';"
                )
                assert row_count is not None and int(row_count["cnt"]) == 1

                # Mismatched payload under same Idempotency-Key -> 409 Conflict (RFC 7807)
                conflict = await client.post(
                    "/v1/refunds",
                    json={**payload, "amount_vnd": 900_000},
                    headers={"Idempotency-Key": "ik-ord1008-001"},
                )
                assert conflict.status_code == 409
                assert conflict.headers["content-type"].startswith("application/problem+json")
                conflict_body = conflict.json()
                assert conflict_body["status"] == 409
                assert conflict_body["title"] == "Idempotency-Key Conflict"
                assert "detail" in conflict_body

                # Blank Idempotency-Key -> 422
                blank_key = await client.post(
                    "/v1/refunds",
                    json=payload,
                    headers={"Idempotency-Key": "   "},
                )
                assert blank_key.status_code == 422

                # Reject direct refund and verify closed-loop notification in session_messages
                reject_resp = await client.post(
                    f"/v1/refunds/{refund_id}/decision",
                    json={
                        "decision": "REJECT",
                        "reviewer_note": "No keycap defect visible in video proof.",
                        "reviewer_id": "sup.tuan",
                    },
                )
                assert reject_resp.status_code == 200
                assert reject_resp.json()["session_notified"] is True

                sess_resp = await client.get("/v1/sessions/sess-direct-1008")
                assert sess_resp.status_code == 200
                msgs = sess_resp.json()["messages"]
                assert len(msgs) == 1
                assert "[Supervisor Update]" in msgs[0]["content"]
                assert "REJECTED" in msgs[0]["content"]

                # Policy violation on direct refund submission -> 422 RFC 7807
                expired = await client.post(
                    "/v1/refunds",
                    json={
                        "order_id": "ORD-1003",
                        "amount_vnd": 450_000,
                        "reason": "Expired lamp",
                        "reason_code": "defective",
                    },
                )
                assert expired.status_code == 422
                assert expired.json()["policy_violation"] == "EXPIRED_RETURN_WINDOW"


async def test_agent_runs_and_stream_support_idempotency_key_replay_and_409(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "agent_idempotency.db"
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
                req_body = {
                    "message": (
                        "My headphones in ORD-1002 have static audio. Please process a refund."
                    ),
                    "session_id": "sess-idem-agent",
                }
                first = await client.post(
                    "/v1/agent/runs",
                    json=req_body,
                    headers={"Idempotency-Key": "ik-agent-1002"},
                )
                assert first.status_code == 200
                assert first.headers["X-Idempotent-Replay"] == "false"
                first_json = first.json()

                replay = await client.post(
                    "/v1/agent/runs",
                    json=req_body,
                    headers={"Idempotency-Key": "ik-agent-1002"},
                )
                assert replay.status_code == 200
                assert replay.headers["X-Idempotent-Replay"] == "true"
                assert replay.json()["run_id"] == first_json["run_id"]
                assert replay.json()["refund_id"] == first_json["refund_id"]

                mismatch = await client.post(
                    "/v1/agent/runs",
                    json={"message": "Check ORD-1001", "session_id": "sess-idem-agent"},
                    headers={"Idempotency-Key": "ik-agent-1002"},
                )
                assert mismatch.status_code == 409

                stream_req = {"message": "Check ORD-1001", "session_id": "sess-idem-stream"}
                stream_first = await client.post(
                    "/v1/agent/runs/stream",
                    json=stream_req,
                    headers={"Idempotency-Key": "ik-stream-1001"},
                )
                assert stream_first.status_code == 200
                assert stream_first.headers["X-Idempotent-Replay"] == "false"

                stream_replay = await client.post(
                    "/v1/agent/runs/stream",
                    json=stream_req,
                    headers={"Idempotency-Key": "ik-stream-1001"},
                )
                assert stream_replay.status_code == 200
                assert stream_replay.headers["X-Idempotent-Replay"] == "true"
                events = _parse_sse_text(stream_replay.text)
                assert events[0][0] == "guardrail_check"
                assert events[0][1]["idempotent_replay"] is True
                assert events[-1][0] == "run_completed"
                assert events[-1][1]["resolution_status"] == "RESOLVED_INFO"

                stream_mismatch = await client.post(
                    "/v1/agent/runs/stream",
                    json={"message": "Check ORD-1002", "session_id": "sess-idem-stream"},
                    headers={"Idempotency-Key": "ik-stream-1001"},
                )
                assert stream_mismatch.status_code == 409
