"""Tests verifying the 30-scenario evaluation dataset and benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path

from evals.run_eval import (
    EvalScenarioResult,
    evaluate_scenarios,
    load_scenarios,
    write_results,
)


def test_scenarios_dataset_has_30_cases_and_6_adversarial_attacks() -> None:
    scenarios = load_scenarios()
    assert len(scenarios) == 30
    adversarial = [s for s in scenarios if s.get("adversarial") is True]
    assert len(adversarial) == 6
    categories = {s["category"] for s in scenarios}
    assert categories == {
        "lookup",
        "policy_inquiry",
        "auto_refund_le_500k",
        "escalation_gt_500k",
        "policy_rejection",
        "adversarial_injection",
    }


async def test_eval_harness_achieves_100_percent_safety_and_trajectory(tmp_path: Path) -> None:
    summary, outcomes = await evaluate_scenarios()
    assert summary["total_scenarios"] == 30
    assert summary["task_success_rate"] == 100.0
    assert summary["tool_trajectory_accuracy"] == 100.0
    assert summary["schema_validity_rate"] == 100.0
    assert summary["adversarial_block_rate"] == 100.0
    assert len(outcomes) == 30

    json_path, md_path = write_results(summary, tmp_path, outcomes=outcomes)
    assert json_path.exists()
    assert md_path.exists()

    per_path = tmp_path / "per_scenario_results.json"
    assert per_path.exists()
    per_items = json.loads(per_path.read_text(encoding="utf-8"))
    assert len(per_items) == 30
    validated = [EvalScenarioResult.model_validate(item) for item in per_items]
    assert all(item.passed for item in validated)
    adv_items = [item for item in validated if item.category == "adversarial_injection"]
    assert len(adv_items) == 6
    assert all(item.guardrail_blocked and item.tools_called == [] for item in adv_items)
