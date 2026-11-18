# OrderOps Agent Evaluation Report

- **Provider Mode:** `local`
- **Scenarios Evaluated:** `30` (including `6` adversarial prompt-injection attacks)

## Overall Metrics

| Metric | Measured Result |
| :--- | :---: |
| TaskSuccessRate | **100.00%** |
| ToolTrajectoryAccuracy | **100.00%** |
| SchemaValidityRate | **100.00%** |
| AdversarialBlockRate | **100.00%** |
| Avg Steps / Run | `1.67` |
| p50 Latency | `3.01 ms` |
| p95 Latency | `5.92 ms` |
| Avg Tokens / Run | `1486.9` |
| Measured Cost / 1,000 Runs (`local`) | `$0.0000` |
| Projected Cost / 1,000 Runs (`gpt-4o-mini`) | `$0.2660` |

## Breakdown by Scenario Category

| Category | Scenarios | TaskSuccessRate | TrajectoryAccuracy | Avg Steps | p50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `lookup` | 5 | 100.00% | 100.00% | 1.00 | 2.20 ms |
| `policy_inquiry` | 3 | 100.00% | 100.00% | 1.00 | 2.31 ms |
| `auto_refund_le_500k` | 5 | 100.00% | 100.00% | 3.00 | 3.07 ms |
| `escalation_gt_500k` | 5 | 100.00% | 100.00% | 3.00 | 3.84 ms |
| `policy_rejection` | 6 | 100.00% | 100.00% | 2.00 | 3.35 ms |
| `adversarial_injection` | 6 | 100.00% | 100.00% | 0.00 | 0.28 ms |
