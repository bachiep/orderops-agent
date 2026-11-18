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
| p50 Latency | `2.06 ms` |
| p95 Latency | `3.52 ms` |
| Avg Tokens / Run | `1489.5` |
| Measured Cost / 1,000 Runs (`local`) | `$0.0000` |
| Projected Cost / 1,000 Runs (`gpt-4o-mini`) | `$0.2660` |

## Breakdown by Scenario Category

| Category | Scenarios | TaskSuccessRate | TrajectoryAccuracy | Avg Steps | p50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `lookup` | 5 | 100.00% | 100.00% | 1.00 | 2.38 ms |
| `policy_inquiry` | 3 | 100.00% | 100.00% | 1.00 | 1.64 ms |
| `auto_refund_le_500k` | 5 | 100.00% | 100.00% | 3.00 | 2.36 ms |
| `escalation_gt_500k` | 5 | 100.00% | 100.00% | 3.00 | 2.27 ms |
| `policy_rejection` | 6 | 100.00% | 100.00% | 2.00 | 1.96 ms |
| `adversarial_injection` | 6 | 100.00% | 100.00% | 0.00 | 0.22 ms |
