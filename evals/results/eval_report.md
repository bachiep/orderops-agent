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
| p50 Latency | `1.76 ms` |
| p95 Latency | `3.49 ms` |
| Avg Tokens / Run | `1486.9` |
| Measured Cost / 1,000 Runs (`local`) | `$0.0000` |
| Projected Cost / 1,000 Runs (`gpt-4o-mini`) | `$0.2660` |

## Breakdown by Scenario Category

| Category | Scenarios | TaskSuccessRate | TrajectoryAccuracy | Avg Steps | p50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `lookup` | 5 | 100.00% | 100.00% | 1.00 | 1.67 ms |
| `policy_inquiry` | 3 | 100.00% | 100.00% | 1.00 | 1.52 ms |
| `auto_refund_le_500k` | 5 | 100.00% | 100.00% | 3.00 | 2.98 ms |
| `escalation_gt_500k` | 5 | 100.00% | 100.00% | 3.00 | 2.28 ms |
| `policy_rejection` | 6 | 100.00% | 100.00% | 2.00 | 1.76 ms |
| `adversarial_injection` | 6 | 100.00% | 100.00% | 0.00 | 0.19 ms |
