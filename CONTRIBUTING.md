# Contributing

Thank you for your interest in contributing to **OrderOps Agent**.

## Development Setup

```bash
# Prerequisites: Python 3.13+, Docker (for integration tests)
git clone https://github.com/bachiep/orderops-agent.git
cd orderops-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v                           # run full test suite (49 tests)
```

## Workflow

1. **Open an Issue** describing the bug or feature.
2. **Create a branch** from `main`: `feat/issue-<n>-<slug>` or `fix/issue-<n>-<slug>`.
3. **Write tests first** — especially for guardrail rules and tool-calling logic.
4. **Follow Conventional Commits**: `feat(guardrails): add financial ceiling check (#9)`.
5. **Ensure CI passes**: `ruff check . && pytest -v`.
6. **Open a Pull Request** referencing the issue: `Closes #<n>`.

## Code Standards

- Python 3.13, async FastAPI, Pydantic v2 strict models.
- Ruff linter — zero warnings.
- 3-layer guardrails: input guard (regex), policy engine (business rules), financial ceiling.
- HITL Supervisor Queue for high-value approvals.
- 30-scenario deterministic evaluation suite in `evals/`.

## Docker Smoke Tests

```bash
docker compose up -d --build
# Test: adversarial guardrail block, HITL refund approval, supervisor session update
docker compose down -v
```

## Evaluation

```bash
python -m evals.run_eval                              # local deterministic baseline (30 scenarios)
LLM_PROVIDER=gemini python -m evals.run_eval          # live Cloud LLM (requires API key)
```