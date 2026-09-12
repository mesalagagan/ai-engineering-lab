# AI Engineering Lab Learning Log

This document records what was built, why it was built that way, and which production Python concepts each milestone introduces.

## Project Setup

### Goal

Create a small, dependable Python laboratory for learning production AI engineering incrementally.

### Decisions

- Use Python 3.13+ and `uv` for the environment and dependency workflow.
- Use a `src/` layout so imports exercise the same package boundaries used by installable applications.
- Keep runtime dependencies empty until a learning stage needs one.
- Use pytest, Ruff, and strict mypy as the baseline quality gates.
- Keep the application package small and avoid introducing a web framework before the domain behavior is clear.

### Initial Structure

```text
src/ai_engineering_lab/  # importable application code
tests/                   # automated tests
pyproject.toml           # metadata and tool configuration
```

## Feature 1: Deterministic Support Ticket Classifier

### Goal

Classify support ticket text into a category, priority, and sentiment without machine learning or an LLM.

### Implementation

- `classifier.py` owns the domain model and deterministic keyword rules.
- `ClassificationResult` is an immutable, slotted dataclass.
- `Category`, `Priority`, and `Sentiment` are `StrEnum` values, which keep outputs constrained and JSON-friendly.
- `cli.py` owns command-line parsing and serialization.
- `__main__.py` provides the `python -m ai_engineering_lab` entry point.
- `test_classifier.py` verifies the requested authentication example, a billing case, and default behavior.

### Production Python Concepts

- Pure functions are easy to test because the same input produces the same result.
- Type hints make the domain contract explicit and are checked with strict mypy.
- A frozen dataclass prevents callers from mutating a classification after it is created.
- Separate domain logic from I/O so a future API or worker can reuse the classifier without importing CLI concerns.
- Small rule functions make precedence visible: authentication rules run before broader account rules.

### Validation

```text
pytest: 4 passed
ruff format --check: passed
ruff check: passed
mypy src tests: passed
CLI example: authentication / high / negative
```

### Deliberate Limitations

The classifier uses substring rules and is intentionally not production-complete. It does not yet handle configuration-driven rules, explain its decisions, or measure classification quality against labeled data. Those are useful later lessons, but adding them now would obscure the first domain boundary.

## Delivery Workflow

For each milestone:

1. Create a dated feature branch.
2. Update this log with the goal and design decisions.
3. Implement the smallest useful slice with tests.
4. Run pytest, Ruff, and mypy.
5. Commit the completed slice and push the branch to GitHub.

GitHub can require reviews, checks, or status approvals through repository settings. The local workflow can consistently validate and push code, but it must not attempt to bypass those repository controls.

## Next Milestones

- Add classification explanations while preserving the current domain contract.
- Move rules into explicit, testable configuration.
- Add a small evaluation dataset and measure rule coverage.
- Introduce an HTTP boundary only after the core domain behavior is stable.