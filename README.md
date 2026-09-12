# AI Engineering Lab

A long-term learning and building project for production Python and AI engineering.

The project starts intentionally small. Runtime dependencies will be added only when a learning stage needs them.

## Requirements

- Python 3.13+
- `uv`

## Setup

```powershell
uv sync
```

## Checks

```powershell
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
```

## Layout

- `src/ai_engineering_lab/`: importable application package
- `tests/`: pytest test suite
- `pyproject.toml`: project metadata and tool configuration
- `.venv/`: project-local virtual environment managed by `uv`
