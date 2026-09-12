# AI Engineering Lab

A long-term learning and building project for production Python and AI engineering.

The project starts intentionally small. Runtime dependencies will be added only when a learning stage needs them.

## Learning Log

The project history and the engineering concepts introduced at each milestone are documented in [docs/learning-log.md](docs/learning-log.md).

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

These same checks run automatically in the `Python quality` GitHub Actions workflow for pull requests targeting `main` or `develop`, and for pushes to those branches. In GitHub repository settings, mark the workflow's `Tests, Ruff, and mypy` job as a required status check under branch protection to enforce the standard before merging.

## Layout

- `src/ai_engineering_lab/`: importable application package
- `classifier.py`: deterministic support ticket classification
- `vectorizer.py`: educational bag-of-words vectors
- `tfidf.py`: normalized TF-IDF vectors and cosine similarity
- `retrieval.py`: reusable historical ticket retrieval
- `evaluation.py`: precision@k and recall@k metrics
- `tests/`: pytest test suite
- `docs/learning-log.md`: milestone history, decisions, and next steps
- `pyproject.toml`: project metadata and tool configuration
- `.venv/`: project-local virtual environment managed by `uv`

## Development Workflow

Each feature is developed on a dated branch, validated with the checks above, committed with a focused message, and pushed to GitHub. Pull request approval remains controlled by the GitHub repository's branch protection and review settings.
