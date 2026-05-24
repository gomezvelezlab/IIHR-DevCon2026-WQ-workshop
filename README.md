# devcon2026

Reusable Python package for analyses migrated from project notebooks.

## Setup

```bash
uv sync
```

## Run

```bash
uv run devcon2026
```

## Test

```bash
uv run pytest
```

## Notebook Migration Pattern

Keep original notebooks in `notebooks/` as references. Move reusable logic into
`src/devcon2026/` using this split:

- data loading: functions that accept file paths and return data objects
- analysis: pure functions that accept data and return results
- plotting: functions that accept results and an optional output path
- CLI/workflows: thin orchestration in `src/devcon2026/cli.py`

Avoid hidden notebook state by passing every input explicitly and returning
values instead of mutating globals.
