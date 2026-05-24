# devcon2026

Reusable Python package for water-quality analyses migrated from project
notebooks.

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

## Demo

```bash
uv run python demo.py
```

The demo follows the dataframe workflow from the original nitrogen notebook. It
loads hydrologic CSV outputs when they are present in
`demo_outputs/example_hydro_model/`; otherwise it runs the packaged hydrologic
model, exports `states1.csv`, `fluxes1.csv`, `discharge1.csv`, and
`south_fork_aorc_forcing.csv`, then uses those files to drive the nitrogen
model. Plots are written to `demo_outputs/`.

## Notebook Migration Pattern

Keep original notebooks in `notebooks/` as references. Move reusable logic into
`src/devcon2026/` using this split:

- data loading: functions that accept file paths and return data objects
- analysis: pure functions that accept data and return results
- plotting: functions that accept results and an optional output path
- CLI/workflows: thin orchestration in `src/devcon2026/cli.py`

Avoid hidden notebook state by passing every input explicitly and returning
values instead of mutating globals.

## Package Modules

- `devcon2026.hydro`: hydrologic model components from the `ruben` branch, plus
  CSV export helpers for nitrogen workflows.
- `devcon2026.nitrogen`: nitrogen soil-process model from the `chucho` branch.

The original notebooks, demos, and data remain useful as references while the
reusable package API is built out.
