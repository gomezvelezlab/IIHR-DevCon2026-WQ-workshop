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
uv run python demo1layer.py
uv run python demo3layer.py
uv run python demo_hydrology_tiles.py
```

Use `--force-hydrology` to rerun hydrology even when exported Parquet artifacts
already exist. Use `--no-progress` for quiet batch runs.

`demo1layer.py` follows the dataframe workflow from the original nitrogen
notebook with a single soil control volume. `demo3layer.py` routes dissolved
nitrogen through soil, active groundwater, and passive groundwater compartments.
`demo_hydrology_tiles.py` compares the legacy tile-drainage formulation against
the water-table formulation and a no-tile scenario.
Hydrology and nitrogen handoff artifacts are Parquet files in
`demo_outputs/example_hydrology_model/`. Plots are written to `demo_outputs/`.

The executable workflow is intentionally thin:

```python
from devcon2026.hydrology import Hydrology, HydrologyArtifactNames
from devcon2026.hydrology import HydrologyParameters, HydrologyStates
from devcon2026.nitrogen import Nitrogen, NitrogenParameters, NitrogenStates

DISCHARGE_ARTIFACT = "discharge1.parquet"
STATES_ARTIFACT = "states1.parquet"
FLUXES_ARTIFACT = "fluxes1.parquet"
FORCING_ARTIFACT = "south_fork_aorc_forcing.parquet"
artifacts = HydrologyArtifactNames(
    discharge=DISCHARGE_ARTIFACT,
    states=STATES_ARTIFACT,
    fluxes=FLUXES_ARTIFACT,
    forcing=FORCING_ARTIFACT,
)

hydrology = Hydrology(
    output_dir="demo_outputs/example_hydrology_model",
    artifact_names=artifacts,
    params=HydrologyParameters(),
    initial_states=HydrologyStates(s_sn=0.01, s_s=0.03, s_gwa=0.2, s_gwp=0.5),
)
hydrology.solve()
hydrology.export()

nitrogen = Nitrogen(
    output_dir="demo_outputs",
    params=NitrogenParameters(v_denit=0.05),
    initial_states=NitrogenStates(
        m_don=500.0,
        m_din=2500.0,
        m_son=4.5e5,
        m_fon=1.0e4,
        m_don_ads=0.0,
    ),
)
nitrogen.load_hydrology(hydrology.output_dir, artifact_names=artifacts)
nitrogen.solve()
nitrogen.export()
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

## Package Modules

- `devcon2026.hydrology`: hydrologic model components from the `ruben` branch, plus
  CSV export helpers for nitrogen workflows.
- `devcon2026.nitrogen`: nitrogen soil-process model from the `chucho` branch.

The original notebooks, demos, and data remain useful as references while the
reusable package API is built out.
