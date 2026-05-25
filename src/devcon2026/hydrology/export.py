"""Export hydrologic model outputs for nitrogen-model workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import MILLIMETERS_PER_METER
from .constants import SECONDS_PER_DAY
from .types import SimulationResult


def _with_time_column(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.insert(0, "time", output.index)
    return output.reset_index(drop=True)


def export_nitrogen_hydrology_inputs(
    result: SimulationResult,
    forcing_df: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "1",
) -> dict[str, Path]:
    """Write hydrologic outputs in the CSV shape expected by nitrogen demos.

    `states<prefix>.csv` stores hydrologic storages in meters and
    `fluxes<prefix>.csv` stores hydrologic fluxes in meters per second, matching
    the chucho-branch notebook that converts those values to nitrogen units.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    discharge_path = output_path / f"discharge{prefix}.csv"
    states_path = output_path / f"states{prefix}.csv"
    fluxes_path = output_path / f"fluxes{prefix}.csv"
    forcing_path = output_path / "south_fork_aorc_forcing.csv"

    discharge_output = result.discharge_cms.rename("discharge_cms").reset_index()
    discharge_output = discharge_output.rename(columns={discharge_output.columns[0]: "time"})
    discharge_output.to_csv(discharge_path, index=False)
    _with_time_column(result.states).to_csv(states_path, index=False)
    _with_time_column(result.fluxes).to_csv(fluxes_path, index=False)

    forcing_output = forcing_df.copy()
    if "time" not in forcing_output.columns:
        forcing_output.insert(0, "time", forcing_output.index)
    forcing_output.to_csv(forcing_path, index=False)

    return {
        "discharge": discharge_path,
        "states": states_path,
        "fluxes": fluxes_path,
        "forcing": forcing_path,
    }


def convert_states_to_nitrogen_units(states_df: pd.DataFrame) -> pd.DataFrame:
    """Convert hydrologic state storages from meters to millimeters."""
    output = states_df.copy()
    state_columns = ["s_sn", "s_s", "s_gwa", "s_gwp"]
    output[state_columns] = output[state_columns] * MILLIMETERS_PER_METER
    return output


def convert_fluxes_to_nitrogen_units(fluxes_df: pd.DataFrame) -> pd.DataFrame:
    """Convert hydrologic fluxes from meters per second to millimeters per day."""
    output = fluxes_df.copy()
    flux_columns = [
        "p_sn",
        "f_sm",
        "p_r",
        "e_a",
        "q_sc",
        "q_sgwa",
        "q_gwatd",
        "q_gwac",
        "q_gwap",
        "q_gwpc",
    ]
    output[flux_columns] = (
        output[flux_columns] * MILLIMETERS_PER_METER * SECONDS_PER_DAY
    )
    return output
