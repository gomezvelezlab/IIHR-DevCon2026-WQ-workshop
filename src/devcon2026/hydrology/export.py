"""Export hydrologic model outputs for nitrogen-model workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import MILLIMETERS_PER_METER
from .constants import SECONDS_PER_DAY
from .types import HydrologySimulationResult


@dataclass(frozen=True)
class HydrologyArtifactNames:
    """Artifact filenames shared by hydrology exports and nitrogen imports."""

    discharge: str = "discharge1.parquet"
    states: str = "states1.parquet"
    fluxes: str = "fluxes1.parquet"
    forcing: str = "south_fork_aorc_forcing.parquet"


def read_table(path: str | Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Read a dataframe artifact from Parquet or CSV."""
    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path)
    return pd.read_csv(table_path, parse_dates=parse_dates)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a dataframe artifact as Parquet or CSV based on suffix."""
    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        df.to_parquet(table_path, engine="fastparquet", index=False)
        return
    df.to_csv(table_path, index=False)


def _with_time_column(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.insert(0, "time", output.index)
    return output.reset_index(drop=True)


def export_nitrogen_hydrology_inputs(
    result: HydrologySimulationResult,
    forcing_df: pd.DataFrame,
    output_dir: str | Path,
    artifact_names: HydrologyArtifactNames | None = None,
    prefix: str = "1",
) -> dict[str, Path]:
    """Write hydrologic outputs in the shape expected by nitrogen demos.

    `states<prefix>` stores hydrologic storages in meters and
    `fluxes<prefix>` stores hydrologic fluxes in meters per second, matching
    the chucho-branch notebook that converts those values to nitrogen units.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    names = artifact_names or HydrologyArtifactNames(
        discharge=f"discharge{prefix}.parquet",
        states=f"states{prefix}.parquet",
        fluxes=f"fluxes{prefix}.parquet",
    )

    discharge_path = output_path / names.discharge
    states_path = output_path / names.states
    fluxes_path = output_path / names.fluxes
    forcing_path = output_path / names.forcing

    discharge_output = result.discharge_cms.rename("discharge_cms").reset_index()
    discharge_output = discharge_output.rename(columns={discharge_output.columns[0]: "time"})
    write_table(discharge_output, discharge_path)
    write_table(_with_time_column(result.states), states_path)
    write_table(_with_time_column(result.fluxes), fluxes_path)

    forcing_output = forcing_df.copy()
    if "time" not in forcing_output.columns:
        forcing_output.insert(0, "time", forcing_output.index)
    write_table(forcing_output, forcing_path)

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
