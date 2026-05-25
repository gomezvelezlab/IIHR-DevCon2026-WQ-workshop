"""Workflow facade for hydrologic model runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .export import HydrologyArtifactNames, export_nitrogen_hydrology_inputs
from .simulation import simulate
from .types import Parameters, SimulationResult, States


def synthetic_hydrology_forcing(
    start: str | pd.Timestamp = "2010-01-01",
    hours: int = 24 * 120,
) -> pd.DataFrame:
    """Create model-ready synthetic forcing data for demos."""
    time = pd.date_range(start, periods=hours, freq="h")
    hour = np.arange(hours, dtype=float)
    seasonal = np.sin(2.0 * np.pi * hour / (24.0 * 365.0))
    storm = np.maximum(0.0, np.sin(2.0 * np.pi * hour / (24.0 * 9.0))) ** 4

    return pd.DataFrame(
        {
            "time": time,
            "TMP_2maboveground": 273.15 + 9.0 + 14.0 * seasonal,
            "temperature_2m_C": 9.0 + 14.0 * seasonal,
            "precipitation_mm": 5.0 * storm,
            "ref_et_mm_hr": 0.05 + 0.03 * np.maximum(0.0, seasonal),
        }
    )


@dataclass
class Hydrology:
    """Stateful workflow facade around the hydrologic model functions."""

    output_dir: Path = Path("demo_outputs/example_hydrology_model")
    artifact_names: HydrologyArtifactNames = field(default_factory=HydrologyArtifactNames)
    params: Parameters = field(default_factory=Parameters)
    initial_states: States = field(
        default_factory=lambda: States(s_sn=0.01, s_s=0.03, s_gwa=0.2, s_gwp=0.5)
    )
    forcing_df: pd.DataFrame | None = None
    start: str | pd.Timestamp = "2010-01-01"
    hours: int = 24 * 120
    result: SimulationResult | None = None
    source: str = "not solved"

    def config(
        self,
        *,
        output_dir: str | Path | None = None,
        artifact_names: HydrologyArtifactNames | None = None,
        params: Parameters | None = None,
        initial_states: States | None = None,
        forcing_df: pd.DataFrame | None = None,
        start: str | pd.Timestamp | None = None,
        hours: int | None = None,
    ) -> Hydrology:
        """Update workflow configuration and return this instance."""
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        if artifact_names is not None:
            self.artifact_names = artifact_names
        if params is not None:
            self.params = params
        if initial_states is not None:
            self.initial_states = initial_states
        if forcing_df is not None:
            self.forcing_df = forcing_df
        if start is not None:
            self.start = start
        if hours is not None:
            self.hours = hours
        return self

    @property
    def required_output_paths(self) -> list[Path]:
        return [
            self.output_dir / self.artifact_names.discharge,
            self.output_dir / self.artifact_names.states,
            self.output_dir / self.artifact_names.fluxes,
            self.output_dir / self.artifact_names.forcing,
        ]

    def cache_exists(self) -> bool:
        return all(path.exists() for path in self.required_output_paths)

    def solve(self, *, use_cache: bool = True, force: bool = False, progress: bool = True) -> Hydrology:
        """Run the hydrologic model unless exported outputs can be reused."""
        if use_cache and not force and self.cache_exists():
            self.source = "loaded from existing CSVs"
            return self

        generated_synthetic = self.forcing_df is None
        if generated_synthetic:
            self.forcing_df = synthetic_hydrology_forcing(start=self.start, hours=self.hours)
        forcing_df = self.forcing_df
        if forcing_df is None:
            raise RuntimeError("Hydrology forcing data was not configured.")
        self.result = simulate(
            forcing_df=forcing_df,
            params=self.params,
            initial_states=self.initial_states,
            progress=progress,
            progress_desc="hydrology",
        )
        self.source = (
            "generated from synthetic forcing"
            if generated_synthetic
            else "generated from configured forcing"
        )
        return self

    def export(self) -> dict[str, Path]:
        """Export solved hydrology outputs for downstream nitrogen workflows."""
        if self.result is None or self.forcing_df is None:
            if self.cache_exists():
                return {
                    "states": self.output_dir / self.artifact_names.states,
                    "fluxes": self.output_dir / self.artifact_names.fluxes,
                    "forcing": self.output_dir / self.artifact_names.forcing,
                    "discharge": self.output_dir / self.artifact_names.discharge,
                }
            raise RuntimeError("Hydrology.solve() must run before export().")
        return export_nitrogen_hydrology_inputs(
            self.result,
            self.forcing_df,
            self.output_dir,
            artifact_names=self.artifact_names,
        )

    def load_outputs(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load exported states, fluxes, and forcing dataframes."""
        if not self.cache_exists():
            raise FileNotFoundError(f"Missing hydrology outputs in {self.output_dir}.")
        states = pd.read_csv(self.output_dir / self.artifact_names.states, parse_dates=["time"])
        fluxes = pd.read_csv(self.output_dir / self.artifact_names.fluxes, parse_dates=["time"])
        forcing = pd.read_csv(
            self.output_dir / self.artifact_names.forcing,
            parse_dates=["time"],
        )
        return states, fluxes, forcing
