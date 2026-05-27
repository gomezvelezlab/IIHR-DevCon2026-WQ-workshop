"""Workflow facade for nitrogen model runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Mapping

import pandas as pd
from numpy.typing import NDArray

from .single_cv import NitrogenModel_SingleCV
from .types import NitrogenParameters, NitrogenStates, coerce_nitrogen_parameters


class Nitrogen:
    """Workflow facade around the single-control-volume nitrogen model."""

    def __init__(
        self,
        *,
        output_dir: str | Path = "demo_outputs",
        params: NitrogenParameters | Mapping[str, float] | None = None,
        initial_states: NitrogenStates | None = None,
        initial_masses: NDArray[Any] | None = None,
        df_forcings: pd.DataFrame | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.params = coerce_nitrogen_parameters(params)
        self.model = NitrogenModel_SingleCV(self.params)
        self.df_forcings = df_forcings
        self.solution_ads: pd.DataFrame | None = None
        self.solution_no_ads: pd.DataFrame | None = None
        self.mass_fluxes: pd.DataFrame | None = None
        self.initial_states = initial_states
        self.initial_masses = initial_masses

    def config(
        self,
        *,
        output_dir: str | Path | None = None,
        params: NitrogenParameters | Mapping[str, float] | None = None,
        initial_states: NitrogenStates | None = None,
        initial_masses: NDArray[Any] | None = None,
        df_forcings: pd.DataFrame | None = None,
    ) -> "Nitrogen":
        """Update workflow configuration and return this instance."""
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        if params is not None:
            self.params = coerce_nitrogen_parameters(params)
            self.model = NitrogenModel_SingleCV(self.params)
        if initial_states is not None:
            self.initial_states = initial_states
            self.initial_masses = None
        if initial_masses is not None:
            self.initial_masses = initial_masses
            self.initial_states = NitrogenStates.from_array(initial_masses)
        if df_forcings is not None:
            self.df_forcings = df_forcings
        return self

    def load_hydrology(self, output_dir: str | Path, artifact_names: Any | None = None) -> "Nitrogen":
        """Load exported hydrology artifacts and build nitrogen forcings."""
        from devcon2026.hydrology import HydrologyArtifactNames

        output_path = Path(output_dir)
        names = artifact_names or HydrologyArtifactNames()
        states = pd.read_csv(output_path / names.states, parse_dates=["time"])
        fluxes = pd.read_csv(output_path / names.fluxes, parse_dates=["time"])
        forcing = pd.read_csv(output_path / names.forcing, parse_dates=["time"])
        self.df_forcings = self.from_hydrology_outputs(states, fluxes, forcing)
        return self

    def from_hydrology_outputs(
        self,
        states: pd.DataFrame,
        fluxes: pd.DataFrame,
        forcing: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the forcing dataframe expected by the nitrogen model."""
        from devcon2026.hydrology.export import convert_fluxes_to_nitrogen_units
        from devcon2026.hydrology.export import convert_states_to_nitrogen_units

        states_mm = convert_states_to_nitrogen_units(states)
        fluxes_mm_day = convert_fluxes_to_nitrogen_units(fluxes)
        time = pd.DatetimeIndex(fluxes["time"])
        df_forcings = pd.DataFrame(
            {
                "time": time,
                "doy": time.dayofyear + time.hour / 24.0,
                "temp": forcing["TMP_2maboveground"].to_numpy() - 273.15,
                "s": states_mm["s_s"].to_numpy(),
                "q_in_1": fluxes_mm_day["p_r"].to_numpy(),
                "q_in_2": fluxes_mm_day["f_sm"].to_numpy(),
                "q_out_1": fluxes_mm_day["q_sc"].to_numpy(),
                "q_out_2": fluxes_mm_day["q_sgwa"].to_numpy(),
                "c_din_in_0": 1.0,
                "c_din_in_1": 0.5,
                "c_don_in_0": 0.0,
                "c_don_in_1": 0.0,
            }
        )
        self.df_forcings = df_forcings
        return df_forcings

    def default_initial_states(self) -> NitrogenStates:
        """Infer default initial nitrogen states from configured forcings."""
        if self.df_forcings is None:
            raise RuntimeError("Nitrogen.load_hydrology() must run before initial states are inferred.")
        mean_storage = float(self.df_forcings["s"].mean())
        return NitrogenStates.from_mean_storage(mean_storage)

    def default_initial_masses(self) -> NDArray[Any]:
        """Initial [DON, DIN, SON, FON, DON adsorbed] masses in kg N/km2."""
        return self.default_initial_states().to_array()

    def solve(self, *, progress: bool = True) -> "Nitrogen":
        """Run nitrogen simulations with and without DON adsorption."""
        if self.df_forcings is None:
            raise RuntimeError("Nitrogen.load_hydrology() must run before solve().")
        if self.initial_states is not None:
            masses0 = self.initial_states.to_array()
        elif self.initial_masses is not None:
            masses0 = self.initial_masses
        else:
            self.initial_states = self.default_initial_states()
            masses0 = self.initial_states.to_array()

        self.solution_ads = self.model.simulate_nitrogen_dynamics(
            df_forcings=self.df_forcings,
            M0=masses0,
            with_DON_ads=True,
            progress=progress,
            progress_desc="nitrogen with DON adsorption",
        )
        self.solution_no_ads = self.model.simulate_nitrogen_dynamics(
            df_forcings=self.df_forcings,
            M0=masses0,
            with_DON_ads=False,
            progress=progress,
            progress_desc="nitrogen without DON adsorption",
        )
        self.mass_fluxes = self.model.get_mass_fluxes_all_species(
            M=self.solution_ads[["m_don", "m_din", "m_son", "m_fon"]].to_numpy(),
            df_forcings=self.df_forcings,
        )
        self.mass_fluxes.insert(0, "time", self.df_forcings["time"].to_numpy())
        return self

    def export(self) -> dict[str, Path]:
        """Write nitrogen solution and flux tables to CSV files."""
        if self.solution_ads is None or self.solution_no_ads is None or self.mass_fluxes is None:
            raise RuntimeError("Nitrogen.solve() must run before export().")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "solution_with_adsorption": self.output_dir / "nitrogen_solution_with_adsorption.csv",
            "solution_without_adsorption": self.output_dir / "nitrogen_solution_without_adsorption.csv",
            "mass_fluxes": self.output_dir / "nitrogen_mass_fluxes.csv",
        }
        self.solution_ads.to_csv(paths["solution_with_adsorption"], index=False)
        self.solution_no_ads.to_csv(paths["solution_without_adsorption"], index=False)
        self.mass_fluxes.to_csv(paths["mass_fluxes"], index=False)
        return paths
