"""Three-compartment nitrogen routing coupled to hydrology outputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from devcon2026.hydrology.export import convert_fluxes_to_nitrogen_units
from devcon2026.hydrology.export import convert_states_to_nitrogen_units

from .single_cv import NitrogenSoilLayer
from .types import NitrogenParameters, NitrogenStates, coerce_nitrogen_parameters


def _concentration(mass: float, storage: float, min_storage: float = 0.0) -> float:
    return mass / storage if storage > min_storage else 0.0


class NitrogenThreeCompartment:
    """Route nitrogen through soil, active GW, and passive GW compartments.

    The soil compartment uses the full soil nitrogen model. Active and passive
    groundwater compartments carry dissolved DON/DIN only; solid pools,
    adsorption, plant uptake, and external land-surface sources are disabled
    there by construction.
    """

    def __init__(
        self,
        params: NitrogenParameters | Mapping[str, float] | None = None,
    ) -> None:
        self.params = coerce_nitrogen_parameters(params)
        self.soil = NitrogenSoilLayer(self.params)

    def from_hydrology_outputs(
        self,
        states: pd.DataFrame,
        fluxes: pd.DataFrame,
        forcing: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build three-compartment forcings from hydrology outputs."""
        states_mm = convert_states_to_nitrogen_units(states)
        fluxes_mm_day = convert_fluxes_to_nitrogen_units(fluxes)
        time = pd.DatetimeIndex(fluxes["time"])
        return pd.DataFrame(
            {
                "time": time,
                "doy": time.dayofyear + time.hour / 24.0,
                "temp": forcing["TMP_2maboveground"].to_numpy() - 273.15,
                "s_soil": states_mm["s_s"].to_numpy(),
                "s_gwa": states_mm["s_gwa"].to_numpy(),
                "s_gwp": states_mm["s_gwp"].to_numpy(),
                "q_rain": fluxes_mm_day["p_r"].to_numpy(),
                "q_snowmelt": fluxes_mm_day["f_sm"].to_numpy(),
                "q_sc": fluxes_mm_day["q_sc"].to_numpy(),
                "q_sgwa": fluxes_mm_day["q_sgwa"].to_numpy(),
                "q_gwatd": fluxes_mm_day["q_gwatd"].to_numpy(),
                "q_gwac": fluxes_mm_day["q_gwac"].to_numpy(),
                "q_gwap": fluxes_mm_day["q_gwap"].to_numpy(),
                "q_gwpc": fluxes_mm_day["q_gwpc"].to_numpy(),
                "c_din_in_0": 1.0,
                "c_din_in_1": 0.5,
                "c_don_in_0": 0.0,
                "c_don_in_1": 0.0,
                "source_din": 0.0,
                "source_don": 0.0,
                "source_son": 0.0,
                "source_fon": 0.0,
            }
        )

    def default_initial_masses(self, df_forcings: pd.DataFrame) -> NDArray[Any]:
        """Default [soil pools, soil adsorbed DON, GW dissolved pools]."""
        mean_storage = float(df_forcings["s_soil"].mean())
        soil = NitrogenStates.from_mean_storage(mean_storage).to_array()
        return np.array([*soil, 0.0, 0.0, 0.0, 0.0], dtype=float)

    def _derivatives(
        self,
        y: NDArray[Any],
        forcings: dict[str, Any],
    ) -> NDArray[Any]:
        soil_m = np.maximum(y[0:4], 0.0)
        gwa_don, gwa_din, gwp_don, gwp_din = np.maximum(y[4:8], 0.0)

        soil_d = self.soil.get_derivatives_all_species(
            M=soil_m.copy(),
            s=forcings["s_soil"],
            q_in=forcings["soil_q_in"],
            q_out=forcings["soil_q_out"],
            c_don_in=forcings["c_don_in"],
            c_din_in=forcings["c_din_in"],
            temp=forcings["temp"],
            source_din=forcings["source_din"],
            source_don=forcings["source_don"],
            source_son=forcings["source_son"],
            source_fon=forcings["source_fon"],
        )

        soil_dry_threshold = self.params.min_dissolved_storage
        c_soil_don = _concentration(
            soil_m[0], forcings["s_soil"], soil_dry_threshold
        )
        c_soil_din = _concentration(
            soil_m[1], forcings["s_soil"], soil_dry_threshold
        )
        c_gwa_don = _concentration(gwa_don, forcings["s_gwa"])
        c_gwa_din = _concentration(gwa_din, forcings["s_gwa"])
        c_gwp_don = _concentration(gwp_don, forcings["s_gwp"])
        c_gwp_din = _concentration(gwp_din, forcings["s_gwp"])

        soil_to_gwa = forcings["q_sgwa"]
        gwa_out = forcings["q_gwatd"] + forcings["q_gwac"] + forcings["q_gwap"]
        gwa_to_gwp = forcings["q_gwap"]
        gwp_out = forcings["q_gwpc"]

        d_gwa_don = soil_to_gwa * c_soil_don - gwa_out * c_gwa_don
        d_gwa_din = soil_to_gwa * c_soil_din - gwa_out * c_gwa_din
        d_gwp_don = gwa_to_gwp * c_gwa_don - gwp_out * c_gwp_don
        d_gwp_din = gwa_to_gwp * c_gwa_din - gwp_out * c_gwp_din
        return np.array([*soil_d, d_gwa_don, d_gwa_din, d_gwp_don, d_gwp_din])

    def simulate(
        self,
        df_forcings: pd.DataFrame,
        initial_masses: NDArray[Any] | None = None,
        *,
        with_soil_don_adsorption: bool = True,
        progress: bool = True,
        progress_desc: str | None = None,
    ) -> pd.DataFrame:
        """Simulate nitrogen through soil, active GW, and passive GW."""
        df_forcings = df_forcings.reset_index(drop=True)
        time_index = pd.DatetimeIndex(df_forcings["time"])
        y0 = (
            self.default_initial_masses(df_forcings)
            if initial_masses is None
            else initial_masses.astype(float)
        )
        y = y0[[0, 1, 2, 3, 5, 6, 7, 8]] if len(y0) == 9 else y0[0:8].copy()
        soil_ads = float(y0[4]) if len(y0) >= 5 else 0.0
        delta_m_don = 0.0

        history = [[*y[0:4], soil_ads, delta_m_don, *y[4:8]]]
        iter_time: Iterable[pd.Timestamp] = list(time_index[1:])
        if progress:
            iter_time = tqdm(
                iter_time,
                total=len(time_index[1:]),
                desc=progress_desc or "nitrogen three-compartment",
                unit="step",
            )

        for i, current_time in enumerate(iter_time):
            forcings = {
                "s_soil": float(df_forcings["s_soil"].iloc[i]),
                "s_gwa": float(df_forcings["s_gwa"].iloc[i]),
                "s_gwp": float(df_forcings["s_gwp"].iloc[i]),
                "soil_q_in": df_forcings[["q_rain", "q_snowmelt"]].iloc[i].to_numpy(dtype=float),
                "soil_q_out": df_forcings[["q_sc", "q_sgwa"]].iloc[i].to_numpy(dtype=float),
                "q_sgwa": float(df_forcings["q_sgwa"].iloc[i]),
                "q_gwatd": float(df_forcings["q_gwatd"].iloc[i]),
                "q_gwac": float(df_forcings["q_gwac"].iloc[i]),
                "q_gwap": float(df_forcings["q_gwap"].iloc[i]),
                "q_gwpc": float(df_forcings["q_gwpc"].iloc[i]),
                "c_don_in": df_forcings[["c_don_in_0", "c_don_in_1"]].iloc[i].to_numpy(dtype=float),
                "c_din_in": df_forcings[["c_din_in_0", "c_din_in_1"]].iloc[i].to_numpy(dtype=float),
                "temp": float(df_forcings["temp"].iloc[i]),
                "source_din": float(df_forcings.get("source_din", pd.Series(0.0, index=df_forcings.index)).iloc[i]),
                "source_don": float(df_forcings.get("source_don", pd.Series(0.0, index=df_forcings.index)).iloc[i]),
                "source_son": float(df_forcings.get("source_son", pd.Series(0.0, index=df_forcings.index)).iloc[i]),
                "source_fon": float(df_forcings.get("source_fon", pd.Series(0.0, index=df_forcings.index)).iloc[i]),
            }
            sol = solve_ivp(  # type: ignore[call-overload]
                lambda _t, state: self._derivatives(state, forcings),
                (0.0, self.params.delta_time_solver),
                y,
                method="LSODA",
            )
            if not sol.success:
                raise RuntimeError(f"Solver failed at {current_time}: {sol.message}")

            y = np.maximum(0.0, sol.y[:, -1]).astype(float)
            if with_soil_don_adsorption:
                soil_storage = cast(float, forcings["s_soil"])
                m_don_new, soil_ads_new, delta_m_don = (
                    self.soil.get_don_mass_balance_equilibrium_adjustment(
                        m_don_current=float(y[0]),
                        m_don_ads_previous=soil_ads,
                        s=soil_storage,
                        params=self.soil.params,
                    )
                )
                y[0] = m_don_new
                soil_ads = soil_ads_new
            history.append([*y[0:4], soil_ads, delta_m_don, *y[4:8]])

        output = pd.DataFrame(
            history,
            columns=[
                "soil_m_don",
                "soil_m_din",
                "soil_m_son",
                "soil_m_fon",
                "soil_m_don_ads",
                "soil_delta_m_don",
                "gwa_m_don",
                "gwa_m_din",
                "gwp_m_don",
                "gwp_m_din",
            ],
        )
        output.insert(0, "time", time_index)
        output["soil_s"] = df_forcings["s_soil"].to_numpy()
        output["gwa_s"] = df_forcings["s_gwa"].to_numpy()
        output["gwp_s"] = df_forcings["s_gwp"].to_numpy()
        dry_thresholds = {
            "soil": self.params.min_dissolved_storage,
            "gwa": 0.0,
            "gwp": 0.0,
        }
        for prefix in ("soil", "gwa", "gwp"):
            storage = output[f"{prefix}_s"]
            output[f"{prefix}_c_don"] = np.divide(
                output[f"{prefix}_m_don"],
                storage,
                out=np.zeros(len(output), dtype=float),
                where=storage.to_numpy() > dry_thresholds[prefix],
            )
            output[f"{prefix}_c_din"] = np.divide(
                output[f"{prefix}_m_din"],
                storage,
                out=np.zeros(len(output), dtype=float),
                where=storage.to_numpy() > dry_thresholds[prefix],
            )
        return output
