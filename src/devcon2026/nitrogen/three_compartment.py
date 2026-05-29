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

    Each hydrologic compartment uses the single-control-volume nitrogen model.
    Soil and active groundwater include denitrification and plant uptake.
    Passive groundwater includes denitrification but disables plant uptake.
    """

    def __init__(
        self,
        params: NitrogenParameters | Mapping[str, float] | None = None,
        *,
        soil_params: NitrogenParameters | Mapping[str, float] | None = None,
        gwa_params: NitrogenParameters | Mapping[str, float] | None = None,
        gwp_params: NitrogenParameters | Mapping[str, float] | None = None,
    ) -> None:
        base_params = coerce_nitrogen_parameters(params)
        self.soil_params = (
            coerce_nitrogen_parameters(soil_params)
            if soil_params is not None
            else base_params
        )
        self.gwa_params = (
            coerce_nitrogen_parameters(gwa_params)
            if gwa_params is not None
            else base_params
        )
        gwp_base_params = (
            coerce_nitrogen_parameters(gwp_params)
            if gwp_params is not None
            else base_params
        )
        self.gwp_params = gwp_base_params.with_updates({"uptake_demand": 0.0})
        self.params = self.soil_params
        self.soil = NitrogenSoilLayer(self.soil_params)
        self.gwa = NitrogenSoilLayer(self.gwa_params)
        self.gwp = NitrogenSoilLayer(self.gwp_params)

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
        """Default [soil pools, soil adsorbed DON, GWA pools, GWP pools]."""
        mean_storage = float(df_forcings["s_soil"].mean())
        soil = NitrogenStates.from_mean_storage(mean_storage).to_array()
        groundwater = np.zeros(4, dtype=float)
        return np.array([*soil, *groundwater, *groundwater], dtype=float)

    def _coerce_initial_masses(self, y0: NDArray[Any]) -> tuple[NDArray[Any], float]:
        """Return 12 model masses and soil adsorbed DON from old or new layouts."""
        y0 = y0.astype(float)
        if len(y0) >= 13:
            masses = y0[[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]].copy()
            masses[[6, 7, 10, 11]] = 0.0
            return masses, float(y0[4])
        if len(y0) == 12:
            masses = y0.copy()
            masses[[6, 7, 10, 11]] = 0.0
            return masses, 0.0
        if len(y0) == 9:
            soil = y0[0:4]
            soil_ads = float(y0[4])
            gwa = np.array([y0[5], y0[6], 0.0, 0.0], dtype=float)
            gwp = np.array([y0[7], y0[8], 0.0, 0.0], dtype=float)
            return np.array([*soil, *gwa, *gwp], dtype=float), soil_ads
        if len(y0) == 8:
            soil = y0[0:4]
            gwa = np.array([y0[4], y0[5], 0.0, 0.0], dtype=float)
            gwp = np.array([y0[6], y0[7], 0.0, 0.0], dtype=float)
            return np.array([*soil, *gwa, *gwp], dtype=float), 0.0
        raise ValueError(
            "initial_masses must have length 8, 9, 12, or 13 for the three-compartment model."
        )

    def _derivatives(
        self,
        y: NDArray[Any],
        forcings: dict[str, Any],
    ) -> NDArray[Any]:
        soil_m = np.maximum(y[0:4], 0.0)
        gwa_m = np.maximum(y[4:8], 0.0)
        gwp_m = np.maximum(y[8:12], 0.0)

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

        soil_dry_threshold = self.soil.params["min_dissolved_storage"]
        c_soil = np.array(
            [
                _concentration(soil_m[0], forcings["s_soil"], soil_dry_threshold),
                _concentration(soil_m[1], forcings["s_soil"], soil_dry_threshold),
            ],
            dtype=float,
        )
        c_gwa = np.array(
            [
                _concentration(gwa_m[0], forcings["s_gwa"]),
                _concentration(gwa_m[1], forcings["s_gwa"]),
            ],
            dtype=float,
        )

        gwa_d = self.gwa.get_derivatives_all_species(
            M=gwa_m.copy(),
            s=forcings["s_gwa"],
            q_in=np.array([forcings["q_sgwa"]], dtype=float),
            q_out=np.array(
                [
                    forcings["q_gwatd"],
                    forcings["q_gwac"],
                    forcings["q_gwap"],
                ],
                dtype=float,
            ),
            c_don_in=np.array([c_soil[0]], dtype=float),
            c_din_in=np.array([c_soil[1]], dtype=float),
            temp=forcings["temp"],
        )
        gwp_d = self.gwp.get_derivatives_all_species(
            M=gwp_m.copy(),
            s=forcings["s_gwp"],
            q_in=np.array([forcings["q_gwap"]], dtype=float),
            q_out=np.array([forcings["q_gwpc"]], dtype=float),
            c_don_in=np.array([c_gwa[0]], dtype=float),
            c_din_in=np.array([c_gwa[1]], dtype=float),
            temp=forcings["temp"],
        )
        return np.array([*soil_d, *gwa_d, *gwp_d])

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
        y, soil_ads = self._coerce_initial_masses(y0)
        delta_m_don = 0.0

        history = [[*y[0:4], soil_ads, delta_m_don, *y[4:12]]]
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
                (0.0, self.soil_params.delta_time_solver),
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
            history.append([*y[0:4], soil_ads, delta_m_don, *y[4:12]])

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
                "gwa_m_son",
                "gwa_m_fon",
                "gwp_m_don",
                "gwp_m_din",
                "gwp_m_son",
                "gwp_m_fon",
            ],
        )
        output.insert(0, "time", time_index)
        output["soil_s"] = df_forcings["s_soil"].to_numpy()
        output["gwa_s"] = df_forcings["s_gwa"].to_numpy()
        output["gwp_s"] = df_forcings["s_gwp"].to_numpy()
        dry_thresholds = {
            "soil": self.soil.params["min_dissolved_storage"],
            "gwa": self.gwa.params["min_dissolved_storage"],
            "gwp": self.gwp.params["min_dissolved_storage"],
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
