"""Simulation loop."""

from __future__ import annotations

from dataclasses import asdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from .constants import MILLIMETERS_PER_HOUR_TO_METERS_PER_SECOND
from .physics import compute_fluxes, route_flux_to_discharge, solve_ivp_fun
from .types import HydrologyFluxes, HydrologyForcings, HydrologyParameters
from .types import HydrologySimulationResult, HydrologyStates


def simulate(
    forcing_df: pd.DataFrame,
    params: HydrologyParameters,
    initial_states: HydrologyStates,
    progress: bool = False,
    progress_desc: str | None = None,
) -> HydrologySimulationResult:
    """Run the hydrologic model over forcing time steps and return outputs."""
    y: NDArray[np.float64] = initial_states.to_array()
    flux_history: list[HydrologyFluxes] = []
    states_history: list[HydrologyStates] = []
    time_index = pd.DatetimeIndex(forcing_df["time"])
    if len(time_index) >= 2:
        time_step_seconds = (time_index[1] - time_index[0]).total_seconds()
        if time_step_seconds != 3600.0:
            raise ValueError(
                f"Expected hourly forcing (3600 s), found {time_step_seconds} s."
            )
    precipitation_mm = forcing_df["precipitation_mm"].to_numpy(dtype=float)  # [mm / timestep]
    temperature_c = forcing_df["temperature_2m_C"].to_numpy(dtype=float)  # [C]
    ref_et_mm_hr = forcing_df["ref_et_mm_hr"].to_numpy(dtype=float)  # [mm hr-1]
    iter_time: Iterable[pd.Timestamp] = time_index
    if progress:
        iter_time = tqdm(
            time_index,
            total=len(time_index),
            desc=progress_desc or "simulate",
            unit="step",
        )

    for i, current_time in enumerate(iter_time):
        forcings = HydrologyForcings(
            p_t=float(precipitation_mm[i]) * MILLIMETERS_PER_HOUR_TO_METERS_PER_SECOND,  # [m s-1]
            t=float(temperature_c[i]),  # [C]
            e_p=float(ref_et_mm_hr[i]) * MILLIMETERS_PER_HOUR_TO_METERS_PER_SECOND,  # [m s-1]
        )
        if not np.isfinite([forcings.p_t, forcings.t, forcings.e_p]).all():
            raise ValueError(f"Non-finite forcing values at {current_time}.")
        t_start = current_time.timestamp()
        t_end = (current_time + pd.Timedelta(hours=1)).timestamp()

        sol = solve_ivp(
            solve_ivp_fun,
            (t_start, t_end),
            y,
            args=(params, forcings),
            method="DOP853",
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed at {current_time}: {sol.message}")

        y = np.maximum(params.min_storage, sol.y[:, -1]).astype(np.float64)

        current_state = HydrologyStates.from_array(y)
        states_history.append(current_state)
        flux_history.append(compute_fluxes(t_end, current_state, params, forcings))

    fluxes_into_channel = np.array(
        [f.fluxes_into_channel() for f in flux_history], dtype=float
    )
    discharge_cms = pd.Series(
        route_flux_to_discharge(fluxes_into_channel, params),  # [m3 s-1]
        index=time_index,
        name="discharge_cms",
    )

    states_df = pd.DataFrame([asdict(s) for s in states_history], index=time_index)
    fluxes_df = pd.DataFrame([asdict(f) for f in flux_history], index=time_index)

    return HydrologySimulationResult(
        discharge_cms=discharge_cms, states=states_df, fluxes=fluxes_df
    )
