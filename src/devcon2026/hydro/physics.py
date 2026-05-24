"""Hydrologic process equations and routing."""

from __future__ import annotations

import calendar
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import scipy.special
from numpy.typing import NDArray

from .constants import SECONDS_PER_HOUR, STEFAN_BOLTZMANN_CONSTANT
from .types import Fluxes, Forcings, Parameters, States


def gamma_hydrograph(t: NDArray[Any], n: float, a: float) -> NDArray[Any]:
    """Gamma unit hydrograph evaluated at time axis ``t``."""
    t_non_neg = np.maximum(t, 0.0)
    return (t_non_neg ** (n - 1) * np.exp(-t_non_neg / a)) / (
        a**n * scipy.special.gamma(n)
    )


def water_year_day(dt: datetime) -> int:
    """Return zero-based water-year day (Oct 1 -> 0)."""
    wy_start_year = dt.year if dt.month >= 10 else dt.year - 1
    wy_start = datetime(wy_start_year, 10, 1, tzinfo=timezone.utc)
    return (dt - wy_start).days


def gamma_1(timestamp: float, params: Parameters) -> float:
    """Seasonal infiltration partition coefficient.

    Suggested calibration setup if this form is retained:
    - Use coupled constraints on ``gamma_x`` and ``gamma_i`` rather than
      independent box bounds.
    - To keep the full seasonal curve in approximately [0.02, 0.98], require
      ``gamma_x + gamma_i <= log10(0.98)`` and
      ``gamma_x - gamma_i >= log10(0.02)``.
    - ``gamma_p`` can still vary independently over [0, 366].
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    day_index = water_year_day(dt)
    ndays = 365 + int(calendar.isleap(dt.year))
    exponent = params.gamma_x + params.gamma_i * np.sin(
        2 * np.pi * (day_index + params.gamma_p) / ndays
    )
    return float(10**exponent)


def gamma_2(timestamp: float, params: Parameters) -> float:
    """Bounded seasonal infiltration partition coefficient.

    Suggested calibration setup:
    - ``gamma_x`` in [-2, 2]
    - ``gamma_i`` in [0, 2]
    - ``gamma_p`` in [0, 366]

    Those box bounds let the logistic seasonal signal explore roughly
    [0.02, 0.98] without any coupled constraints.
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    day_index = water_year_day(dt)
    ndays = 365 + int(calendar.isleap(dt.year))
    raw = params.gamma_x + params.gamma_i * np.sin(
        2 * np.pi * (day_index + params.gamma_p) / ndays
    )
    return float(1.0 / (1.0 + np.exp(-raw)))


def penman_monteith_hourly_asce(
    air_temperature: float | NDArray[Any],
    wind_speed_at_2m: float | NDArray[Any],
    incoming_shortwave_radiation: float | NDArray[Any],
    incoming_longwave_radiation: float | NDArray[Any],
    atmospheric_pressure: float | NDArray[Any],
    actual_vapor_pressure: float | NDArray[Any],
    params: Parameters,
) -> float | NDArray[Any]:
    """Hourly ASCE Penman-Monteith reference ET [mm/hr]."""
    wm2_to_mj_hr = 0.0036  # [MJ hr-1] per [W]

    pressure_kpa = atmospheric_pressure / 1000.0  # [kPa]
    ea_kpa = actual_vapor_pressure / 1000.0  # [kPa]
    temp_k = air_temperature + 273.16  # [K]

    rad_ns = (1.0 - params.pet_albedo) * incoming_shortwave_radiation  # [W m-2]
    rad_l_out = params.pet_emissivity * STEFAN_BOLTZMANN_CONSTANT * temp_k**4  # [W m-2]
    rad_n = rad_ns + incoming_longwave_radiation - rad_l_out  # [W m-2]

    daytime = rad_n > 0
    g_heat = np.where(daytime, 0.1 * rad_n, 0.5 * rad_n)  # [W m-2]

    rad_n_mj = rad_n * wm2_to_mj_hr  # [MJ m-2 hr-1]
    g_heat_mj = g_heat * wm2_to_mj_hr  # [MJ m-2 hr-1]

    es = 0.6108 * np.exp((17.27 * air_temperature) / (air_temperature + 237.3))  # [kPa]
    delta = 4098.0 * es / (air_temperature + 237.3) ** 2  # [kPa C-1]
    gamma_ = 0.000665 * pressure_kpa  # [kPa C-1]

    c_n = 37.0
    c_d = np.where(daytime, 0.24, 0.96)

    numerator = 0.408 * delta * (rad_n_mj - g_heat_mj) + gamma_ * (  # [mm hr-1]
        c_n / temp_k
    ) * wind_speed_at_2m * (es - ea_kpa)
    denominator = delta + gamma_ * (1.0 + c_d * wind_speed_at_2m)  # [kPa C-1]

    return np.maximum(numerator / denominator, 0.0)  # [mm hr-1]


def compute_extra_vars(
    forcing_df: pd.DataFrame,
    params: Parameters,
) -> pd.DataFrame:
    """Derive model-ready forcing variables from raw AORC columns."""
    df = forcing_df.copy()
    df["temperature_2m_K"] = df["TMP_2maboveground"]  # [K]
    df["precipitation_mm"] = df["APCP_surface"]  # [mm / timestep]
    df["wind_u_10m_ms"] = df["UGRD_10maboveground"]  # [m s-1]
    df["wind_v_10m_ms"] = df["VGRD_10maboveground"]  # [m s-1]
    df["shortwave_down_Wm2"] = df["DSWRF_surface"]  # [W m-2]
    df["longwave_down_Wm2"] = df["DLWRF_surface"]  # [W m-2]
    df["pressure_Pa"] = df["PRES_surface"]  # [Pa]
    df["specific_humidity_kgkg"] = df["SPFH_2maboveground"]  # [kg kg-1]

    df["temperature_2m_C"] = df["temperature_2m_K"] - 273.15  # [C]

    wind_speed_10m = np.sqrt(df["wind_u_10m_ms"] ** 2 + df["wind_v_10m_ms"] ** 2)  # [m s-1]
    df["wind_speed_10m_ms"] = wind_speed_10m  # [m s-1]
    df["wind_speed_2m_ms"] = wind_speed_10m * 4.87 / np.log(67.8 * 10 - 5.42)  # [m s-1]

    q = df["specific_humidity_kgkg"]  # [kg kg-1]
    df["vapor_pressure_Pa"] = q * df["pressure_Pa"] / (0.622 + 0.378 * q)  # [Pa]

    df["ref_et_mm_hr"] = penman_monteith_hourly_asce(
        df["temperature_2m_C"].to_numpy(),
        df["wind_speed_2m_ms"].to_numpy(),
        df["shortwave_down_Wm2"].to_numpy(),
        df["longwave_down_Wm2"].to_numpy(),
        df["pressure_Pa"].to_numpy(),
        df["vapor_pressure_Pa"].to_numpy(),
        params,
    )
    return df


def compute_fluxes(
    t: float, states: States, params: Parameters, forcings: Forcings
) -> Fluxes:
    """Compute all process fluxes for a single model state and forcing."""
    s_sn = max(0.0, states.s_sn)  # [m]
    s_s = max(0.0, states.s_s)  # [m]
    s_gwa = max(0.0, states.s_gwa)  # [m]
    s_gwp = max(0.0, states.s_gwp)  # [m]

    p_sn = np.where(forcings.t > params.t_0, 0.0, forcings.p_t)  # [m s-1]
    p_r = forcings.p_t - p_sn  # [m s-1]

    f_sm_pot = max(0.0, params.k_sn * forcings.t)  # [m s-1]
    f_sm = f_sm_pot * (1 - np.exp(-(s_sn / params.m_sn)))  # [m s-1]

    e_a = (
        params.c_e
        * forcings.e_p
        * ((s_s / params.s_max) * (1 + params.m_s))
        / ((s_s / params.s_max) + params.m_s)
    )  # [m s-1]

    soil_wetness = (s_s / params.s_max) ** params.beta_s
    q_inf = (p_r + f_sm) * soil_wetness  # [m s-1]
    gamma_s = gamma_2(t, params)
    q_sc = gamma_s * q_inf  # [m s-1]
    q_sgwv = params.k_sgw * soil_wetness  # [m s-1]
    q_sgwa = (1.0 - gamma_s) * q_inf + q_sgwv  # [m s-1]

    q_gwac = params.k_gwac * (s_gwa / params.s_gwa_max) ** params.beta_gwac  # [m s-1]
    # Relative-threshold form:
    # q_gwatd = k_td * max(0, s_gwa / s_gwa_max - s_ref_td) * s_gwa_max
    # Equivalent implementation with fewer operations:
    q_gwatd = params.k_td * max(0.0, s_gwa - params.s_ref_td * params.s_gwa_max)  # [m s-1]
    q_gwap = params.k_gwap * (s_gwa / params.s_gwa_max)  # [m s-1]
    q_gwpc = params.k_gwpc * s_gwp  # [m s-1]

    return Fluxes(
        p_sn=float(p_sn),
        f_sm=float(f_sm),
        p_r=float(p_r),
        e_a=float(e_a),
        q_sc=float(q_sc),
        q_sgwa=float(q_sgwa),
        q_gwatd=float(q_gwatd),
        q_gwac=float(q_gwac),
        q_gwap=float(q_gwap),
        q_gwpc=float(q_gwpc),
    )


def solve_ivp_fun(
    t: float,
    y: NDArray[Any],
    params: Parameters,
    forcings: Forcings,
) -> NDArray[Any]:
    """RHS callback for ``scipy.integrate.solve_ivp``."""
    fluxes = compute_fluxes(t, States.from_array(y), params, forcings)
    return fluxes.compute_derivatives().to_array()


def route_flux_to_discharge(
    fluxes_into_channel: NDArray[Any],
    params: Parameters,
) -> NDArray[Any]:
    """Route channel inflow flux to outlet discharge using gamma UH."""
    time_axis = np.arange(0, len(fluxes_into_channel), dtype=float) * SECONDS_PER_HOUR  # [s]
    unit_hydrograph = gamma_hydrograph(time_axis, params.n_gu, params.a_gu_seconds)
    # Discrete convolution approximates the continuous integral:
    # q_out(t) = integral(q_in(t-tau) * h(tau) d tau), so multiply by dt.
    outlet_flux = np.convolve(fluxes_into_channel, unit_hydrograph)[
        : len(fluxes_into_channel)
    ] * SECONDS_PER_HOUR  # [m s-1]
    return outlet_flux * (params.area_km2 * 1e6)  # [m3 s-1]
