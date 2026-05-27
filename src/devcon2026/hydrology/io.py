"""Input loaders for parameters, forcings, and observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tomli

from .physics import compute_extra_vars
from .types import HydrologyParameters

_FORCING_BASE_CACHE: dict[str, pd.DataFrame] = {}
_OBS_BASE_CACHE: dict[str, pd.DataFrame] = {}


def load_parameters(path: str | Path) -> HydrologyParameters:
    with open(path, "rb") as fp:
        parsed = tomli.load(fp)

    flattened: dict[str, float] = {}
    for _, values in parsed.items():
        flattened.update(values)

    return HydrologyParameters(**flattened)


def load_forcing_data(
    path: str | Path,
    start_time: str | None,
    end_time: str | None,
    params: HydrologyParameters,
) -> pd.DataFrame:
    cache_key = str(path)
    if cache_key not in _FORCING_BASE_CACHE:
        base = pd.read_csv(path, parse_dates=["time"])
        base["time"] = pd.to_datetime(base["time"], utc=True)
        _FORCING_BASE_CACHE[cache_key] = base
    forcing_df = _FORCING_BASE_CACHE[cache_key].copy()
    if start_time is not None:
        forcing_df = forcing_df[
            forcing_df["time"] >= pd.Timestamp(start_time, tz="UTC")
        ]
    if end_time is not None:
        forcing_df = forcing_df[forcing_df["time"] < pd.Timestamp(end_time, tz="UTC")]

    forcing_cols = [
        "APCP_surface",
        "DLWRF_surface",
        "DSWRF_surface",
        "PRES_surface",
        "SPFH_2maboveground",
        "TMP_2maboveground",
        "UGRD_10maboveground",
        "VGRD_10maboveground",
    ]

    missing_rows = forcing_df[forcing_cols].isna().any(axis=1)
    if missing_rows.any():
        first_ts = forcing_df.loc[missing_rows, "time"].iloc[0]
        missing_count = int(missing_rows.sum())
        print(
            "Found missing forcing rows; filling values "
            f"(rows={missing_count}, first={first_ts})."
        )

        # Precipitation is event-like; fill missing with 0.
        forcing_df["APCP_surface"] = forcing_df["APCP_surface"].fillna(0.0)

        # For continuous meteorological variables, use time interpolation.
        continuous_cols = [
            "DLWRF_surface",
            "DSWRF_surface",
            "PRES_surface",
            "SPFH_2maboveground",
            "TMP_2maboveground",
            "UGRD_10maboveground",
            "VGRD_10maboveground",
        ]
        forcing_df[continuous_cols] = forcing_df[continuous_cols].interpolate(
            method="linear",
            axis=0,
            limit_direction="both",
        )

    still_missing = forcing_df[forcing_cols].isna().any(axis=1)
    if still_missing.any():
        bad_ts = forcing_df.loc[still_missing, "time"].iloc[0]
        raise ValueError(f"Forcing data still contains NaNs after fill at {bad_ts}.")

    if not np.isfinite(forcing_df[forcing_cols].to_numpy(dtype=float)).all():
        raise ValueError("Forcing data contains non-finite values (inf/-inf).")

    return compute_extra_vars(forcing_df, params).reset_index(drop=True)


def load_observed_discharge(
    path: str | Path, start_time: str | None, end_time: str | None
) -> pd.Series:
    cache_key = str(path)
    if cache_key not in _OBS_BASE_CACHE:
        obs_df = pd.read_csv(path, parse_dates=["datetime"])
        obs_df["datetime"] = pd.to_datetime(obs_df["datetime"], utc=True)
        _OBS_BASE_CACHE[cache_key] = obs_df
    obs_df = _OBS_BASE_CACHE[cache_key].copy()
    obs_df = obs_df.set_index("datetime").sort_index()
    hourly = obs_df["streamflow_cms"].resample("1h").mean().rename("streamflow_cms")
    if start_time is not None:
        hourly = hourly[hourly.index >= pd.Timestamp(start_time, tz="UTC")]
    if end_time is not None:
        hourly = hourly[hourly.index < pd.Timestamp(end_time, tz="UTC")]
    return hourly
