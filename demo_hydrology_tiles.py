"""Compare legacy, water-table, and no-tile hydrology simulations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from devcon2026.hydrology import Hydrology
from devcon2026.hydrology import HydrologyParameters
from devcon2026.hydrology import HydrologyStates
from devcon2026.hydrology.export import convert_fluxes_to_nitrogen_units
from devcon2026.hydrology.export import convert_states_to_nitrogen_units
from devcon2026.hydrology.io import load_forcing_data
from devcon2026.hydrology.physics import water_table_depth
from devcon2026.tables import read_table

OUTPUT_DIR = Path("demo_outputs/hydrology_tile_comparison")
HYDROLOGY_FORCING_PARQUET = Path("data/hydrology_forcings.parquet")


def base_parameters() -> HydrologyParameters:
    return HydrologyParameters(
        t_0=0.0,
        m_sn=0.002,
        k_sn=1.157e-7,
        c_e=0.8,
        s_max=0.05,
        m_s=1e-5,
        beta_s=2.0,
        k_sgw=1e-7,
        k_gwpc=1e-7,
        k_gwap=1e-6,
        k_gwac=1e-6,
        beta_gwac=2.0,
        s_gwa_max=1.0,
        s_ref_td=0.5,
        tile_drainage_method="water_table",
        water_table_reference_depth=1.8,
        tile_depth=1.0,
        specific_yield=0.1,
        k_td=1e-5,
        gamma_x=-0.34,
        gamma_i=0.32,
        gamma_p=336.0,
        pet_albedo=0.23,
        pet_emissivity=0.98,
        n_gu=2.0,
        a_gu_seconds=2 * 60 * 60,
        area_km2=100.0,
    )


def initial_states() -> HydrologyStates:
    return HydrologyStates(
        s_sn=0.01,
        s_s=0.03,
        s_gwa=0.2,
        s_gwp=0.5,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun scenarios even when cached outputs exist",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable hydrology progress bars",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="limit forcing rows for a quick comparison run",
    )
    return parser.parse_args()


def run_scenario(
    name: str,
    params: HydrologyParameters,
    forcing_df: pd.DataFrame,
    *,
    force: bool,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    hydrology = Hydrology(
        output_dir=OUTPUT_DIR / name,
        params=params,
        initial_states=initial_states(),
        forcing_df=forcing_df.copy(),
    )
    hydrology.solve(force=force, progress=progress)
    hydrology.export()
    states, fluxes, _ = hydrology.load_outputs()
    if hydrology.result is None:
        discharge_df = read_table(
            hydrology.output_dir / hydrology.artifact_names.discharge,
            parse_dates=["time"],
        )
        discharge = pd.Series(
            discharge_df["discharge_cms"].to_numpy(),
            index=pd.DatetimeIndex(discharge_df["time"]),
            name="discharge_cms",
        )
    else:
        discharge = hydrology.result.discharge_cms
    return states, fluxes, discharge


def water_table_depth_series(states: pd.DataFrame, params: HydrologyParameters) -> pd.Series:
    return pd.Series(
        states["s_gwa"].map(lambda storage: water_table_depth(float(storage), params)).to_numpy(),
        index=pd.DatetimeIndex(states["time"]),
        name="water_table_depth",
    )


def plot_comparison(
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series]],
    params_by_name: dict[str, HydrologyParameters],
) -> None:
    colors = {
        "current": "#4c566a",
        "new_tiles": "#0072b2",
        "new_notiles": "#d55e00",
    }
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True, layout="constrained")
    for name, (states, fluxes, discharge) in results.items():
        states_mm = convert_states_to_nitrogen_units(states)
        fluxes_mm_day = convert_fluxes_to_nitrogen_units(fluxes)
        time = pd.to_datetime(fluxes_mm_day["time"])
        total_channel_flux_mm_day = (
            fluxes_mm_day["q_sc"]
            + fluxes_mm_day["q_gwatd"]
            + fluxes_mm_day["q_gwac"]
            + fluxes_mm_day["q_gwpc"]
        )
        cumulative_channel_mm = total_channel_flux_mm_day.cumsum() / 24.0
        color = colors[name]

        axs[0].plot(discharge.index, discharge, linewidth=0.7, label=name, color=color)
        axs[1].plot(time, fluxes_mm_day["q_gwatd"], linewidth=0.7, label=name, color=color)
        axs[2].plot(time, total_channel_flux_mm_day, linewidth=0.7, label=name, color=color)
        axs[3].plot(states_mm["time"], states_mm["s_gwa"], linewidth=0.7, label=name, color=color)
        axs[4].plot(time, cumulative_channel_mm, linewidth=0.9, label=name, color=color)

    axs[0].set_ylabel("Discharge (m3/s)")
    axs[1].set_ylabel("Tile flux (mm/day)")
    axs[2].set_ylabel("Channel flux (mm/day)")
    axs[3].set_ylabel("Active GW (mm)")
    axs[4].set_ylabel("Cumulative outflow (mm)")
    for ax in axs:
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)

    fig.savefig(OUTPUT_DIR / "hydrology_tile_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    for name, (states, _, _) in results.items():
        if params_by_name[name].tile_drainage_method == "water_table":
            water_depth = water_table_depth_series(states, params_by_name[name])
            ax.plot(water_depth.index, water_depth, linewidth=0.7, label=name)
    ax.axhline(base_parameters().tile_depth, color="black", linewidth=0.9, linestyle="--")
    ax.invert_yaxis()
    ax.set_ylabel("Water table depth (m)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.savefig(OUTPUT_DIR / "hydrology_tile_water_table.png", dpi=150)
    plt.close(fig)


def summarize(
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series]]
) -> pd.DataFrame:
    rows = []
    for name, (_, fluxes, discharge) in results.items():
        fluxes_mm_day = convert_fluxes_to_nitrogen_units(fluxes)
        total_channel_flux_mm_day = (
            fluxes_mm_day["q_sc"]
            + fluxes_mm_day["q_gwatd"]
            + fluxes_mm_day["q_gwac"]
            + fluxes_mm_day["q_gwpc"]
        )
        rows.append(
            {
                "scenario": name,
                "mean_discharge_cms": discharge.mean(),
                "max_discharge_cms": discharge.max(),
                "mean_tile_flux_mm_day": fluxes_mm_day["q_gwatd"].mean(),
                "max_tile_flux_mm_day": fluxes_mm_day["q_gwatd"].max(),
                "cumulative_tile_mm": fluxes_mm_day["q_gwatd"].sum() / 24.0,
                "cumulative_channel_mm": total_channel_flux_mm_day.sum() / 24.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    progress = not args.no_progress
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params = base_parameters()
    forcing = load_forcing_data(
        HYDROLOGY_FORCING_PARQUET,
        start_time=None,
        end_time=None,
        params=params,
    )
    if args.max_rows is not None:
        forcing = forcing.head(args.max_rows).copy()

    params_by_name = {
        "current": replace(params, tile_drainage_method="relative_storage"),
        "new_tiles": replace(params, tile_drainage_method="water_table"),
        "new_notiles": replace(params, tile_drainage_method="none"),
    }
    results = {
        name: run_scenario(
            name,
            scenario_params,
            forcing,
            force=args.force,
            progress=progress,
        )
        for name, scenario_params in params_by_name.items()
    }

    plot_comparison(results, params_by_name)
    summary = summarize(results)
    summary.to_parquet(OUTPUT_DIR / "hydrology_tile_summary.parquet", engine="fastparquet", index=False)

    print("Hydrology tile-drainage comparison")
    print(f"Forcing rows: {len(forcing)}")
    print(summary.to_string(index=False))
    print(f"Saved comparison outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
