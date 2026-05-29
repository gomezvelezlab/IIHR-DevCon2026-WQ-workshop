from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

# matplotlib.use("Agg")

# Add the directory containing script.py to the Python path
import os
import sys

from matplotlib import pyplot as plt

from .tables import read_table, write_table
from .nitrogen import Nitrogen, NitrogenThreeCompartment
from .hydrology import (
    Hydrology,
    HydrologyParameters,
)

from .hydrology.export import (
    convert_fluxes_to_nitrogen_units,
    convert_states_to_nitrogen_units,
)

def apply_time_window(
    df: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    time = pd.to_datetime(df["time"], utc=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    return df.loc[(time >= start_ts) & (time < end_ts)].reset_index(drop=True)


def run_hydrology_scenario(
    name: str,
    params: HydrologyParameters,
    utils_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Running hydrology scenario: {name}")
    hydrology = Hydrology(
        output_dir=utils_params["HYDROLOGY_OUTPUT_DIR"] / name,
        artifact_names=utils_params["HYDROLOGY_ARTIFACTS"],
        params=params,
        initial_states=utils_params["HYDROLOGY_INITIAL_STATES"],
        forcing_path=utils_params["HYDROLOGY_FORCING_PARQUET"],
        forcing_start=utils_params["HYDROLOGY_SPIN_START"],
        forcing_end=utils_params["SIMULATION_END"],
    )
    hydrology.solve(force=utils_params["FORCE_HYDROLOGY"], progress=utils_params["SHOW_PROGRESS"])
    hydrology.export()
    print(f"Hydrology scenario {name}: {hydrology.source}")
    states, fluxes, meteorology = hydrology.load_outputs()
    return (
        apply_time_window(states, start=utils_params["NITROGEN_SPIN_START"], end=utils_params["SIMULATION_END"]),
        apply_time_window(fluxes, start=utils_params["NITROGEN_SPIN_START"], end=utils_params["SIMULATION_END"]),
        apply_time_window(meteorology, start=utils_params["NITROGEN_SPIN_START"], end=utils_params["SIMULATION_END"]),
    )


def add_channel_concentrations(
    solution: pd.DataFrame,
    forcings: pd.DataFrame,
) -> pd.DataFrame:
    output = solution.copy()
    q_soil_channel = forcings["q_sc"].to_numpy()
    q_active_channel = (forcings["q_gwatd"] + forcings["q_gwac"]).to_numpy()
    q_passive_channel = forcings["q_gwpc"].to_numpy()
    q_channel = q_soil_channel + q_active_channel + q_passive_channel

    for species in ("din", "don"):
        mass_flux = (
            q_soil_channel * output[f"soil_c_{species}"].to_numpy()
            + q_active_channel * output[f"gwa_c_{species}"].to_numpy()
            + q_passive_channel * output[f"gwp_c_{species}"].to_numpy()
        )
        output[f"channel_c_{species}"] = 0.0
        mask = q_channel > 0.0
        output.loc[mask, f"channel_c_{species}"] = mass_flux[mask] / q_channel[mask]
    return output


def run_nitrogen_scenario(
    hydrology_name: str,
    nitrogen_name: str,
    states: pd.DataFrame,
    fluxes: pd.DataFrame,
    meteorology: pd.DataFrame,
    *,
    with_soil_don_adsorption: bool,
    utils_params: dict,
) -> pd.DataFrame:
    scenario_name = f"{hydrology_name}_{nitrogen_name}"
    scenario_dir = utils_params["SCENARIO_OUTPUT_DIR"] / scenario_name
    solution_path = scenario_dir / "nitrogen_solution.parquet"
    forcing_path = scenario_dir / "nitrogen_forcings.parquet"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting nitrogen scenario: {scenario_name}")
    if solution_path.exists() and not utils_params["FORCE_NITROGEN"]:
        print(f"Loading nitrogen scenario: {scenario_name}")
        return read_table(solution_path, parse_dates=["time"])

    print(f"Running nitrogen scenario: {scenario_name}")

    model = NitrogenThreeCompartment(
        soil_params=utils_params["NITROGEN_SOIL_PARAMS"],
        gwa_params=utils_params["NITROGEN_GWA_PARAMS"],
        gwp_params=utils_params["NITROGEN_GWP_PARAMS"],
    )
    df_forcings = model.from_hydrology_outputs(states, fluxes, meteorology)
    source_helper = Nitrogen(params=model.params)
    df_forcings = source_helper.add_nitrogen_source_forcings(
        df_forcings,
        utils_params["NITROGEN_FORCING_PARQUET"],
    )
    write_table(df_forcings, forcing_path)

    solution = model.simulate(
        df_forcings,
        with_soil_don_adsorption=with_soil_don_adsorption,
        progress=utils_params["SHOW_PROGRESS"],
        progress_desc=f"nitrogen {scenario_name}",
    )
    solution = add_channel_concentrations(solution, df_forcings)

    write_table(solution, solution_path)
    print(f"Saved nitrogen scenario: {scenario_name}")
    return solution


def plot_hydrology_scenarios(
    hydrology_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    utils_params: dict,
) -> None:
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True, layout="constrained")
    results_start = pd.Timestamp(utils_params["RESULTS_START"], tz="UTC")
    for name, (states, fluxes, _) in hydrology_outputs.items():
        states_mm = convert_states_to_nitrogen_units(states)
        fluxes_mm_day = convert_fluxes_to_nitrogen_units(fluxes)
        states_mm = states_mm[states_mm["time"] >= results_start]
        fluxes_mm_day = fluxes_mm_day[fluxes_mm_day["time"] >= results_start]
        tile_flux = fluxes_mm_day["q_gwatd"].rolling(24, min_periods=1).mean()
        gw_channel_flux = (
            (
                fluxes_mm_day["q_gwac"]
                + fluxes_mm_day["q_gwatd"]
                + fluxes_mm_day["q_gwpc"]
            )
            .rolling(24, min_periods=1)
            .mean()
        )
        axs[0].plot(states_mm["time"], states_mm["s_s"], linewidth=0.7, label=name)
        axs[1].plot(
            states_mm["time"],
            states_mm["s_gwa"],
            linewidth=0.7,
            label=name,
        )
        axs[2].plot(
            fluxes_mm_day["time"],
            tile_flux,
            linewidth=0.7,
            label=name,
        )
        axs[3].plot(
            fluxes_mm_day["time"],
            gw_channel_flux,
            linewidth=0.7,
            label=name,
        )
    _, _, first_forcing = next(iter(hydrology_outputs.values()))
    first_forcing = first_forcing[first_forcing["time"] >= results_start]
    # axs[3].plot(
    #     first_forcing["time"],
    #     first_forcing["TMP_2maboveground"] - 273.15,
    #     linewidth=0.7,
    # )
    axs[0].set_ylabel("Soil water storage (mm)")
    axs[1].set_ylabel("Active GW storage (mm)")
    axs[2].set_ylabel("24h tile flux (mm/day)")
    axs[3].set_ylabel("24h GW channel flux (mm/day)")
    # axs[3].set_ylabel("Air temp (C)")
    for ax in axs:
        ax.legend()
    fig.savefig(utils_params["HYDROLOGY_FORCINGS_PLOT"], dpi=150)
    plt.close(fig)


def plot_forcing_scenarios(
    hydrology_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    utils_params: dict,
) -> None:
    fig, axs = plt.subplots(6, 1, figsize=(11, 13), sharex=False, layout="constrained")
    _, first_fluxes, first_meteorology = next(iter(hydrology_outputs.values()))

    meteorology = apply_time_window(
        first_meteorology, start=utils_params["RESULTS_START"], end=utils_params["SIMULATION_END"]
    ).copy()
    fluxes = apply_time_window(first_fluxes, start=utils_params["RESULTS_START"], end=utils_params["SIMULATION_END"])
    fluxes_mm_day = convert_fluxes_to_nitrogen_units(fluxes)

    meteorology["time"] = pd.to_datetime(meteorology["time"], utc=True)
    fluxes_mm_day["time"] = pd.to_datetime(fluxes_mm_day["time"], utc=True)

    hydro_daily = pd.DataFrame(
        {
            "precipitation": meteorology.set_index("time")["precipitation_mm"]
            .resample("D")
            .sum(),
            "pet": meteorology.set_index("time")["ref_et_mm_hr"].resample("D").sum(),
            "aet": (fluxes_mm_day.set_index("time")["e_a"] / 24.0).resample("D").sum(),
        }
    )

    nitrogen_daily = read_table(utils_params["NITROGEN_FORCING_PARQUET"], parse_dates=["date"])
    nitrogen_daily["date"] = pd.to_datetime(nitrogen_daily["date"])
    nitrogen_daily = apply_time_window(
        nitrogen_daily.rename(columns={"date": "time"}),
        start=utils_params["RESULTS_START"],
        end=utils_params["SIMULATION_END"],
    )

    hydro_plots = [
        ("precipitation", "Precipitation (mm/day)"),
        ("pet", "PET (mm/day)"),
        ("aet", "AET (mm/day)"),
    ]
    nitrogen_plots = [
        ("deposition_kgN_km2_day", "Deposition (kg N/km2/day)"),
        ("fertilizer_kgN_km2_day", "Fertilizer (kg N/km2/day)"),
        ("manure_kgN_km2_day", "Manure (kg N/km2/day)"),
    ]

    for ax, (column, label) in zip(axs[:3], hydro_plots):
        ax.plot(hydro_daily.index, hydro_daily[column], linewidth=0.7)
        ax.set_ylabel(label)
    for ax, (column, label) in zip(axs[3:], nitrogen_plots):
        ax.plot(nitrogen_daily["time"], nitrogen_daily[column], linewidth=0.7)
        ax.set_ylabel(label)

    fig.savefig(utils_params["FORCINGS_PLOT"], dpi=150)
    plt.close(fig)


def plot_concentration_scenarios(
    solutions: dict[str, pd.DataFrame],
    *,
    species: str,
    output_path: Path,
    utils_params: dict,
) -> None:
    variables = [
        (f"soil_c_{species}", f"Soil {species.upper()} (mg/L)"),
        (f"gwa_c_{species}", f"Active GW {species.upper()} (mg/L)"),
        (f"gwp_c_{species}", f"Passive GW {species.upper()} (mg/L)"),
        (f"channel_c_{species}", f"Channel {species.upper()} (mg/L)"),
    ]
    fig, axs = plt.subplots(
        len(variables),
        1,
        figsize=(11, 2.4 * len(variables)),
        sharex=True,
        layout="constrained",
    )
    for scenario_name, solution in solutions.items():
        plotted = apply_time_window(solution, start=utils_params["RESULTS_START"], end=utils_params["SIMULATION_END"])
        for ax, (column, label) in zip(axs, variables):
            ax.plot(
                plotted["time"], plotted[column], linewidth=0.7, label=scenario_name
            )
            ax.set_ylabel(label)
            if column == f"soil_c_{species}":
                ax.set_ylim(0.0, 300.0)
    for ax in axs:
        ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mass_scenarios(
    solutions: dict[str, pd.DataFrame],
    *,
    species: str,
    output_path: Path,
    utils_params: dict,
) -> None:
    variables = [
        (f"soil_m_{species}", f"Soil {species.upper()} (kg N/km2)"),
        (f"gwa_m_{species}", f"Active GW {species.upper()} (kg N/km2)"),
        (f"gwp_m_{species}", f"Passive GW {species.upper()} (kg N/km2)"),
    ]
    fig, axs = plt.subplots(
        len(variables),
        1,
        figsize=(11, 2.4 * len(variables)),
        sharex=True,
        layout="constrained",
    )
    for scenario_name, solution in solutions.items():
        plotted = apply_time_window(solution, start=utils_params["RESULTS_START"], end=utils_params["SIMULATION_END"])
        for ax, (column, label) in zip(axs, variables):
            ax.plot(
                plotted["time"], plotted[column], linewidth=0.7, label=scenario_name
            )
            ax.set_ylabel(label)
    for ax in axs:
        ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

