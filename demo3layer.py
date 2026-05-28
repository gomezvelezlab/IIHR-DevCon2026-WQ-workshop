"""Run a three-compartment nitrogen routing demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

# Add the directory containing script.py to the Python path
import os
import sys

from matplotlib import pyplot as plt

utils_path = os.path.abspath("./src/")
if utils_path not in sys.path:
    sys.path.append(utils_path)

from devcon2026.hydrology import (
    Hydrology,
    HydrologyArtifactNames,
    HydrologyParameters,
    HydrologyStates,
)
from devcon2026.hydrology.export import (
    convert_fluxes_to_nitrogen_units,
    convert_states_to_nitrogen_units,
)
from devcon2026.nitrogen import Nitrogen, NitrogenParameters, NitrogenThreeCompartment
from devcon2026.tables import read_table, write_table

OUTPUT_DIR = Path("demo_outputs")
HYDROLOGY_OUTPUT_DIR = OUTPUT_DIR / "hydrology_3layer"
SCENARIO_OUTPUT_DIR = OUTPUT_DIR / "nitrogen_3layer_scenarios"
HYDROLOGY_FORCING_PARQUET = Path("data/hydrology_forcings.parquet")
NITROGEN_FORCING_PARQUET = Path("data/nitrogen_forcings.parquet")
HYDROLOGY_FORCINGS_PLOT = OUTPUT_DIR / "hydrology_3layer_variants.png"
NITROGEN_DIN_PLOT = OUTPUT_DIR / "nitrogen_3layer_din_scenarios.png"
NITROGEN_DON_PLOT = OUTPUT_DIR / "nitrogen_3layer_don_scenarios.png"
NITROGEN_DIN_MASS_PLOT = OUTPUT_DIR / "nitrogen_3layer_din_mass_scenarios.png"
NITROGEN_DON_MASS_PLOT = OUTPUT_DIR / "nitrogen_3layer_don_mass_scenarios.png"

HYDROLOGY_SPIN_START = "2007-01-01"
NITROGEN_SPIN_START = "2008-01-01"
RESULTS_START = "2009-01-01"
SIMULATION_END = "2018-01-01"
FORCE_HYDROLOGY = True
FORCE_NITROGEN = True
SHOW_PROGRESS = True

HYDROLOGY_ARTIFACTS = HydrologyArtifactNames(
    discharge="discharge1.parquet",
    states="states1.parquet",
    fluxes="fluxes1.parquet",
    forcing="model_ready_hydrology_forcing.parquet",
)

HYDROLOGY_PARAMS = HydrologyParameters(
    t_0=0.0,  # snow-rain temperature threshold [C]
    m_sn=0.002,  # snowmelt storage scale [m]
    k_sn=1.157e-7,  # degree-day snowmelt coefficient [m/s/C]
    c_e=0.8,  # actual ET correction coefficient [1]
    s_max=0.1,  # maximum soil storage [m]
    m_s=1e-5,  # soil ET shape parameter [1]
    beta_s=2.0,  # soil wetness exponent [1]
    k_sgw=1e-7,  # vertical groundwater recharge flux scale [m/s]
    k_gwpc=1e-7,  # passive groundwater recession coefficient [1/s]
    k_gwap=1e-6,  # active-to-passive groundwater transfer coefficient [1/s]
    k_gwac=1e-6,  # active groundwater channel flux scale [m/s]
    beta_gwac=2.0,  # active groundwater channel exponent [1]
    s_gwa_max=1.0,  # maximum active groundwater storage [m]
    s_ref_td=0.005,  # legacy relative tile drainage activation threshold [1]
    tile_drainage_method="relative_storage",  # tile drainage formulation [method]
    k_td=1e-4,  # tile drainage coefficient [1/s]
    gamma_x=-0.34,  # seasonal infiltration partition offset [1]
    gamma_i=0.32,  # seasonal infiltration partition amplitude [1]
    gamma_p=336.0,  # seasonal infiltration partition phase [day]
    pet_albedo=0.23,  # surface albedo for reference ET [1]
    pet_emissivity=0.98,  # surface emissivity for reference ET [1]
    n_gu=2.0,  # gamma unit hydrograph shape parameter [1]
    a_gu_seconds=2 * 60 * 60,  # gamma unit hydrograph scale parameter [s]
    area_km2=100.0,  # drainage area [km2]
)

HYDROLOGY_INITIAL_STATES = HydrologyStates(
    s_sn=0.01,  # snow storage [m]
    s_s=0.03,  # soil storage [m]
    s_gwa=0.2,  # active GW storage [m]
    s_gwp=0.5,  # passive GW storage [m]
)

HYDROLOGY_SCENARIOS = {
    "tiles": HYDROLOGY_PARAMS,
    "no_tiles": HYDROLOGY_PARAMS.with_updates({"k_td": 0.0}),
}

NITROGEN_PARAMS = NitrogenParameters(
    s_wp=20.0,  # wilting point soil water storage [mm]
    s_max=1 * HYDROLOGY_PARAMS.s_max,  # maximum soil water storage [mm]
    min_dissolved_storage=0.1,  # minimum storage for dissolved concentrations [mm]
    smf_sat=0.8,  # saturated moisture factor [1]
    beta_sm=1.0,  # moisture factor exponent [1]
    rel_saturation_low=0.2,  # low relative saturation threshold [1]
    rel_saturation_high=0.9,  # high relative saturation threshold [1]
    rel_sat_limit_exp=0.7,  # exponential moisture limitation threshold [1]
    beta_exp=2.5,  # exponential moisture factor exponent [1]
    v_degrad_son=1e-5,  # maximum degradation rate of slow organic nitrogen [1/day]
    v_dissol_son=1e-5,  # maximum dissolution rate of slow organic nitrogen [1/day]
    v_dissol_fon=1e-3,  # maximum dissolution rate of fast organic nitrogen [1/day]
    v_min_fon=1e-3,  # maximum mineralization rate of fast organic nitrogen [1/day]
    v_denit=5e-2,  # maximum denitrification rate [1/day]
    k_denit=1.5,  # denitrification half-saturation concentration [mg/L]
    uptake_demand=50.0,  # plant inorganic nitrogen uptake demand [kg N/km2/day]
    delta_time_solver=1.0 / 24.0,  # nitrogen solver time step [day]
    freundlich_exponent=1.0,  # Freundlich DON adsorption exponent [1]
    freundlich_constant=100.0,  # Freundlich DON adsorption constant [(mg N/kg soil)/(mg N/L)^n]
    soil_bulk_density=1.3,  # soil bulk density used for DON adsorption [kg/L]
    deposition_din_fraction=1.0,
    deposition_don_fraction=0.0,
    deposition_son_fraction=0.0,
    deposition_fon_fraction=0.0,
    fertilizer_din_fraction=0.0,
    fertilizer_don_fraction=0.0,
    fertilizer_son_fraction=0.0,
    fertilizer_fon_fraction=1.0,
    manure_din_fraction=0.0,
    manure_don_fraction=0.0,
    manure_son_fraction=0.0,
    manure_fon_fraction=1.0,
)

NITROGEN_SCENARIOS = {
    "adsorption": True,
    "no_adsorption": False,
}


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Running hydrology scenario: {name}")
    hydrology = Hydrology(
        output_dir=HYDROLOGY_OUTPUT_DIR / name,
        artifact_names=HYDROLOGY_ARTIFACTS,
        params=params,
        initial_states=HYDROLOGY_INITIAL_STATES,
        forcing_path=HYDROLOGY_FORCING_PARQUET,
        forcing_start=HYDROLOGY_SPIN_START,
        forcing_end=SIMULATION_END,
    )
    hydrology.solve(force=FORCE_HYDROLOGY, progress=SHOW_PROGRESS)
    hydrology.export()
    print(f"Hydrology scenario {name}: {hydrology.source}")
    states, fluxes, meteorology = hydrology.load_outputs()
    return (
        apply_time_window(states, start=NITROGEN_SPIN_START, end=SIMULATION_END),
        apply_time_window(fluxes, start=NITROGEN_SPIN_START, end=SIMULATION_END),
        apply_time_window(meteorology, start=NITROGEN_SPIN_START, end=SIMULATION_END),
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
) -> pd.DataFrame:
    scenario_name = f"{hydrology_name}_{nitrogen_name}"
    scenario_dir = SCENARIO_OUTPUT_DIR / scenario_name
    solution_path = scenario_dir / "nitrogen_solution.parquet"
    forcing_path = scenario_dir / "nitrogen_forcings.parquet"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting nitrogen scenario: {scenario_name}")
    if solution_path.exists() and not FORCE_NITROGEN:
        print(f"Loading nitrogen scenario: {scenario_name}")
        return read_table(solution_path, parse_dates=["time"])

    print(f"Running nitrogen scenario: {scenario_name}")

    model = NitrogenThreeCompartment(NITROGEN_PARAMS)
    df_forcings = model.from_hydrology_outputs(states, fluxes, meteorology)
    source_helper = Nitrogen(params=model.params)
    df_forcings = source_helper.add_nitrogen_source_forcings(
        df_forcings,
        NITROGEN_FORCING_PARQUET,
    )
    write_table(df_forcings, forcing_path)

    solution = model.simulate(
        df_forcings,
        with_soil_don_adsorption=with_soil_don_adsorption,
        progress=SHOW_PROGRESS,
        progress_desc=f"nitrogen {scenario_name}",
    )
    solution = add_channel_concentrations(solution, df_forcings)

    write_table(solution, solution_path)
    print(f"Saved nitrogen scenario: {scenario_name}")
    return solution


def plot_hydrology_scenarios(
    hydrology_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> None:
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True, layout="constrained")
    results_start = pd.Timestamp(RESULTS_START, tz="UTC")
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
    fig.savefig(HYDROLOGY_FORCINGS_PLOT, dpi=150)
    plt.close(fig)


def plot_concentration_scenarios(
    solutions: dict[str, pd.DataFrame],
    *,
    species: str,
    output_path: Path,
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
        plotted = apply_time_window(solution, start=RESULTS_START, end=SIMULATION_END)
        for ax, (column, label) in zip(axs, variables):
            ax.plot(
                plotted["time"], plotted[column], linewidth=0.7, label=scenario_name
            )
            ax.set_ylabel(label)
    for ax in axs:
        ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mass_scenarios(
    solutions: dict[str, pd.DataFrame],
    *,
    species: str,
    output_path: Path,
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
        plotted = apply_time_window(solution, start=RESULTS_START, end=SIMULATION_END)
        for ax, (column, label) in zip(axs, variables):
            ax.plot(
                plotted["time"], plotted[column], linewidth=0.7, label=scenario_name
            )
            ax.set_ylabel(label)
    for ax in axs:
        ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    SCENARIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hydrology_outputs = {
        name: run_hydrology_scenario(name, params)
        for name, params in HYDROLOGY_SCENARIOS.items()
    }
    solutions: dict[str, pd.DataFrame] = {}
    for hydrology_name, (states, fluxes, meteorology) in hydrology_outputs.items():
        for nitrogen_name, with_adsorption in NITROGEN_SCENARIOS.items():
            scenario_name = f"{hydrology_name}_{nitrogen_name}"
            solutions[scenario_name] = run_nitrogen_scenario(
                hydrology_name,
                nitrogen_name,
                states,
                fluxes,
                meteorology,
                with_soil_don_adsorption=with_adsorption,
            )

    plot_hydrology_scenarios(hydrology_outputs)
    plot_concentration_scenarios(
        solutions, species="din", output_path=NITROGEN_DIN_PLOT
    )
    plot_concentration_scenarios(
        solutions, species="don", output_path=NITROGEN_DON_PLOT
    )
    plot_mass_scenarios(solutions, species="din", output_path=NITROGEN_DIN_MASS_PLOT)
    plot_mass_scenarios(solutions, species="don", output_path=NITROGEN_DON_MASS_PLOT)

    baseline = "tiles_adsorption"
    results = apply_time_window(
        solutions[baseline], start=RESULTS_START, end=SIMULATION_END
    )
    print("Three-compartment nitrogen simulation demo")
    print(f"Hydrology spin window: {HYDROLOGY_SPIN_START} to {NITROGEN_SPIN_START}")
    print(f"Nitrogen spin window: {NITROGEN_SPIN_START} to {RESULTS_START}")
    print(f"Hydrology scenarios: {', '.join(HYDROLOGY_SCENARIOS)}")
    print(f"Nitrogen scenarios: {', '.join(NITROGEN_SCENARIOS)}")
    print(f"Rows used for plotted/final summary: {len(results)}")
    print(
        f"Final {baseline} soil DIN concentration: {results['soil_c_din'].iloc[-1]:.3f} mg N/L"
    )
    print(
        f"Final {baseline} active GW DIN concentration: {results['gwa_c_din'].iloc[-1]:.3f} mg N/L"
    )
    print(
        f"Final {baseline} passive GW DIN concentration: {results['gwp_c_din'].iloc[-1]:.3f} mg N/L"
    )
    print(
        f"Final {baseline} channel DIN concentration: {results['channel_c_din'].iloc[-1]:.3f} mg N/L"
    )
    print(f"Saved scenario outputs to {SCENARIO_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
