"""Run a three-compartment nitrogen routing demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from devcon2026.hydrology import Hydrology
from devcon2026.hydrology import HydrologyArtifactNames
from devcon2026.hydrology import HydrologyParameters
from devcon2026.hydrology import HydrologyStates
from devcon2026.hydrology.export import convert_fluxes_to_nitrogen_units
from devcon2026.hydrology.export import convert_states_to_nitrogen_units
from devcon2026.nitrogen import Nitrogen
from devcon2026.nitrogen import NitrogenParameters
from devcon2026.nitrogen import NitrogenThreeCompartment
from devcon2026.tables import write_table


OUTPUT_DIR = Path("demo_outputs")
HYDROLOGY_OUTPUT_DIR = OUTPUT_DIR / "example_hydrology_3layer_model"
HYDROLOGY_FORCING_PARQUET = Path("data/hydrology_forcings.parquet")
NITROGEN_FORCING_PARQUET = Path("data/nitrogen_forcings.parquet")
NITROGEN_SOLUTION_PARQUET = OUTPUT_DIR / "nitrogen_3layer_solution.parquet"
NITROGEN_FORCINGS_PARQUET = OUTPUT_DIR / "nitrogen_3layer_forcings.parquet"
HYDROLOGY_FORCINGS_PLOT = OUTPUT_DIR / "hydrologic_forcings.png"
NITROGEN_SOLUTION_PLOT = OUTPUT_DIR / "nitrogen_3layer_solution.png"

HYDROLOGY_SPIN_START = "2007-01-01"
NITROGEN_SPIN_START = "2008-01-01"
RESULTS_START = "2009-01-01"
SIMULATION_END = "2018-01-01"
FORCE_HYDROLOGY = False
SHOW_PROGRESS = True

HYDROLOGY_ARTIFACTS = HydrologyArtifactNames(
    discharge="discharge1.parquet",
    states="states1.parquet",
    fluxes="fluxes1.parquet",
    forcing="south_fork_aorc_forcing.parquet",
)

HYDROLOGY_PARAMS = HydrologyParameters(
    t_0=0.0,  # snow-rain temperature threshold [C]
    m_sn=0.002,  # snowmelt storage scale [m]
    k_sn=1.157e-7,  # degree-day snowmelt coefficient [m/s/C]
    c_e=0.8,  # actual ET correction coefficient [1]
    s_max=0.05,  # maximum soil storage [m]
    m_s=1e-5,  # soil ET shape parameter [1]
    beta_s=2.0,  # soil wetness exponent [1]
    k_sgw=1e-7,  # vertical groundwater recharge flux scale [m/s]
    k_gwpc=1e-7,  # passive groundwater recession coefficient [1/s]
    k_gwap=1e-6,  # active-to-passive groundwater transfer coefficient [1/s]
    k_gwac=1e-6,  # active groundwater channel flux scale [m/s]
    beta_gwac=2.0,  # active groundwater channel exponent [1]
    s_gwa_max=1.0,  # maximum active groundwater storage [m]
    s_ref_td=0.08,  # legacy relative tile drainage activation threshold [1]
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

NITROGEN_PARAMS = NitrogenParameters(
    s_wp=20.0,  # wilting point soil water storage [mm]
    s_max=147.3,  # maximum soil water storage [mm]
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
    uptake_demand=10.0,  # plant inorganic nitrogen uptake demand [kg N/km2/day]
    delta_time_solver=1.0 / 24.0,  # nitrogen solver time step [day]
    freundlich_exponent=1.0,  # Freundlich DON adsorption exponent [1]
    freundlich_constant=100.0,  # Freundlich DON adsorption constant [(mg N/kg soil)/(mg N/L)^n]
    soil_bulk_density=1.3,  # soil bulk density used for DON adsorption [kg/L]
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


def plot_hydrologic_forcings(
    states: pd.DataFrame,
    fluxes: pd.DataFrame,
    forcings: pd.DataFrame,
) -> None:
    states_mm = convert_states_to_nitrogen_units(states)
    fluxes_mm_day = convert_fluxes_to_nitrogen_units(fluxes)
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    axs[0].plot(states_mm["time"], states_mm["s_s"], linewidth=0.7)
    axs[0].plot(states_mm["time"], states_mm["s_gwa"], linewidth=0.7)
    axs[0].set_ylabel("Storage (mm)")
    axs[1].plot(fluxes_mm_day["time"], fluxes_mm_day["p_r"], linewidth=0.7, label="Rain")
    axs[1].plot(fluxes_mm_day["time"], fluxes_mm_day["f_sm"], linewidth=0.7, label="Snowmelt")
    axs[1].set_ylabel("Fluxes in (mm/day)")
    axs[1].legend()
    axs[2].plot(fluxes_mm_day["time"], fluxes_mm_day["q_sc"], linewidth=0.7, label="Surface/channel")
    axs[2].plot(fluxes_mm_day["time"], fluxes_mm_day["q_sgwa"], linewidth=0.7, label="Soil to active GW")
    axs[2].plot(fluxes_mm_day["time"], fluxes_mm_day["q_gwatd"], linewidth=0.7, label="Tile")
    axs[2].set_ylabel("Fluxes out (mm/day)")
    axs[2].legend()
    axs[3].plot(forcings["time"], forcings["TMP_2maboveground"] - 273.15, linewidth=0.7)
    axs[3].set_ylabel("Air temp (C)")
    fig.savefig(HYDROLOGY_FORCINGS_PLOT, dpi=150)
    plt.close(fig)


def plot_three_compartment_solution(solution: pd.DataFrame) -> None:
    plotted = apply_time_window(solution, start=RESULTS_START, end=SIMULATION_END)
    variables = [
        ("soil_c_din", "Soil DIN (mg/L)"),
        ("gwa_c_din", "Active GW DIN (mg/L)"),
        ("gwp_c_din", "Passive GW DIN (mg/L)"),
        ("soil_c_don", "Soil DON (mg/L)"),
        ("gwa_c_don", "Active GW DON (mg/L)"),
        ("gwp_c_don", "Passive GW DON (mg/L)"),
    ]
    fig, axs = plt.subplots(
        len(variables),
        1,
        figsize=(10, 2.1 * len(variables)),
        sharex=True,
        layout="constrained",
    )
    for ax, (column, label) in zip(axs, variables):
        ax.plot(plotted["time"], plotted[column], linewidth=0.7)
        ax.set_ylabel(label)
    fig.savefig(NITROGEN_SOLUTION_PLOT, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    hydrology = Hydrology(
        output_dir=HYDROLOGY_OUTPUT_DIR,
        artifact_names=HYDROLOGY_ARTIFACTS,
        params=HYDROLOGY_PARAMS,
        initial_states=HYDROLOGY_INITIAL_STATES,
        forcing_path=HYDROLOGY_FORCING_PARQUET,
        forcing_start=HYDROLOGY_SPIN_START,
        forcing_end=SIMULATION_END,
    )
    hydrology.solve(force=FORCE_HYDROLOGY, progress=SHOW_PROGRESS)
    hydrology.export()

    states, fluxes, meteorology = hydrology.load_outputs()
    states = apply_time_window(states, start=NITROGEN_SPIN_START, end=SIMULATION_END)
    fluxes = apply_time_window(fluxes, start=NITROGEN_SPIN_START, end=SIMULATION_END)
    meteorology = apply_time_window(meteorology, start=NITROGEN_SPIN_START, end=SIMULATION_END)

    model = NitrogenThreeCompartment(NITROGEN_PARAMS)
    df_forcings = model.from_hydrology_outputs(states, fluxes, meteorology)
    source_helper = Nitrogen(params=model.params)
    df_forcings = source_helper.add_nitrogen_source_forcings(
        df_forcings,
        NITROGEN_FORCING_PARQUET,
    )
    solution = model.simulate(
        df_forcings,
        with_soil_don_adsorption=True,
        progress=SHOW_PROGRESS,
    )

    write_table(solution, NITROGEN_SOLUTION_PARQUET)
    write_table(df_forcings, NITROGEN_FORCINGS_PARQUET)
    plot_hydrologic_forcings(states, fluxes, meteorology)
    plot_three_compartment_solution(solution)

    results = apply_time_window(solution, start=RESULTS_START, end=SIMULATION_END)
    print("Three-compartment nitrogen simulation demo")
    print(f"Hydrology outputs: {hydrology.source}")
    print(f"Hydrology spin window: {HYDROLOGY_SPIN_START} to {NITROGEN_SPIN_START}")
    print(f"Nitrogen spin window: {NITROGEN_SPIN_START} to {RESULTS_START}")
    print(f"Forcing rows passed to nitrogen: {len(df_forcings)}")
    print(f"Rows used for plotted/final summary: {len(results)}")
    print(
        f"Final soil DIN concentration: {results['soil_c_din'].iloc[-1]:.3f} mg N/L"
    )
    print(
        f"Final active GW DIN concentration: {results['gwa_c_din'].iloc[-1]:.3f} mg N/L"
    )
    print(
        f"Final passive GW DIN concentration: {results['gwp_c_din'].iloc[-1]:.3f} mg N/L"
    )
    print(f"Saved three-compartment outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
