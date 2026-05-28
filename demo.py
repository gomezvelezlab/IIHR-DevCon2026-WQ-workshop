"""Run a dataframe-driven nitrogen simulation demo.

This mirrors the `origin/chucho` notebook workflow, but uses synthetic
hydrologic model outputs because the notebook's CSV inputs are not committed.
"""

from __future__ import annotations

import argparse
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
from devcon2026.nitrogen import NitrogenStates

OUTPUT_DIR = Path("demo_outputs")
HYDROLOGY_OUTPUT_DIR = OUTPUT_DIR / "example_hydrology_model"
HYDROLOGY_FORCING_PARQUET = Path("data/hydrology_forcings.parquet")
NITROGEN_FORCING_PARQUET = Path("data/nitrogen_forcings.parquet")
DISCHARGE_ARTIFACT = "discharge1.parquet"
STATES_ARTIFACT = "states1.parquet"
FLUXES_ARTIFACT = "fluxes1.parquet"
FORCING_ARTIFACT = "south_fork_aorc_forcing.parquet"
HYDROLOGY_ARTIFACTS = HydrologyArtifactNames(
    discharge=DISCHARGE_ARTIFACT,
    states=STATES_ARTIFACT,
    fluxes=FLUXES_ARTIFACT,
    forcing=FORCING_ARTIFACT,
)


def plot_hydrologic_forcings(
    states: pd.DataFrame, fluxes: pd.DataFrame, forcings: pd.DataFrame
) -> None:
    states_mm = convert_states_to_nitrogen_units(states)
    fluxes_mm_day = convert_fluxes_to_nitrogen_units(fluxes)
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    axs[0].plot(states_mm["time"], states_mm["s_s"], linewidth=0.7)
    axs[0].set_ylabel("Soil storage (mm)")
    axs[1].plot(
        fluxes_mm_day["time"], fluxes_mm_day["p_r"], linewidth=0.7, label="Rainfall"
    )
    axs[1].plot(
        fluxes_mm_day["time"], fluxes_mm_day["f_sm"], linewidth=0.7, label="Snowmelt"
    )
    axs[1].set_ylabel("Fluxes in (mm/day)")
    axs[1].legend()
    axs[2].plot(
        fluxes_mm_day["time"],
        fluxes_mm_day["q_sc"],
        linewidth=0.7,
        label="Flow to channel",
    )
    axs[2].plot(
        fluxes_mm_day["time"],
        fluxes_mm_day["q_sgwa"],
        linewidth=0.7,
        label="Flow to active groundwater",
    )
    axs[2].set_ylabel("Fluxes out (mm/day)")
    axs[2].legend()
    axs[3].plot(
        forcings["time"],
        forcings["TMP_2maboveground"] - 273.15,
        linewidth=0.7,
    )
    axs[3].set_ylabel("Air temp (C)")
    fig.savefig(OUTPUT_DIR / "hydrologic_forcings.png", dpi=150)
    plt.close(fig)


def plot_nitrogen_solution(
    solution_ads: pd.DataFrame, solution_no_ads: pd.DataFrame
) -> None:
    variables = ["m_don", "m_din", "m_son", "m_fon", "m_don_ads", "c_din", "c_don"]
    fig, axs = plt.subplots(
        len(variables),
        1,
        figsize=(10, 2.2 * len(variables)),
        sharex=True,
        layout="constrained",
    )
    for ax, variable in zip(axs, variables):
        ax.plot(
            solution_ads["time"],
            solution_ads[variable],
            linewidth=0.7,
            label="With DON adsorption",
        )
        if variable in solution_no_ads:
            ax.plot(
                solution_no_ads["time"],
                solution_no_ads[variable],
                linewidth=0.7,
                label="Without DON adsorption",
            )
        ax.set_ylabel(variable)
        ax.legend(loc="upper right")
    fig.savefig(OUTPUT_DIR / "nitrogen_solution.png", dpi=150)
    plt.close(fig)


def plot_mass_fluxes(mass_fluxes: pd.DataFrame) -> None:
    variables = [
        "r_don_flux",
        "r_din_flux",
        "q_adv_din_in_flux",
        "q_adv_din_out_flux",
        "d_din_flux",
        "u_din_flux",
    ]
    fig, axs = plt.subplots(
        len(variables),
        1,
        figsize=(10, 2.2 * len(variables)),
        sharex=True,
        layout="constrained",
    )
    for ax, variable in zip(axs, variables):
        ax.plot(mass_fluxes["time"], mass_fluxes[variable], linewidth=0.7)
        ax.set_ylabel(variable)
    fig.savefig(OUTPUT_DIR / "nitrogen_mass_fluxes.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-hydrology",
        action="store_true",
        help="rerun the synthetic hydrologic model even when exported CSVs exist",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable hydrologic and nitrogen progress bars",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress = not args.no_progress
    OUTPUT_DIR.mkdir(exist_ok=True)

    hydrology_params = HydrologyParameters(
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
    hydrology_initial_states = HydrologyStates(
        s_sn=0.01,
        s_s=0.03,
        s_gwa=0.2,
        s_gwp=0.5,
    )
    hydrology = Hydrology(
        output_dir=HYDROLOGY_OUTPUT_DIR,
        artifact_names=HYDROLOGY_ARTIFACTS,
        params=hydrology_params,
        initial_states=hydrology_initial_states,
        forcing_path=HYDROLOGY_FORCING_PARQUET,
    )
    hydrology.solve(force=args.force_hydrology, progress=progress)
    hydrology.export()

    nitrogen_params = NitrogenParameters(
        s_wp=20.0,
        s_max=147.3,
        smf_sat=0.8,
        beta_sm=1.0,
        rel_saturation_low=0.2,
        rel_saturation_high=0.9,
        rel_sat_limit_exp=0.7,
        beta_exp=2.5,
        v_degrad_son=1e-5,
        v_dissol_son=1e-5,
        v_dissol_fon=1e-3,
        v_min_fon=1e-3,
        v_denit=5e-2,
        k_denit=1.5,
        uptake_demand=10.0,
        delta_time_solver=1.0 / 24.0,
        freundlich_exponent=1.0,
        freundlich_constant=100.0,
        soil_bulk_density=1.3,
    )
    nitrogen_initial_states = NitrogenStates(
        m_don=500.0,
        m_din=2500.0,
        m_son=4.5e5,
        m_fon=1.0e4,
        m_don_ads=0.0,
    )
    nitrogen = Nitrogen(
        output_dir=OUTPUT_DIR,
        params=nitrogen_params,
        initial_states=nitrogen_initial_states,
        nitrogen_forcing_path=NITROGEN_FORCING_PARQUET,
    )
    nitrogen.load_hydrology(
        hydrology.output_dir,
        artifact_names=HYDROLOGY_ARTIFACTS,
    )
    nitrogen.solve(progress=progress)
    nitrogen.export()

    states, fluxes, meteorology = hydrology.load_outputs()
    if (
        nitrogen.df_forcings is None
        or nitrogen.solution_ads is None
        or nitrogen.solution_no_ads is None
        or nitrogen.mass_fluxes is None
    ):
        raise RuntimeError("Nitrogen.solve() did not produce outputs.")

    plot_hydrologic_forcings(states, fluxes, meteorology)
    plot_nitrogen_solution(nitrogen.solution_ads, nitrogen.solution_no_ads)
    plot_mass_fluxes(nitrogen.mass_fluxes)

    print("Nitrogen simulation demo")
    print(f"Hydrology outputs: {hydrology.source}")
    print(f"Forcing rows: {len(nitrogen.df_forcings)}")
    print(
        f"Final DIN concentration: {nitrogen.solution_ads['c_din'].iloc[-1]:.3f} mg N/L"
    )
    print(
        f"Final DON concentration: {nitrogen.solution_ads['c_don'].iloc[-1]:.3f} mg N/L"
    )
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
