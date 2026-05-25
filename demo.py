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
from devcon2026.hydrology.export import convert_fluxes_to_nitrogen_units
from devcon2026.hydrology.export import convert_states_to_nitrogen_units
from devcon2026.nitrogen import Nitrogen

OUTPUT_DIR = Path("demo_outputs")
HYDROLOGY_OUTPUT_DIR = OUTPUT_DIR / "example_hydrology_model"


def plot_hydrologic_forcings(
    states: pd.DataFrame, fluxes: pd.DataFrame, forcings: pd.DataFrame
) -> None:
    states_mm = convert_states_to_nitrogen_units(states)
    fluxes_mm_day = convert_fluxes_to_nitrogen_units(fluxes)
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    axs[0].plot(states_mm["time"], states_mm["s_s"], linewidth=0.7)
    axs[0].set_ylabel("Soil storage (mm)")
    axs[1].plot(fluxes_mm_day["time"], fluxes_mm_day["p_r"], linewidth=0.7, label="Rainfall")
    axs[1].plot(fluxes_mm_day["time"], fluxes_mm_day["f_sm"], linewidth=0.7, label="Snowmelt")
    axs[1].set_ylabel("Fluxes in (mm/day)")
    axs[1].legend()
    axs[2].plot(fluxes_mm_day["time"], fluxes_mm_day["q_sc"], linewidth=0.7, label="Flow to channel")
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


def plot_nitrogen_solution(solution_ads: pd.DataFrame, solution_no_ads: pd.DataFrame) -> None:
    variables = ["m_don", "m_din", "m_son", "m_fon", "m_don_ads", "c_din", "c_don"]
    fig, axs = plt.subplots(
        len(variables), 1, figsize=(10, 2.2 * len(variables)), sharex=True, layout="constrained"
    )
    for ax, variable in zip(axs, variables):
        ax.plot(solution_ads["time"], solution_ads[variable], linewidth=0.7, label="With DON adsorption")
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
        len(variables), 1, figsize=(10, 2.2 * len(variables)), sharex=True, layout="constrained"
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

    hydrology = Hydrology()
    hydrology.config(output_dir=HYDROLOGY_OUTPUT_DIR)
    hydrology.solve(force=args.force_hydrology, progress=progress)
    hydrology.export()

    nitrogen = Nitrogen()
    nitrogen.config(output_dir=OUTPUT_DIR)
    nitrogen.load_hydrology(hydrology.output_dir)
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
    print(f"Final DIN concentration: {nitrogen.solution_ads['c_din'].iloc[-1]:.3f} mg N/L")
    print(f"Final DON concentration: {nitrogen.solution_ads['c_don'].iloc[-1]:.3f} mg N/L")
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
