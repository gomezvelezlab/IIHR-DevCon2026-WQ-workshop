"""Run a dataframe-driven nitrogen simulation demo.

This mirrors the `origin/chucho` notebook workflow, but uses synthetic
hydrologic model outputs because the notebook's CSV inputs are not committed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from devcon2026.nitrogen import NitrogenModel_SingleCV
from devcon2026.nitrogen import default_soil_parameters

OUTPUT_DIR = Path("demo_outputs")


def synthetic_hydrologic_outputs(hours: int = 24 * 120) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create hydrologic states, fluxes, and meteorological forcings."""
    time = pd.date_range("2010-01-01", periods=hours, freq="h")
    hour = np.arange(hours, dtype=float)
    seasonal = np.sin(2.0 * np.pi * hour / (24.0 * 365.0))
    storm = np.maximum(0.0, np.sin(2.0 * np.pi * hour / (24.0 * 9.0))) ** 4

    states = pd.DataFrame(
        {
            "time": time,
            "s_s": 90.0 + 20.0 * seasonal + 12.0 * storm,
            "s_gwa": 300.0 + 8.0 * seasonal,
            "s_gwp": 900.0 + 3.0 * seasonal,
        }
    )
    fluxes = pd.DataFrame(
        {
            "time": time,
            "p_r": 5.0 * storm,
            "f_sm": np.where((time.month <= 3) | (time.month == 12), 0.15 * storm, 0.0),
            "e_a": 1.4 + 0.8 * np.maximum(0.0, seasonal),
            "q_sc": 0.7 * storm,
            "q_sgwa": 0.25 + 0.03 * states["s_s"],
            "q_gwatd": 0.08 + 0.005 * states["s_gwa"],
            "q_gwac": 0.05 + 0.002 * states["s_gwa"],
            "q_gwap": 0.02 + 0.001 * states["s_gwa"],
            "q_gwpc": 0.01 + 0.0005 * states["s_gwp"],
        }
    )
    forcings = pd.DataFrame(
        {
            "time": time,
            "TMP_2maboveground": 273.15 + 9.0 + 14.0 * seasonal,
        }
    )
    return states, fluxes, forcings


def soil_control_volume_forcing(
    states: pd.DataFrame, fluxes: pd.DataFrame, forcings: pd.DataFrame
) -> pd.DataFrame:
    """Build the forcing dataframe expected by `NitrogenModel_SingleCV`."""
    time = pd.DatetimeIndex(fluxes["time"])
    df = pd.DataFrame(
        {
            "time": time,
            "doy": time.dayofyear + time.hour / 24.0,
            "temp": forcings["TMP_2maboveground"].to_numpy() - 273.15,
            "s": states["s_s"].to_numpy(),
            "q_in_1": fluxes["p_r"].to_numpy(),
            "q_in_2": fluxes["f_sm"].to_numpy(),
            "q_out_1": fluxes["q_sc"].to_numpy(),
            "q_out_2": fluxes["q_sgwa"].to_numpy(),
            "c_din_in_0": 1.0,
            "c_din_in_1": 0.5,
            "c_don_in_0": 0.0,
            "c_don_in_1": 0.0,
        }
    )
    return df


def initial_masses(forcings: pd.DataFrame) -> np.ndarray:
    """Initial [DON, DIN, SON, FON, DON adsorbed] masses in kg N/km2."""
    mean_storage = float(forcings["s"].mean())
    return np.array(
        [
            5.0 * mean_storage,
            25.0 * mean_storage,
            4.5e5,
            1.0e4,
            0.0,
        ],
        dtype=float,
    )


def run_simulation(df_forcings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = NitrogenModel_SingleCV(default_soil_parameters())
    masses0 = initial_masses(df_forcings)
    solution_ads = model.simulate_nitrogen_dynamics(
        df_forcings=df_forcings,
        M0=masses0,
        with_DON_ads=True,
        progress=False,
    )
    solution_no_ads = model.simulate_nitrogen_dynamics(
        df_forcings=df_forcings,
        M0=masses0,
        with_DON_ads=False,
        progress=False,
    )
    mass_fluxes = model.get_mass_fluxes_all_species(
        M=solution_ads[["m_don", "m_din", "m_son", "m_fon"]].to_numpy(),
        df_forcings=df_forcings,
    )
    mass_fluxes.insert(0, "time", df_forcings["time"].to_numpy())
    return solution_ads, solution_no_ads, mass_fluxes


def plot_hydrologic_forcings(
    states: pd.DataFrame, fluxes: pd.DataFrame, forcings: pd.DataFrame
) -> None:
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    axs[0].plot(states["time"], states["s_s"], linewidth=0.7)
    axs[0].set_ylabel("Soil storage (mm)")
    axs[1].plot(fluxes["time"], fluxes["p_r"], linewidth=0.7, label="Rainfall")
    axs[1].plot(fluxes["time"], fluxes["f_sm"], linewidth=0.7, label="Snowmelt")
    axs[1].set_ylabel("Fluxes in (mm/day)")
    axs[1].legend()
    axs[2].plot(fluxes["time"], fluxes["q_sc"], linewidth=0.7, label="Flow to channel")
    axs[2].plot(
        fluxes["time"],
        fluxes["q_sgwa"],
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


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    states, fluxes, meteorology = synthetic_hydrologic_outputs()
    df_forcings = soil_control_volume_forcing(states, fluxes, meteorology)
    solution_ads, solution_no_ads, mass_fluxes = run_simulation(df_forcings)

    plot_hydrologic_forcings(states, fluxes, meteorology)
    plot_nitrogen_solution(solution_ads, solution_no_ads)
    plot_mass_fluxes(mass_fluxes)

    print("Nitrogen simulation demo")
    print(f"Forcing rows: {len(df_forcings)}")
    print(f"Final DIN concentration: {solution_ads['c_din'].iloc[-1]:.3f} mg N/L")
    print(f"Final DON concentration: {solution_ads['c_don'].iloc[-1]:.3f} mg N/L")
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
