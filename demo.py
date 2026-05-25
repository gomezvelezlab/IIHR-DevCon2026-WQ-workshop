"""Run a dataframe-driven nitrogen simulation demo.

This mirrors the `origin/chucho` notebook workflow, but uses synthetic
hydrologic model outputs because the notebook's CSV inputs are not committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from devcon2026.hydrology import Parameters
from devcon2026.hydrology import States
from devcon2026.hydrology import export_nitrogen_hydrology_inputs
from devcon2026.hydrology import simulate
from devcon2026.hydrology.export import convert_fluxes_to_nitrogen_units
from devcon2026.hydrology.export import convert_states_to_nitrogen_units
from devcon2026.nitrogen import NitrogenModel_SingleCV
from devcon2026.nitrogen import default_soil_parameters

OUTPUT_DIR = Path("demo_outputs")
HYDROLOGY_OUTPUT_DIR = OUTPUT_DIR / "example_hydrology_model"


def synthetic_hydrology_forcing(hours: int = 24 * 120) -> pd.DataFrame:
    """Create model-ready forcing data for the hydrologic model."""
    time = pd.date_range("2010-01-01", periods=hours, freq="h")
    hour = np.arange(hours, dtype=float)
    seasonal = np.sin(2.0 * np.pi * hour / (24.0 * 365.0))
    storm = np.maximum(0.0, np.sin(2.0 * np.pi * hour / (24.0 * 9.0))) ** 4

    return pd.DataFrame(
        {
            "time": time,
            "TMP_2maboveground": 273.15 + 9.0 + 14.0 * seasonal,
            "temperature_2m_C": 9.0 + 14.0 * seasonal,
            "precipitation_mm": 5.0 * storm,
            "ref_et_mm_hr": 0.05 + 0.03 * np.maximum(0.0, seasonal),
        }
    )


def ensure_hydrology_outputs(
    output_dir: Path, *, force: bool = False, progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load hydrology CSVs if present; otherwise run hydrology and export them."""
    required = [
        output_dir / "states1.csv",
        output_dir / "fluxes1.csv",
        output_dir / "south_fork_aorc_forcing.csv",
    ]
    generated = force or not all(path.exists() for path in required)
    if generated:
        forcing = synthetic_hydrology_forcing()
        result = simulate(
            forcing_df=forcing,
            params=Parameters(),
            initial_states=States(s_sn=0.01, s_s=0.03, s_gwa=0.2, s_gwp=0.5),
            progress=progress,
            progress_desc="hydrology",
        )
        export_nitrogen_hydrology_inputs(result, forcing, output_dir)

    states = pd.read_csv(output_dir / "states1.csv", parse_dates=["time"])
    fluxes = pd.read_csv(output_dir / "fluxes1.csv", parse_dates=["time"])
    forcing = pd.read_csv(output_dir / "south_fork_aorc_forcing.csv", parse_dates=["time"])
    source = "generated from synthetic forcing" if generated else "loaded from existing CSVs"
    return states, fluxes, forcing, source


def soil_control_volume_forcing(states: pd.DataFrame, fluxes: pd.DataFrame, forcings: pd.DataFrame) -> pd.DataFrame:
    """Build the forcing dataframe expected by `NitrogenModel_SingleCV`."""
    states_mm = convert_states_to_nitrogen_units(states)
    fluxes_mm_day = convert_fluxes_to_nitrogen_units(fluxes)
    time = pd.DatetimeIndex(fluxes["time"])
    df = pd.DataFrame(
        {
            "time": time,
            "doy": time.dayofyear + time.hour / 24.0,
            "temp": forcings["TMP_2maboveground"].to_numpy() - 273.15,
            "s": states_mm["s_s"].to_numpy(),
            "q_in_1": fluxes_mm_day["p_r"].to_numpy(),
            "q_in_2": fluxes_mm_day["f_sm"].to_numpy(),
            "q_out_1": fluxes_mm_day["q_sc"].to_numpy(),
            "q_out_2": fluxes_mm_day["q_sgwa"].to_numpy(),
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


def run_simulation(
    df_forcings: pd.DataFrame, *, progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = NitrogenModel_SingleCV(default_soil_parameters())
    masses0 = initial_masses(df_forcings)
    solution_ads = model.simulate_nitrogen_dynamics(
        df_forcings=df_forcings,
        M0=masses0,
        with_DON_ads=True,
        progress=progress,
        progress_desc="nitrogen with DON adsorption",
    )
    solution_no_ads = model.simulate_nitrogen_dynamics(
        df_forcings=df_forcings,
        M0=masses0,
        with_DON_ads=False,
        progress=progress,
        progress_desc="nitrogen without DON adsorption",
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
    states, fluxes, meteorology, hydrology_source = ensure_hydrology_outputs(
        HYDROLOGY_OUTPUT_DIR,
        force=args.force_hydrology,
        progress=progress,
    )
    df_forcings = soil_control_volume_forcing(states, fluxes, meteorology)
    solution_ads, solution_no_ads, mass_fluxes = run_simulation(
        df_forcings,
        progress=progress,
    )

    plot_hydrologic_forcings(states, fluxes, meteorology)
    plot_nitrogen_solution(solution_ads, solution_no_ads)
    plot_mass_fluxes(mass_fluxes)

    print("Nitrogen simulation demo")
    print(f"Hydrology outputs: {hydrology_source}")
    print(f"Forcing rows: {len(df_forcings)}")
    print(f"Final DIN concentration: {solution_ads['c_din'].iloc[-1]:.3f} mg N/L")
    print(f"Final DON concentration: {solution_ads['c_don'].iloc[-1]:.3f} mg N/L")
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
