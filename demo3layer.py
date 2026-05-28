"""Run a three-compartment nitrogen routing demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from demo1layer import HYDROLOGY_ARTIFACTS
from demo1layer import HYDROLOGY_FORCING_PARQUET
from demo1layer import HYDROLOGY_OUTPUT_DIR
from demo1layer import OUTPUT_DIR
from demo1layer import plot_hydrologic_forcings

from devcon2026.hydrology import Hydrology
from devcon2026.hydrology import HydrologyParameters
from devcon2026.hydrology import HydrologyStates
from devcon2026.hydrology.export import read_table
from devcon2026.hydrology.export import write_table
from devcon2026.nitrogen import Nitrogen
from devcon2026.nitrogen import NitrogenParameters
from devcon2026.nitrogen import NitrogenThreeCompartment

NITROGEN_FORCING_PARQUET = Path("data/nitrogen_forcings.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-hydrology",
        action="store_true",
        help="rerun hydrology even when exported artifacts exist",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable hydrologic and nitrogen progress bars",
    )
    return parser.parse_args()


def hydrology_parameters() -> HydrologyParameters:
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


def nitrogen_parameters() -> NitrogenParameters:
    return NitrogenParameters(
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


def plot_three_compartment_solution(solution: pd.DataFrame) -> None:
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
        ax.plot(solution["time"], solution[column], linewidth=0.7)
        ax.set_ylabel(label)
    fig.savefig(OUTPUT_DIR / "nitrogen_3layer_solution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    progress = not args.no_progress
    OUTPUT_DIR.mkdir(exist_ok=True)

    hydrology = Hydrology(
        output_dir=HYDROLOGY_OUTPUT_DIR,
        artifact_names=HYDROLOGY_ARTIFACTS,
        params=hydrology_parameters(),
        initial_states=HydrologyStates(
            s_sn=0.01,
            s_s=0.03,
            s_gwa=0.2,
            s_gwp=0.5,
        ),
        forcing_path=HYDROLOGY_FORCING_PARQUET,
    )
    hydrology.solve(force=args.force_hydrology, progress=progress)
    hydrology.export()

    states, fluxes, meteorology = hydrology.load_outputs()
    model = NitrogenThreeCompartment(nitrogen_parameters())
    df_forcings = model.from_hydrology_outputs(states, fluxes, meteorology)
    source_helper = Nitrogen(params=model.params)
    df_forcings = source_helper.add_nitrogen_source_forcings(
        df_forcings,
        NITROGEN_FORCING_PARQUET,
    )
    solution = model.simulate(
        df_forcings,
        with_soil_don_adsorption=True,
        progress=progress,
    )

    solution_path = OUTPUT_DIR / "nitrogen_3layer_solution.parquet"
    forcing_path = OUTPUT_DIR / "nitrogen_3layer_forcings.parquet"
    write_table(solution, solution_path)
    write_table(df_forcings, forcing_path)
    plot_hydrologic_forcings(states, fluxes, meteorology)
    plot_three_compartment_solution(solution)

    saved_solution = read_table(solution_path, parse_dates=["time"])
    print("Three-compartment nitrogen simulation demo")
    print(f"Hydrology outputs: {hydrology.source}")
    print(f"Forcing rows: {len(df_forcings)}")
    print(
        f"Final soil DIN concentration: {saved_solution['soil_c_din'].iloc[-1]:.3f} mg N/L"
    )
    print(
        f"Final active GW DIN concentration: {saved_solution['gwa_c_din'].iloc[-1]:.3f} mg N/L"
    )
    print(
        f"Final passive GW DIN concentration: {saved_solution['gwp_c_din'].iloc[-1]:.3f} mg N/L"
    )
    print(f"Saved three-compartment outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
