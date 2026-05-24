"""Reproduce the nitrogen-model exploration notebook with the package API."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from devcon2026.nitrogen import (
    D_DIN,
    Params,
    U_DIN,
    concfactor,
    exponential_moisturefactor,
    moisturefactor,
    tempfactor,
)

OUTPUT_DIR = Path("demo_outputs")
LABEL_FONTSIZE = 10
LINEWIDTH = 1


def plot_temperature_factor(output_dir: Path) -> None:
    temperatures = np.arange(-5.0, 35.0, 0.5)
    factors = [tempfactor(float(temp)) for temp in temperatures]

    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
    ax.plot(temperatures, factors, color="black", linewidth=LINEWIDTH)
    ax.set_xlabel("Temperature (C)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Temperature Factor (-)", fontsize=LABEL_FONTSIZE)
    ax.axvline(x=0.0, color="red", linestyle="--", label="Threshold at 0 C")
    ax.axvline(x=5.0, color="orange", linestyle="--", label="Threshold at 5 C")
    ax.tick_params(axis="both", labelsize=LABEL_FONTSIZE - 2)
    ax.legend(fontsize=LABEL_FONTSIZE - 2)
    fig.savefig(output_dir / "temperature_factor.png", dpi=150)
    plt.close(fig)


def plot_moisture_factors(output_dir: Path) -> None:
    params = Params(
        S_s_max=1000.0,
        S_wp=250.0,
        smf_sat=0.6,
        beta_sm=2.5,
        rel_saturation_low=0.1,
        rel_saturation_high=0.3,
        rel_sat_limit_exp=0.7,
        beta_exp=2.5,
    )
    soil_storage = np.arange(0.0, params.S_s_max, 10.0)
    relative_storage = soil_storage / params.S_s_max
    moisture_factors = [
        moisturefactor(float(storage), params) for storage in soil_storage
    ]
    exponential_factors = [
        exponential_moisturefactor(float(storage), params) for storage in soil_storage
    ]

    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
    ax.plot(
        relative_storage,
        moisture_factors,
        color="black",
        linewidth=LINEWIDTH,
        label="Soil moisture factor",
    )
    ax.plot(
        relative_storage,
        exponential_factors,
        color="blue",
        linewidth=LINEWIDTH,
        label="Exponential soil moisture factor",
    )
    ax.set_xlabel("Relative soil storage, S_s/S_s_max (-)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Soil Moisture Factor (-)", fontsize=LABEL_FONTSIZE)
    ax.axvline(
        x=params.S_wp / params.S_s_max,
        color="red",
        linestyle="--",
        label="Wilting point",
    )
    ax.tick_params(axis="both", labelsize=LABEL_FONTSIZE - 2)
    ax.legend(fontsize=LABEL_FONTSIZE - 2)
    fig.savefig(output_dir / "moisture_factors.png", dpi=150)
    plt.close(fig)


def plot_concentration_factor(output_dir: Path) -> None:
    half_saturation = 1.5
    concentrations = np.arange(0.0, 20.0, 0.2)
    factors = [
        concfactor(float(concentration), half_saturation)
        for concentration in concentrations
    ]

    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
    ax.plot(concentrations, factors, color="black", linewidth=LINEWIDTH)
    ax.set_xlabel("Concentration (mg/L)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Concentration Factor (-)", fontsize=LABEL_FONTSIZE)
    ax.axvline(
        x=half_saturation,
        color="darkgrey",
        linestyle="--",
        label="Half-saturation parameter",
    )
    ax.axhline(y=0.5, color="grey", linestyle="--", label="Concentration factor = 0.5")
    ax.axhline(y=1.0, color="red", linestyle="--", label="Concentration factor = 1.0")
    ax.tick_params(axis="both", labelsize=LABEL_FONTSIZE - 2)
    ax.legend(fontsize=LABEL_FONTSIZE - 2)
    fig.savefig(output_dir / "concentration_factor.png", dpi=150)
    plt.close(fig)


def print_rate_examples() -> None:
    params = Params()
    concentration_din = 0.5
    soil_storage = 100.0
    temperature = 20.0

    print("Example nitrogen rates")
    print(f"D_DIN: {D_DIN(concentration_din, soil_storage, temperature, params):.6g}")
    print(f"U_DIN: {U_DIN(concentration_din, soil_storage, params):.6g}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_temperature_factor(OUTPUT_DIR)
    plot_moisture_factors(OUTPUT_DIR)
    plot_concentration_factor(OUTPUT_DIR)
    print_rate_examples()
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
