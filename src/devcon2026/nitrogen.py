# -----------------------------------------------------------------------------
# Author: Jesus Gomez-Velez (University of Iowa)
# Contact: jesus-gomezvelez@uiowa.edu
# Created: 2026-02-15
# Description: HYPE model.
# -----------------------------------------------------------------------------

"""Nitrogen soil processes from the HYPE model.

The functions are based on the mathematical descriptions of the HYPE model and
are designed to be modular and reusable. Concentrations of dissolved
constituents are in mg/L, soil pools are in kg/km2, and soil water is in mm.
Recall that 1 mm*mg/(L*day) = 1 kg/(km2*day).
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Ix",
    "Params",
    "D_DIN",
    "Q_DIN",
    "Q_FON",
    "Q_SON",
    "R_DIN",
    "R_DON",
    "R_FON",
    "R_SON",
    "R_degrad_SON",
    "R_dissol_FON",
    "R_dissol_SON",
    "R_min_FON",
    "U_DIN",
    "concfactor",
    "derivatives",
    "exponential_moisturefactor",
    "moisturefactor",
    "tempfactor",
]


@dataclass
class Params:
    S_wp: float = 20.0  # Wilting point storage (mm)
    S_s_max: float = 100.0  # Maximum soil storage (mm) -- Assumed to equal the thickness of soil layer (m) (thickm in HYPE)
    smf_sat: float = 0.8  # Saturated moisture factor (satact)
    beta_sm: float = 1.0  # Exponent of moisture factor (thetapow)
    rel_saturation_low: float = 0.2  # Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
    rel_saturation_high: float = 0.9  # High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100
    rel_sat_limit_exp: float = 0.7  # Limitation parameter of moisture factor (dimensionless)
    beta_exp: float = 2.5  # Exponent of exponential moisture factor
    V_degrad_SON: float = 0.001  # Maximum degradation rate of soil slow organic nitrogen (1/day) -- This is the parameter degrhNpar in HYPE.
    V_dissol_SON: float = 0.01  # Maximum dissolution rate of soil slow organic nitrogen (1/day) -- This is the parameter dissolhNpar in HYPE.
    V_dissol_FON: float = 0.005  # Maximum dissolution rate of soil fast organic nitrogen (1/day) -- This is the parameter dissolfNpar in HYPE.
    V_min_FON: float = 0.02  # Maximum mineralization rate of soil fast organic nitrogen (1/day) -- This is the parameter minfNpar in HYPE.
    V_denit: float = 0.05  # Maximum denitrification rate (1/day)
    K_denit: float = 1.5  # Half-saturation parameter (mg/L)
    uptake_demand: float = 10.0  # Plant uptake demand (kg/km2/d)
    delta_time_solver: float = 1.0  # Time step (day)


def tempfactor(temp: float) -> float:
    """
    This is gamma_T in the mathematical description of our version of the HYPE model.
    Calculate a temperature factor based on Q10=2
    Reference rate at 20°C with thresholds at 0°C and 5°C.

    Args:
        temp: Temperature (°C)

    Returns:
        Temperature factor (dimensionless).
        Monotonically increasing function with a value of 2.73 @ T=20°C.
    """
    if temp < 0.0:
        return 0.0

    tempf = 2.0 ** ((temp - 20.0) / 10.0)

    if temp < 5.0:
        tempf *= temp / 5.0

    return tempf


def moisturefactor(
    S_s: float,
    params: Params,
) -> float:
    """
    Calculate a soil moisture dependence factor.

    Args:
        S_s: Soil storage (mm)
        params: Model parameters including:
            S_wp: Water content at wilting point (mm)
            S_s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100

    Returns:
        Soil moisture dependence factor (dimensionless, 0-1)
    """

    if S_s <= params.S_wp:
        return 0.0
    if S_s >= params.S_s_max:
        return params.smf_sat

    term1 = (1.0 - params.smf_sat) * (
        (params.S_s_max - S_s) / (params.rel_saturation_high * params.S_s_max)
    ) ** params.beta_sm + params.smf_sat
    term2 = (
        (S_s - params.S_wp) / (params.rel_saturation_low * params.S_s_max)
    ) ** params.beta_sm
    return min(1.0, term1, term2)


def concfactor(C: float, K: float) -> float:
    """
    Calculate a concentration dependence factor based on half saturation function.
    This is a Michaelis-Menten Kinetics function. Equation: $f = c /(c + K}$ with
    $c$ being the concentration and $K$ the half-saturation parameter (or Michaelis constant).
    The constant $K$ represents the concentration at which the f is exactly $1/2$.

    Args:
        C: Current concentration (mg/L)
        K: Half-saturation parameter (mg/L)

    Returns:
        Concentration factor (dimensionless, 0-1)
    """

    return C / (C + K)


def exponential_moisturefactor(
    S_s: float,
    params: Params,
) -> float:
    """
    Calculate an exponential soil moisture dependence factor for denitrification.

    Args:
        S_s: Soil storage (mm)
        params: Model parameters including:
            S_s_max: Maximum soil storage (mm)
            rel_sat_limit_exp: Limitation parameter of moisture factor (dimensionless)
            beta_exp: Exponent of exponential moisture factor

    Returns:
        Exponential moisture factor (dimensionless)
    """
    rel_sat = S_s / params.S_s_max

    if rel_sat > 1.0:
        return 1.0

    if rel_sat > params.rel_sat_limit_exp:
        return (
            (rel_sat - params.rel_sat_limit_exp)
            / (1.0 - params.rel_sat_limit_exp)
        ) ** params.beta_exp

    return 0.0


def D_DIN(
    C_DIN: float,
    S_s: float,
    temp: float,
    params: Params,
) -> float:
    """
    Denitrification of inorganic nitrogen in one soil layer.
    HYPE name: soil_denitrification

    Args:
        C_DIN: Inorganic N concentration (mg/L)
        S_s: Soil water (mm)
        temp: Soil temperature (°C)
        params: Model parameters including:
            V_denit: Maximum denitrification rate (1/day)
            K_denit: Half-saturation parameter (mg/L)
            S_s_max: Maximum soil storage (mm)
            rel_sat_limit_exp: Limitation parameter of moisture factor (dimensionless)
            beta_exp: Exponent of exponential moisture factor

    Returns:
        Denitrification rate (kg N/km2/d)
    """

    # Environmental factors (HYPE: tmpfcn, smfcn, concfcn)
    tmpf = tempfactor(temp)
    smf = exponential_moisturefactor(S_s, params)
    concf = concfactor(C_DIN, params.K_denit)

    # Denitrification (HYPE: sink)
    D_DIN = params.V_denit * C_DIN * S_s * tmpf * smf * concf

    return D_DIN


def U_DIN(
    C_DIN: float,
    S_s: float,
    params: Params,
) -> float:
    """
    Calculate plant nutrient uptake from soil.
    Operates on first two soil layers only.

    Args:
        C_DIN: Inorganic N concentration (mg/L)
        S_s: Soil water (mm)
        params: Model parameters including:
            S_wp: Water content at wilting point (mm)
            uptake_demand: Plant uptake demand (kg/km2/d)
            delta_time_solver: Time step (day)

    Returns:
        uptake: Plant uptake (kg N/km2/d)
    """

    # Maximum uptake fraction (limited by wilting point)
    max_uptake_frac = (
        (1 / params.delta_time_solver) * (S_s - params.S_wp) / S_s
        if S_s > 0
        else 0.0
    )

    # Actual uptake (limited by demand and available pool)
    # available inorganic N pool (kg/km2)
    pool = C_DIN * S_s
    # actual uptake limited by demand and available pool
    allowed = max_uptake_frac * pool
    uptake = params.uptake_demand if params.uptake_demand <= allowed else allowed

    return uptake


def R_degrad_SON(
    M_SON: float,
    S_s: float,
    temp: float,
    params: Params,
) -> float:
    """
    Calculate the degradation rate of soil slow organic nitrogen (SON).

    Args:
        M_SON: Mass of soil slow organic nitrogen (kg/km2)
        S_s: Soil water (mm)
        temp: Soil temperature (°C)
        params: Model parameters including:
            V_degrad_SON: Maximum degradation rate of soil slow organic nitrogen (1/day) -- This is the parameter degrhNpar in HYPE.
            S_wp: Water content at wilting point (mm)
            S_s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100

    Returns:
        Degradation rate of soil slow organic nitrogen (kg N/km2/d)
    """

    # Environmental factors
    tmpf = tempfactor(temp)
    smf = moisturefactor(S_s, params)

    # Degradation rate
    reaction_rate = params.V_degrad_SON * M_SON * tmpf * smf

    return reaction_rate


def R_dissol_SON(
    M_SON: float,
    S_s: float,
    temp: float,
    params: Params,
) -> float:
    """
    Calculate the dissolution rate of soil slow organic nitrogen (SON).

    Args:
        M_SON: Mass of soil slow organic nitrogen (kg/km2)
        S_s: Soil water (mm)
        temp: Soil temperature (°C)
        params: Model parameters including:
            V_dissol_SON: Maximum dissolution rate of soil slow organic nitrogen (1/day) -- This is the parameter dissolhNpar in HYPE.
            S_wp: Water content at wilting point (mm)
            S_s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100

    Returns:
        Dissolution rate of soil slow organic nitrogen (kg N/km2/d)
    """

    # Environmental factors
    tmpf = tempfactor(temp)
    smf = moisturefactor(S_s, params)

    # Dissolution rate
    reaction_rate = params.V_dissol_SON * M_SON * tmpf * smf

    return reaction_rate


def R_dissol_FON(
    M_FON: float,
    S_s: float,
    temp: float,
    params: Params,
) -> float:
    """
    Calculate the dissolution rate of soil fast organic nitrogen (FON).

    Args:
        M_FON: Mass of soil fast organic nitrogen (kg/km2)
        S_s: Soil water (mm)
        temp: Soil temperature (°C)
        params: Model parameters including:
            V_dissol_FON: Maximum dissolution rate of soil fast organic nitrogen (1/day) -- This is the parameter dissolfNpar in HYPE.
            S_wp: Water content at wilting point (mm)
            S_s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100

    Returns:
        Dissolution rate of soil fast organic nitrogen (kg N/km2/d)
    """

    # Environmental factors
    tmpf = tempfactor(temp)
    smf = moisturefactor(S_s, params)

    # Dissolution rate
    reaction_rate = params.V_dissol_FON * M_FON * tmpf * smf

    return reaction_rate


def R_min_FON(
    M_FON: float,
    S_s: float,
    temp: float,
    params: Params,
) -> float:
    """
    Calculate the mineralization rate of soil fast organic nitrogen (FON).

    Args:
        M_FON: Mass of soil fast organic nitrogen (kg/km2)
        S_s: Soil water (mm)
        temp: Soil temperature (°C)
        params: Model parameters including:
            V_min_FON: Maximum mineralization rate of soil fast organic nitrogen (1/day) -- This is the parameter minfNpar in HYPE.
            S_wp: Water content at wilting point (mm)
            S_s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100

    Returns:
        Mineralization rate of soil fast organic nitrogen (kg N/km2/d)
    """

    # Environmental factors
    tmpf = tempfactor(temp)
    smf = moisturefactor(S_s, params)

    # Mineralization rate
    reaction_rate = params.V_min_FON * M_FON * tmpf * smf

    return reaction_rate


def R_SON(
    M_SON: float,
    S_s: float,
    temp: float,
    params: Params,
) -> float:
    """
    Calculate the reactionrate of soil slow organic nitrogen (SON).

    Args:
        M_SON: Mass of soil slow organic nitrogen (kg/km2)
        S_s: Soil water (mm)
        temp: Soil temperature (°C)
        params: Model parameters including:
            V_degrad_SON: Maximum degradation rate of soil slow organic nitrogen (1/day) -- This is the parameter degrhNpar in HYPE.
            V_dissol_SON: Maximum dissolution rate of soil slow organic nitrogen (1/day) -- This is the parameter dissolhNpar in HYPE.
            S_wp: Water content at wilting point (mm)
            S_s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100

    Returns:
        Reaction rate of soil slow organic nitrogen (kg N/km2/d)
    """

    # Mineralization rate
    reaction_rate = -R_degrad_SON(M_SON, S_s, temp, params) - R_dissol_SON(
        M_SON, S_s, temp, params
    )

    return reaction_rate


def R_FON(
    M_FON: float,
    M_SON: float,
    S_s: float,
    temp: float,
    params: Params,
) -> float:
    """
    Calculate the reactionrate of soil fast organic nitrogen (FON).

    Args:
        M_FON: Mass of soil fast organic nitrogen (kg/km2)
        M_SON: Mass of soil slow organic nitrogen (kg/km2)
        S_s: Soil water (mm)
        temp: Soil temperature (°C)
        params: Model parameters including:
            V_degrad_SON: Maximum degradation rate of soil slow organic nitrogen (1/day) -- This is the parameter degrhNpar in HYPE.
            V_dissol_FON: Maximum dissolution rate of soil fast organic nitrogen (1/day) -- This is the parameter dissolfNpar in HYPE.
            V_min_FON: Maximum mineralization rate of soil fast organic nitrogen (1/day) -- This is the parameter minfNpar in HYPE.
            S_wp: Water content at wilting point (mm)
            S_s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100

    Returns:
        Reaction rate of soil fast organic nitrogen (kg N/km2/d)
    """

    # Mineralization rate
    reaction_rate = (
        R_degrad_SON(M_SON, S_s, temp, params)
        - R_dissol_FON(M_FON, S_s, temp, params)
        - R_min_FON(M_FON, S_s, temp, params)
    )

    return reaction_rate


def R_DON(
    M_FON: float,
    M_SON: float,
    S_s: float,
    temp: float,
    params: Params,
) -> float:
    """
    Calculate the reactionrate of soil dissolved organic nitrogen (DON).

    Args:
        M_FON: Mass of soil fast organic nitrogen (kg/km2)
        M_SON: Mass of soil slow organic nitrogen (kg/km2)
        S_s: Soil water (mm)
        temp: Soil temperature (°C)
        params: Model parameters including:
            V_dissol_SON: Maximum dissolution rate of soil slow organic nitrogen (1/day) -- This is the parameter dissolhNpar in HYPE.
            V_dissol_FON: Maximum dissolution rate of soil fast organic nitrogen (1/day) -- This is the parameter dissolfNpar in HYPE.
            S_wp: Water content at wilting point (mm)
            S_s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100

    Returns:
        Reaction rate of soil dissolved organic nitrogen (kg N/km2/d)
    """

    reaction_rate = R_dissol_SON(M_SON, S_s, temp, params) + R_dissol_FON(
        M_FON, S_s, temp, params
    )

    return reaction_rate


def R_DIN(
    M_FON: float,
    M_DIN: float,
    S_s: float,
    temp: float,
    params: Params,
) -> float:
    """
    Calculate the reactionrate of soil dissolved inorganic nitrogen (DIN).

    Args:
        M_FON: Mass of soil fast organic nitrogen (kg/km2)
        M_DIN: Mass of soil dissolved inorganic nitrogen (kg/km2)
        S_s: Soil water (mm)
        temp: Soil temperature (°C)
        params: Model parameters including:
            V_min_FON: Maximum mineralization rate of soil fast organic nitrogen (1/day) -- This is the parameter minfNpar in HYPE.
            S_wp: Water content at wilting point (mm)
            uptake_demand: Plant uptake demand (kg/km2/d)
            delta_time_solver: Time step (day)
            V_denit: Maximum denitrification rate (1/day)
            K_denit: Half-saturation concentration for denitrification (mg/L)
            S_s_max: Maximum soil storage (mm)
            rel_sat_limit_exp: Limitation parameter of moisture factor for denitrification (dimensionless)
            beta_exp: Exponent of exponential soil moisture factor for denitrification
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100

    Returns:
        Reaction rate of soil dissolved inorganic nitrogen (kg N/km2/d)
    """
    C_DIN = M_DIN / S_s if S_s > 0 else 0.0

    reaction_rate = (
        R_min_FON(M_FON, S_s, temp, params)
        - U_DIN(C_DIN, S_s, params)
        - D_DIN(C_DIN, S_s, temp, params)
    )

    return reaction_rate


def Q_SON() -> float:
    """
    Calculate the source/sink term of soil slow organic nitrogen (SON) due to plant litter input.
    This is a placeholder function that can be expanded to include more complex dynamics of plant litter input.

    Returns:
        Source/sink term of soil slow organic nitrogen (kg N/km2/d)
    """
    return 0.0


def Q_FON() -> float:
    """
    Calculate the source/sink term of soil fast organic nitrogen (FON) due to plant litter input, dry atmospheric deposition, and solid fertilizers and manure.
    This is a placeholder function that can be expanded to include more complex dynamics of plant litter input.

    Returns:
        Source/sink term of soil fast organic nitrogen (kg N/km2/d)
    """
    return 0.0


def Q_DIN() -> float:
    """
    Calculate the source/sink term of soil dissolved inorganic nitrogen (DIN) due to wet atmospheric deposition and liquid fertilizers and manure.
    This is a placeholder function that can be expanded to include more complex dynamics of plant litter input.

    Returns:
        Source/sink term of soil dissolved inorganic nitrogen (kg N/km2/d)
    """
    return 0.0


class Ix(IntEnum):
    """Index for soil nitrogen pools in the mass array M."""
    SON = 0
    FON = 1
    DON = 2
    DIN = 3


def derivatives(
    M: NDArray[np.floating], S_s: float, temp: float, params: Params
) -> NDArray[np.floating]:
    """
    Calculate the derivatives of the mass ODEs for soil slow organic nitrogen (SON), soil fast organic nitrogen (FON), and soil dissolved inorganic nitrogen (DIN).
    This is a placeholder function that can be expanded to include the actual calculations of the derivatives based on the reaction rates and source/sink terms.

    Args:
        M: Array of masses [M_SON, M_FON, M_DON, M_DIN] (kg/km2)
        S_s: Soil water (mm)
        temp: Soil temperature (°C)
        params: Model parameters

    Returns:
        Array of derivatives [dM_SON/dt, dM_FON/dt, dM_DON/dt, dM_DIN/dt] (kg N/km2/d)
    """

    dM_dt = np.array(
        [
            R_SON(M[Ix.SON], S_s, temp, params) + Q_SON(),  # dM_SON/dt
            R_FON(M[Ix.FON], M[Ix.SON], S_s, temp, params) + Q_FON(),  # dM_FON/dt
            R_DON(M[Ix.DON], M[Ix.SON], S_s, temp, params),  # dM_DON/dt
            R_DIN(M[Ix.FON], M[Ix.DIN], S_s, temp, params) + Q_DIN(),  # dM_DIN/dt
        ]
    )

    return dM_dt
