"""Core model data structures."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from typing import Literal

import pandas as pd

from devcon2026.types import ArrayConvertible
from .constants import SECONDS_PER_HOUR

TileDrainageMethod = Literal["relative_storage", "water_table", "none"]


@dataclass
class HydrologySimulationResult:
    """Container for main simulation outputs indexed by model time."""

    discharge_cms: pd.Series
    states: pd.DataFrame
    fluxes: pd.DataFrame


@dataclass(kw_only=True)
class HydrologyStates(ArrayConvertible):
    """Hydrologic state storages."""

    s_sn: float = field(metadata={"unit": "m", "description": "snow storage"})
    s_s: float = field(metadata={"unit": "m", "description": "soil storage"})
    s_gwa: float = field(metadata={"unit": "m", "description": "active GW storage"})
    s_gwp: float = field(metadata={"unit": "m", "description": "passive GW storage"})


@dataclass(kw_only=True)
class HydrologyDerivatives(ArrayConvertible):
    """Time-derivatives of hydrologic state storages."""

    ds_sn: float = field(
        metadata={"unit": "m/s", "description": "change in snow storage"}
    )
    ds_s: float = field(
        metadata={"unit": "m/s", "description": "change in soil storage"}
    )
    ds_gwa: float = field(
        metadata={"unit": "m/s", "description": "change in active groundwater storage"}
    )
    ds_gwp: float = field(
        metadata={"unit": "m/s", "description": "change in passive groundwater storage"}
    )


@dataclass
class HydrologyFluxes:
    """Instantaneous hydrologic fluxes for a model time step."""

    p_sn: float = field(metadata={"unit": "m/s", "description": "snow precipitation"})
    f_sm: float = field(metadata={"unit": "m/s", "description": "snowmelt"})
    p_r: float = field(metadata={"unit": "m/s", "description": "rain precipitation"})
    e_a: float = field(
        metadata={"unit": "m/s", "description": "actual evapotranspiration"}
    )
    q_sc: float = field(metadata={"unit": "m/s", "description": "surface runoff"})
    q_sgwa: float = field(
        metadata={"unit": "m/s", "description": "subsurface GW inflow"}
    )
    q_gwatd: float = field(
        metadata={"unit": "m/s", "description": "GW to tile drainage"}
    )
    q_gwac: float = field(
        metadata={"unit": "m/s", "description": "GW active to channel"}
    )
    q_gwap: float = field(
        metadata={"unit": "m/s", "description": "GW active to passive"}
    )
    q_gwpc: float = field(
        metadata={"unit": "m/s", "description": "GW passive to channel"}
    )

    def fluxes_into_channel(self) -> float:
        """Total lateral flux routed to the channel network [m/s]."""
        return self.q_sc + self.q_gwac + self.q_gwatd + self.q_gwpc

    def compute_derivatives(self) -> "HydrologyDerivatives":
        """Compute state derivatives implied by the flux balance."""
        return HydrologyDerivatives(
            ds_sn=self.p_sn - self.f_sm,
            ds_s=self.p_r + self.f_sm - self.e_a - self.q_sc - self.q_sgwa,
            ds_gwa=self.q_sgwa - self.q_gwatd - self.q_gwac - self.q_gwap,
            ds_gwp=self.q_gwap - self.q_gwpc,
        )


@dataclass
class HydrologyParameters:
    """Model parameter set grouped by process family."""

    t_0: float = field(
        default=0.0,
        metadata={"unit": "°C", "description": "snow-rain temperature threshold"},
    )
    m_sn: float = field(
        default=0.002,
        metadata={"unit": "m", "description": "snowmelt storage scale"},
    )
    k_sn: float = field(
        default=1.157e-7,
        metadata={"unit": "m/s/°C", "description": "degree-day snowmelt coefficient"},
    )

    c_e: float = field(
        default=0.8,
        metadata={"unit": "1", "description": "actual ET correction coefficient"},
    )
    s_max: float = field(
        default=0.05,
        metadata={"unit": "m", "description": "maximum soil storage"},
    )
    m_s: float = field(
        default=1e-5,
        metadata={"unit": "1", "description": "soil ET shape parameter"},
    )
    beta_s: float = field(
        default=2.0,
        metadata={"unit": "1", "description": "soil wetness exponent"},
    )
    k_sgw: float = field(
        default=1e-7,
        metadata={"unit": "m/s", "description": "vertical groundwater recharge flux scale"},
    )

    k_gwpc: float = field(
        default=1e-7,
        metadata={"unit": "1/s", "description": "passive groundwater recession coefficient"},
    )
    k_gwap: float = field(
        default=1e-6,
        metadata={"unit": "1/s", "description": "active-to-passive groundwater transfer coefficient"},
    )
    k_gwac: float = field(
        default=1e-6,
        metadata={"unit": "m/s", "description": "active groundwater channel flux scale"},
    )
    beta_gwac: float = field(
        default=2.0,
        metadata={"unit": "1", "description": "active groundwater channel exponent"},
    )
    s_gwa_max: float = field(
        default=1.0,
        metadata={"unit": "m", "description": "maximum active groundwater storage"},
    )
    s_ref_td: float = field(
        default=0.5,
        metadata={"unit": "1", "description": "legacy relative tile drainage activation threshold"},
    )
    tile_drainage_method: TileDrainageMethod = field(
        default="water_table",
        metadata={"unit": "method", "description": "tile drainage formulation"},
    )
    water_table_reference_depth: float = field(
        default=1.8,
        metadata={"unit": "m", "description": "water table depth when active groundwater storage is zero"},
    )
    tile_depth: float = field(
        default=1.0,
        metadata={"unit": "m", "description": "tile drain depth below land surface"},
    )
    specific_yield: float = field(
        default=0.1,
        metadata={"unit": "1", "description": "drainable porosity converting active groundwater storage to water-table rise"},
    )
    k_td: float = field(
        default=1e-5,
        metadata={"unit": "1/s", "description": "tile drainage coefficient"},
    )

    gamma_x: float = field(
        default=-0.34,
        metadata={"unit": "1", "description": "seasonal infiltration partition offset"},
    )
    gamma_i: float = field(
        default=0.32,
        metadata={"unit": "1", "description": "seasonal infiltration partition amplitude"},
    )
    gamma_p: float = field(
        default=336.0,
        metadata={"unit": "day", "description": "seasonal infiltration partition phase"},
    )

    pet_albedo: float = field(
        default=0.23,
        metadata={"unit": "1", "description": "surface albedo for reference ET"},
    )
    pet_emissivity: float = field(
        default=0.98,
        metadata={"unit": "1", "description": "surface emissivity for reference ET"},
    )

    n_gu: float = field(
        default=2.0,
        metadata={"unit": "1", "description": "gamma unit hydrograph shape parameter"},
    )
    a_gu_seconds: float = field(
        default=2 * SECONDS_PER_HOUR,
        metadata={"unit": "s", "description": "gamma unit hydrograph scale parameter"},
    )

    area_km2: float = field(
        default=100.0,
        metadata={"unit": "km2", "description": "drainage area"},
    )

    def with_updates(self, updates: dict[str, Any]) -> "HydrologyParameters":
        """Return a copy with selected parameter updates."""
        return replace(self, **updates)


@dataclass(kw_only=True)
class HydrologyForcings:
    """Atmospheric forcing variables supplied at each simulation step."""

    p_t: float = field(metadata={"unit": "m/s", "description": "total precipitation"})
    t: float = field(metadata={"unit": "°C", "description": "air temperature"})
    e_p: float = field(
        metadata={"unit": "m/s", "description": "potential evapotranspiration"}
    )
