"""Core model data structures."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Type, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .constants import SECONDS_PER_HOUR

T = TypeVar("T", bound="ArrayConvertible")


@dataclass
class SimulationResult:
    """Container for main simulation outputs indexed by model time."""

    discharge_cms: pd.Series
    states: pd.DataFrame
    fluxes: pd.DataFrame


@dataclass
class ArrayConvertible:
    """Mixin to convert dataclass fields to/from numeric arrays for ODE solving."""

    def to_array(self) -> NDArray[np.float64]:
        """Serialize dataclass values to a float64 numpy vector."""
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=float)

    @classmethod
    def from_array(cls: Type[T], y: NDArray[np.floating[Any]]) -> T:
        """Instantiate dataclass from a positionally aligned numpy vector."""
        return cls(**{f.name: float(y[i]) for i, f in enumerate(fields(cls))})


@dataclass(kw_only=True)
class States(ArrayConvertible):
    """Hydrologic state storages."""

    s_sn: float = field(metadata={"unit": "m", "description": "snow storage"})
    s_s: float = field(metadata={"unit": "m", "description": "soil storage"})
    s_gwa: float = field(metadata={"unit": "m", "description": "active GW storage"})
    s_gwp: float = field(metadata={"unit": "m", "description": "passive GW storage"})


@dataclass(kw_only=True)
class Derivatives(ArrayConvertible):
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
class Fluxes:
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

    def compute_derivatives(self) -> Derivatives:
        """Compute state derivatives implied by the flux balance."""
        return Derivatives(
            ds_sn=self.p_sn - self.f_sm,
            ds_s=self.p_r + self.f_sm - self.e_a - self.q_sc - self.q_sgwa,
            ds_gwa=self.q_sgwa - self.q_gwatd - self.q_gwac - self.q_gwap,
            ds_gwp=self.q_gwap - self.q_gwpc,
        )


@dataclass
class Parameters:
    """Model parameter set grouped by process family."""

    t_0: float = 0.0
    m_sn: float = 0.002
    k_sn: float = 1.157e-7

    c_e: float = 0.8
    s_max: float = 0.05
    m_s: float = 1e-5
    beta_s: float = 2.0
    k_sgw: float = 1e-7

    k_gwpc: float = 1e-7
    k_gwap: float = 1e-6
    k_gwac: float = 1e-6
    beta_gwac: float = 2.0
    s_gwa_max: float = 1.0
    s_ref_td: float = 0.5
    k_td: float = 1e-5

    gamma_x: float = -0.34
    gamma_i: float = 0.32
    gamma_p: float = 336.0

    # PET / radiation
    pet_albedo: float = 0.23
    pet_emissivity: float = 0.98

    n_gu: float = 2.0
    a_gu_seconds: float = 2 * SECONDS_PER_HOUR

    area_km2: float = 100.0

    def with_updates(self, updates: dict[str, float]) -> "Parameters":
        """Return a copy with selected parameter updates."""
        return replace(self, **updates)


@dataclass(kw_only=True)
class Forcings:
    """Atmospheric forcing variables supplied at each simulation step."""

    p_t: float = field(metadata={"unit": "m/s", "description": "total precipitation"})
    t: float = field(metadata={"unit": "°C", "description": "air temperature"})
    e_p: float = field(
        metadata={"unit": "m/s", "description": "potential evapotranspiration"}
    )
