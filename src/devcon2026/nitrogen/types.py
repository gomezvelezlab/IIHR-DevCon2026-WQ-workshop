"""Nitrogen model data structures."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from typing import Mapping

from devcon2026.types import ArrayConvertible


@dataclass
class NitrogenParameters:
    """Parameter set for a single soil control volume nitrogen model."""

    s_wp: float = 20.0
    s_max: float = 147.3
    smf_sat: float = 0.8
    beta_sm: float = 1.0
    rel_saturation_low: float = 0.2
    rel_saturation_high: float = 0.9
    rel_sat_limit_exp: float = 0.7
    beta_exp: float = 2.5
    v_degrad_son: float = 1e-5
    v_dissol_son: float = 1e-5
    v_dissol_fon: float = 1e-3
    v_min_fon: float = 1e-3
    v_denit: float = 5e-2
    k_denit: float = 1.5
    uptake_demand: float = 10.0
    delta_time_solver: float = 1.0 / 24.0
    freundlich_exponent: float = 1.0
    freundlich_constant: float = 100.0
    soil_bulk_density: float = 1.3

    def to_dict(self) -> dict[str, float]:
        """Return parameters as the dictionary expected by the legacy solver."""
        return asdict(self)

    def with_updates(self, updates: Mapping[str, float]) -> "NitrogenParameters":
        """Return a copy with selected parameter updates."""
        return replace(self, **dict(updates))


@dataclass(kw_only=True)
class NitrogenStates(ArrayConvertible):
    """Initial nitrogen masses in kg N/km2."""

    m_don: float
    m_din: float
    m_son: float = 4.5e5
    m_fon: float = 1.0e4
    m_don_ads: float = 0.0

    @classmethod
    def from_mean_storage(cls, mean_storage: float) -> "NitrogenStates":
        """Infer default initial dissolved masses from mean soil storage."""
        return cls(
            m_don=5.0 * mean_storage,
            m_din=25.0 * mean_storage,
            m_son=4.5e5,
            m_fon=1.0e4,
            m_don_ads=0.0,
        )


def coerce_nitrogen_parameters(
    params: NitrogenParameters | Mapping[str, float] | None,
) -> NitrogenParameters:
    if params is None:
        return NitrogenParameters()
    if isinstance(params, NitrogenParameters):
        return params
    return NitrogenParameters().with_updates(params)


def default_soil_parameters() -> dict[str, float]:
    """Return baseline parameters for a single soil control volume."""
    return NitrogenParameters().to_dict()
