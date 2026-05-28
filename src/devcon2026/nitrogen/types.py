"""Nitrogen model data structures."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Mapping

from devcon2026.types import ArrayConvertible


@dataclass
class NitrogenParameters:
    """Parameter set for a single soil control volume nitrogen model."""

    s_wp: float = field(
        default=20.0,
        metadata={"unit": "mm", "description": "wilting point soil water storage"},
    )
    s_max: float = field(
        default=147.3,
        metadata={"unit": "mm", "description": "maximum soil water storage"},
    )
    min_dissolved_storage: float = field(
        default=0.1,
        metadata={
            "unit": "mm",
            "description": "minimum storage for dissolved concentration and advective flux calculations",
        },
    )
    smf_sat: float = field(
        default=0.8,
        metadata={"unit": "1", "description": "saturated moisture factor"},
    )
    beta_sm: float = field(
        default=1.0,
        metadata={"unit": "1", "description": "moisture factor exponent"},
    )
    rel_saturation_low: float = field(
        default=0.2,
        metadata={"unit": "1", "description": "low relative saturation threshold"},
    )
    rel_saturation_high: float = field(
        default=0.9,
        metadata={"unit": "1", "description": "high relative saturation threshold"},
    )
    rel_sat_limit_exp: float = field(
        default=0.7,
        metadata={"unit": "1", "description": "exponential moisture limitation threshold"},
    )
    beta_exp: float = field(
        default=2.5,
        metadata={"unit": "1", "description": "exponential moisture factor exponent"},
    )
    v_degrad_son: float = field(
        default=1e-5,
        metadata={"unit": "1/day", "description": "maximum degradation rate of slow organic nitrogen"},
    )
    v_dissol_son: float = field(
        default=1e-5,
        metadata={"unit": "1/day", "description": "maximum dissolution rate of slow organic nitrogen"},
    )
    v_dissol_fon: float = field(
        default=1e-3,
        metadata={"unit": "1/day", "description": "maximum dissolution rate of fast organic nitrogen"},
    )
    v_min_fon: float = field(
        default=1e-3,
        metadata={"unit": "1/day", "description": "maximum mineralization rate of fast organic nitrogen"},
    )
    v_denit: float = field(
        default=5e-2,
        metadata={"unit": "1/day", "description": "maximum denitrification rate"},
    )
    k_denit: float = field(
        default=1.5,
        metadata={"unit": "mg/L", "description": "denitrification half-saturation concentration"},
    )
    uptake_demand: float = field(
        default=10.0,
        metadata={"unit": "kg N/km2/day", "description": "plant inorganic nitrogen uptake demand"},
    )
    delta_time_solver: float = field(
        default=1.0 / 24.0,
        metadata={"unit": "day", "description": "nitrogen solver time step"},
    )
    freundlich_exponent: float = field(
        default=1.0,
        metadata={"unit": "1", "description": "Freundlich DON adsorption exponent"},
    )
    freundlich_constant: float = field(
        default=100.0,
        metadata={"unit": "(mg N/kg soil)/(mg N/L)^n", "description": "Freundlich DON adsorption constant"},
    )
    soil_bulk_density: float = field(
        default=1.3,
        metadata={"unit": "kg/L", "description": "soil bulk density used for DON adsorption"},
    )
    deposition_din_fraction: float = field(
        default=1.0,
        metadata={"unit": "1", "description": "fraction of atmospheric deposition entering DIN"},
    )
    deposition_don_fraction: float = field(
        default=0.0,
        metadata={"unit": "1", "description": "fraction of atmospheric deposition entering DON"},
    )
    deposition_son_fraction: float = field(
        default=0.0,
        metadata={"unit": "1", "description": "fraction of atmospheric deposition entering SON"},
    )
    deposition_fon_fraction: float = field(
        default=0.0,
        metadata={"unit": "1", "description": "fraction of atmospheric deposition entering FON"},
    )
    fertilizer_din_fraction: float = field(
        default=1.0,
        metadata={"unit": "1", "description": "fraction of fertilizer application entering DIN"},
    )
    fertilizer_don_fraction: float = field(
        default=0.0,
        metadata={"unit": "1", "description": "fraction of fertilizer application entering DON"},
    )
    fertilizer_son_fraction: float = field(
        default=0.0,
        metadata={"unit": "1", "description": "fraction of fertilizer application entering SON"},
    )
    fertilizer_fon_fraction: float = field(
        default=0.0,
        metadata={"unit": "1", "description": "fraction of fertilizer application entering FON"},
    )
    manure_din_fraction: float = field(
        default=0.0,
        metadata={"unit": "1", "description": "fraction of manure application entering DIN"},
    )
    manure_don_fraction: float = field(
        default=0.0,
        metadata={"unit": "1", "description": "fraction of manure application entering DON"},
    )
    manure_son_fraction: float = field(
        default=0.0,
        metadata={"unit": "1", "description": "fraction of manure application entering SON"},
    )
    manure_fon_fraction: float = field(
        default=1.0,
        metadata={"unit": "1", "description": "fraction of manure application entering FON"},
    )

    def to_dict(self) -> dict[str, float]:
        """Return parameters as the dictionary expected by the legacy solver."""
        return asdict(self)

    def with_updates(self, updates: Mapping[str, float]) -> "NitrogenParameters":
        """Return a copy with selected parameter updates."""
        return replace(self, **dict(updates))


@dataclass(kw_only=True)
class NitrogenStates(ArrayConvertible):
    """Initial nitrogen masses in kg N/km2."""

    m_don: float = field(
        metadata={"unit": "kg N/km2", "description": "dissolved organic nitrogen mass"}
    )
    m_din: float = field(
        metadata={"unit": "kg N/km2", "description": "dissolved inorganic nitrogen mass"}
    )
    m_son: float = field(
        default=4.5e5,
        metadata={"unit": "kg N/km2", "description": "slow organic nitrogen mass"},
    )
    m_fon: float = field(
        default=1.0e4,
        metadata={"unit": "kg N/km2", "description": "fast organic nitrogen mass"},
    )
    m_don_ads: float = field(
        default=0.0,
        metadata={"unit": "kg N/km2", "description": "adsorbed dissolved organic nitrogen mass"},
    )

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
