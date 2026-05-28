"""Reusable water-quality analysis tools."""

from importlib.metadata import PackageNotFoundError, version

from devcon2026.hydrology import Hydrology
from devcon2026.hydrology import HydrologyDerivatives
from devcon2026.hydrology import HydrologyFluxes
from devcon2026.hydrology import HydrologyForcings
from devcon2026.hydrology import HydrologyParameters
from devcon2026.hydrology import HydrologySimulationResult
from devcon2026.hydrology import HydrologyStates
from devcon2026.nitrogen import Nitrogen
from devcon2026.nitrogen import NitrogenModel_SingleCV
from devcon2026.nitrogen import NitrogenParameters
from devcon2026.nitrogen import NitrogenSoilLayer
from devcon2026.nitrogen import NitrogenStates
from devcon2026.nitrogen import NitrogenThreeCompartment
from devcon2026.nitrogen import default_soil_parameters

try:
    __version__ = version("devcon2026")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "Hydrology",
    "HydrologyDerivatives",
    "HydrologyFluxes",
    "HydrologyForcings",
    "HydrologyParameters",
    "HydrologySimulationResult",
    "HydrologyStates",
    "Nitrogen",
    "NitrogenModel_SingleCV",
    "NitrogenParameters",
    "NitrogenSoilLayer",
    "NitrogenStates",
    "NitrogenThreeCompartment",
    "__version__",
    "default_soil_parameters",
]
