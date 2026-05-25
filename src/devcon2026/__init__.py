"""Reusable water-quality analysis tools."""

from importlib.metadata import PackageNotFoundError, version

from devcon2026.hydrology import Hydrology
from devcon2026.nitrogen import Nitrogen
from devcon2026.nitrogen import NitrogenModel_SingleCV
from devcon2026.nitrogen import default_soil_parameters

try:
    __version__ = version("devcon2026")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "Hydrology",
    "Nitrogen",
    "NitrogenModel_SingleCV",
    "__version__",
    "default_soil_parameters",
]
