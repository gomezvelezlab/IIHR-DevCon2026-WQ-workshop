"""Reusable water-quality analysis tools."""

from importlib.metadata import PackageNotFoundError, version

from devcon2026.nitrogen import Params

try:
    __version__ = version("devcon2026")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["Params", "__version__"]
