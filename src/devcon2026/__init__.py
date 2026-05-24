"""Reusable analysis tools for the devcon2026 project."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("devcon2026")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
