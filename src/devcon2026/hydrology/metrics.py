"""Objective metrics and time-series alignment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def nse(observed: NDArray[Any], simulated: NDArray[Any]) -> float:
    """Nash-Sutcliffe Efficiency (higher is better, max=1)."""
    denom = np.sum((observed - np.mean(observed)) ** 2)
    if np.isclose(denom, 0.0):
        return float("-inf")
    return float(1.0 - (np.sum((observed - simulated) ** 2) / denom))


def rmse(observed: NDArray[Any], simulated: NDArray[Any]) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((observed - simulated) ** 2)))


def align_series(
    simulated: pd.Series, observed: pd.Series
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Inner-align simulated and observed series and return numeric arrays."""
    aligned = pd.concat(
        [simulated.rename("sim"), observed.rename("obs")], axis=1
    ).dropna()
    if aligned.empty:
        raise ValueError(
            "No overlapping simulation/observation timestamps after alignment."
        )
    return aligned["obs"].to_numpy(dtype=float), aligned["sim"].to_numpy(dtype=float)
