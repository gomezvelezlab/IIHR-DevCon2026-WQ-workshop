"""Period orchestration helpers for spinup/evaluation workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .io import load_forcing_data, load_observed_discharge
from .simulation import simulate
from .types import Parameters, SimulationResult, States


@dataclass(frozen=True)
class PeriodWindow:
    """Defines a spinup/evaluation window for one model run.

    ``eval_end`` is exclusive.
    """

    name: str
    spinup_start: str
    eval_start: str
    eval_end: str


@dataclass
class PeriodRun:
    """Outputs for a period run including full and evaluation-only series."""

    full_result: SimulationResult
    eval_simulated: pd.Series
    eval_observed: pd.Series


def run_period(
    forcing_csv: str,
    observed_flow_csv: str,
    params: Parameters,
    initial_states: States,
    window: PeriodWindow,
    progress: bool = False,
) -> PeriodRun:
    """Run a single spinup+evaluation period and return eval slices for scoring."""
    forcing_df = load_forcing_data(
        forcing_csv,
        start_time=window.spinup_start,
        end_time=window.eval_end,
        params=params,
    )
    full_result = simulate(
        forcing_df,
        params,
        initial_states,
        progress=progress,
        progress_desc=window.name,
    )

    eval_start_ts = pd.Timestamp(window.eval_start, tz="UTC")
    eval_end_ts = pd.Timestamp(window.eval_end, tz="UTC")
    eval_simulated = full_result.discharge_cms[
        (full_result.discharge_cms.index >= eval_start_ts)
        & (full_result.discharge_cms.index < eval_end_ts)
    ]

    eval_observed = load_observed_discharge(
        observed_flow_csv,
        start_time=window.eval_start,
        end_time=window.eval_end,
    )

    return PeriodRun(
        full_result=full_result,
        eval_simulated=eval_simulated,
        eval_observed=eval_observed,
    )
