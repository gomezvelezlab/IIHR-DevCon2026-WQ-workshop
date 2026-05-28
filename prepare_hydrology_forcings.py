"""Clip hydrology forcings to the nitrogen forcing date range.

By default this reads:
  data/nitrogen_forcings.parquet
  data/hydrology_forcings_larger.csv

and writes:
  data/hydrology_forcings.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
DEFAULT_NITROGEN_FORCING = DATA_DIR / "nitrogen_forcings.parquet"
DEFAULT_HYDROLOGY_CSV = DATA_DIR / "hydrology_forcings_larger.csv"
DEFAULT_OUTPUT_PARQUET = DATA_DIR / "hydrology_forcings.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nitrogen-forcing",
        type=Path,
        default=DEFAULT_NITROGEN_FORCING,
        help="nitrogen forcing table with a date column; CSV and Parquet are supported",
    )
    parser.add_argument("--hydrology-csv", type=Path, default=DEFAULT_HYDROLOGY_CSV)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    return parser.parse_args()


def load_date_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    if path.suffix.lower() == ".parquet":
        nitrogen = pd.read_parquet(path, columns=["date"])
        nitrogen["date"] = pd.to_datetime(nitrogen["date"])
    else:
        nitrogen = pd.read_csv(path, usecols=["date"], parse_dates=["date"])
    if nitrogen.empty:
        raise ValueError(f"{path} does not contain any nitrogen forcing rows.")

    start = nitrogen["date"].min().normalize()
    end_exclusive = nitrogen["date"].max().normalize() + pd.Timedelta(days=1)
    return start, end_exclusive


def clip_hydrology(
    hydrology_csv: Path, start: pd.Timestamp, end_exclusive: pd.Timestamp
) -> pd.DataFrame:
    hydrology = pd.read_csv(hydrology_csv, parse_dates=["time"])
    if hydrology.empty:
        raise ValueError(f"{hydrology_csv} does not contain any hydrology rows.")

    in_range = (hydrology["time"] >= start) & (hydrology["time"] < end_exclusive)
    clipped = hydrology.loc[in_range].copy()
    if clipped.empty:
        end_inclusive = end_exclusive - pd.Timedelta(days=1)
        raise ValueError(
            f"{hydrology_csv} has no rows from {start.date()} through "
            f"{end_inclusive.date()}."
        )
    return clipped


def main() -> None:
    args = parse_args()
    start, end_exclusive = load_date_range(args.nitrogen_forcing)
    clipped = clip_hydrology(args.hydrology_csv, start, end_exclusive)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_parquet(args.output_parquet, engine="fastparquet", index=False)

    end_inclusive = end_exclusive - pd.Timedelta(days=1)
    print(
        f"Wrote {len(clipped)} rows for {start.date()} through "
        f"{end_inclusive.date()}."
    )
    print(f"Parquet: {args.output_parquet}")


if __name__ == "__main__":
    main()
