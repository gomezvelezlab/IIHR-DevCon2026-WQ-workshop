"""Clip hydrology forcings to the nitrogen forcing date range.

By default this reads:
  data/nitrogen_forcings.csv
  data/hydrology_forcings_larger.csv

and writes:
  data/hydrology_forcings.csv
  data/hydrology_forcings.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
DEFAULT_NITROGEN_CSV = DATA_DIR / "nitrogen_forcings.csv"
DEFAULT_HYDROLOGY_CSV = DATA_DIR / "hydrology_forcings_larger.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "hydrology_forcings.csv"
DEFAULT_OUTPUT_PARQUET = DATA_DIR / "hydrology_forcings.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nitrogen-csv", type=Path, default=DEFAULT_NITROGEN_CSV)
    parser.add_argument("--hydrology-csv", type=Path, default=DEFAULT_HYDROLOGY_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    return parser.parse_args()


def load_date_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
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
    start, end_exclusive = load_date_range(args.nitrogen_csv)
    clipped = clip_hydrology(args.hydrology_csv, start, end_exclusive)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_csv(args.output_csv, index=False)
    clipped.to_parquet(args.output_parquet, engine="fastparquet", index=False)

    end_inclusive = end_exclusive - pd.Timedelta(days=1)
    print(
        f"Wrote {len(clipped)} rows for {start.date()} through "
        f"{end_inclusive.date()}."
    )
    print(f"CSV: {args.output_csv}")
    print(f"Parquet: {args.output_parquet}")


if __name__ == "__main__":
    main()
