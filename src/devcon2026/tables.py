"""Small table I/O helpers shared by demos and workflow facades."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Read a dataframe artifact from Parquet or CSV."""
    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path)
    return pd.read_csv(table_path, parse_dates=parse_dates)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a dataframe artifact as Parquet or CSV based on suffix."""
    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        df.to_parquet(table_path, index=False)
        # df.to_parquet(table_path, engine="fastparquet", index=False)
        return
    df.to_csv(table_path, index=False)
