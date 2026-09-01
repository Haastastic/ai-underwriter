"""Load the raw "Give Me Some Credit" CSV into a validated DataFrame."""

from pathlib import Path

import pandas as pd

from src.data.schema import RAW_COLUMNS


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the Kaggle CSV and return only the expected raw columns.

    The Kaggle export's first column is an unnamed row-id (`Unnamed: 0`)
    left over from the original index -- it carries no information and is
    dropped here rather than downstream.
    """
    df = pd.read_csv(path)

    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])

    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    return df[RAW_COLUMNS].reset_index(drop=True)
