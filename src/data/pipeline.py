"""End-to-end data-layer entry point: raw CSV -> model-ready DataFrame."""

from pathlib import Path

import pandas as pd

from src.data.clean import clean_data
from src.data.load import load_raw_data
from src.features.engineer import engineer_features


def load_and_prepare(path: str | Path) -> pd.DataFrame:
    """Load, clean, and feature-engineer the raw dataset in one call."""
    return engineer_features(clean_data(load_raw_data(path)))
