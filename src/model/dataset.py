"""Bridge the data layer to the model layer.

In: a raw "Give Me Some Credit" CSV path.
Out: a numeric feature matrix `X` and target vector `y`, plus a stratified
train/validation split.

Only plain pandas objects cross this boundary -- no XGBoost types -- so the
model library can be swapped without touching the data layer or this module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.pipeline import load_and_prepare
from src.data.schema import TARGET_COLUMN


def build_model_frame(raw_path: str | Path) -> pd.DataFrame:
    """Raw CSV -> cleaned, feature-engineered frame (target column included)."""
    return load_and_prepare(raw_path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split an engineered frame into features `X` and target `y`.

    Every non-target column is used as a model feature. The data layer has
    already imputed missing values and one-hot encoded the only categorical
    (age bin), so all columns must be numeric here; a non-numeric column
    means something upstream changed and is treated as an error rather than
    silently dropped.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Frame is missing target column {TARGET_COLUMN!r}")

    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns present: {non_numeric}")

    return X, y


def train_val_split(
    X: pd.DataFrame, y: pd.Series, val_fraction: float, seed: int
):
    """Stratified train/validation split.

    Stratified on the target so the rare positive (default) class keeps the
    same prevalence in both splits -- otherwise a small validation set can
    swing the reported AUC/KS purely on class-balance noise.
    """
    return train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
