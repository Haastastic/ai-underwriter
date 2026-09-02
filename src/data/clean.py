"""Clean raw Give Me Some Credit data: fix invalid values, handle missingness.

Known data-quality issues in this dataset (documented here since they aren't
obvious from the column names alone):
  - `age` contains a small number of 0 values, which is not a valid
    applicant age.
  - The three "days past due" count columns contain sentinel error codes
    96 and 98 -- these are known data-entry artifacts in the Kaggle export,
    not real counts of 96/98 late payments, so they're treated as missing.
  - `MonthlyIncome` and `NumberOfDependents` have real missingness
    (~20% and ~3% of rows respectively in the full dataset).

Imputation values (the medians) can either be computed from the frame being
cleaned -- the default, used for batch training -- or passed in via `stats`.
Single-row inference at serving time must pass `stats` from
`fit_clean_stats` on the training data, since the median of one row is
meaningless.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
_PAST_DUE_SENTINELS = {96, 98}


def fit_clean_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute the imputation medians `clean_data` needs, from a training frame.

    Returns a plain dict (JSON-serialisable) so it can be persisted next to a
    model and reloaded for single-row inference.
    """
    df = df.copy()
    age = df["age"].mask(df["age"] == 0)

    past_due_medians = {}
    for col in _PAST_DUE_COLUMNS:
        series = df[col].mask(df[col].isin(_PAST_DUE_SENTINELS))
        past_due_medians[col] = float(series.median())

    return {
        "age_median": float(age.median()),
        "past_due_medians": past_due_medians,
        "income_median": float(df["MonthlyIncome"].median()),
    }


def clean_data(
    df: pd.DataFrame, stats: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Return a cleaned copy of the raw DataFrame.

    Adds `income_missing` / `dependents_missing` indicator columns before
    imputing those fields, so downstream layers can still see which values
    were originally absent rather than genuinely zero/median.

    If `stats` is None the imputation medians are taken from `df` itself
    (batch use). Pass `stats` from `fit_clean_stats` for single-row use.
    """
    df = df.copy()

    if stats is None:
        age_median = df["age"].mask(df["age"] == 0).median()
        past_due_medians = None
        income_median = df["MonthlyIncome"].median()
    else:
        age_median = stats["age_median"]
        past_due_medians = stats["past_due_medians"]
        income_median = stats["income_median"]

    df.loc[df["age"] == 0, "age"] = pd.NA
    df["age"] = df["age"].fillna(age_median)

    for col in _PAST_DUE_COLUMNS:
        df.loc[df[col].isin(_PAST_DUE_SENTINELS), col] = pd.NA
        median = (
            df[col].median() if past_due_medians is None else past_due_medians[col]
        )
        df[col] = df[col].fillna(median)

    df["income_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(income_median)

    df["dependents_missing"] = df["NumberOfDependents"].isna().astype(int)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    return df
