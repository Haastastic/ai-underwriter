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
"""

import pandas as pd

_PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
_PAST_DUE_SENTINELS = {96, 98}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw DataFrame.

    Adds `income_missing` / `dependents_missing` indicator columns before
    imputing those fields, so downstream layers can still see which values
    were originally absent rather than genuinely zero/median.
    """
    df = df.copy()

    df.loc[df["age"] == 0, "age"] = pd.NA
    df["age"] = df["age"].fillna(df["age"].median())

    for col in _PAST_DUE_COLUMNS:
        df.loc[df[col].isin(_PAST_DUE_SENTINELS), col] = pd.NA
        df[col] = df[col].fillna(df[col].median())

    df["income_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    df["dependents_missing"] = df["NumberOfDependents"].isna().astype(int)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    return df
