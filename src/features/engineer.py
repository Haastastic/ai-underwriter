"""Interpretable feature engineering for Give Me Some Credit.

Kept deliberately small: ratios and bins that a loan officer can reason
about, rather than broad automated feature combinatorics. This is what
keeps the later SHAP/adverse-action story readable.
"""

import pandas as pd

# Public so downstream layers (e.g. the Phase 8 fairness audit) group by the
# *same* age bands the model's one-hot features use, rather than redefining
# edges that could drift out of sync.
AGE_BIN_EDGES = [0, 25, 35, 45, 55, 65, 120]
AGE_BIN_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

# Backwards-compatible private aliases (kept so nothing importing the old
# names breaks).
_AGE_BIN_EDGES = AGE_BIN_EDGES
_AGE_BIN_LABELS = AGE_BIN_LABELS

# Every column `engineer_features` produces that is a function of `age`:
# the raw field, the one-hot age bands, and the per-year-of-age ratio. This is
# the list a model version excludes when it must not see age at all (see
# `src.model.config` -- v2 drops all of these); it is defined here, in the
# layer that creates the columns, so it cannot drift from the feature code.
AGE_DERIVED_FEATURES: tuple[str, ...] = (
    "age",
    "credit_lines_per_year_of_age",
    *(f"age_bin_{label}" for label in AGE_BIN_LABELS),
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio, flag, and binned/encoded features to a cleaned DataFrame.

    Expects `clean_data` output (no missing values in age/income/dependents/
    past-due columns).
    """
    df = df.copy()

    df["total_past_due_count"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["has_past_due"] = (df["total_past_due_count"] > 0).astype(int)

    df["income_per_dependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["has_dependents"] = (df["NumberOfDependents"] > 0).astype(int)

    df["credit_lines_per_year_of_age"] = (
        df["NumberOfOpenCreditLinesAndLoans"] / df["age"]
    )

    age_bin = pd.cut(df["age"], bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS)
    age_dummies = pd.get_dummies(age_bin, prefix="age_bin", dtype=int)
    df = pd.concat([df, age_dummies], axis=1)

    return df
