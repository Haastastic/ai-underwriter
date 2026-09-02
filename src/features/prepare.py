"""Turn one raw application into a model-ready feature row.

The batch path (`src.data.pipeline.load_and_prepare`) assumes a whole
dataset: it imputes from column medians and one-hot encodes whatever age
bins happen to be present. Neither holds for a single application, so
serving uses this module instead:

  - missing values are filled from `clean_stats` (see
    `src.data.clean.fit_clean_stats`), computed once on the training data;
  - the engineered frame is reindexed to the model's exact feature list, so
    the five age-bin columns a single row can't produce are filled with 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.data.clean import clean_data
from src.data.schema import FEATURE_COLUMNS
from src.features.engineer import engineer_features


def prepare_application(
    raw: dict[str, Any],
    clean_stats: dict[str, Any],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Raw application dict -> 1-row DataFrame with columns == `feature_names`.

    `raw` must contain the ten raw GMSC input fields (`MonthlyIncome` and
    `NumberOfDependents` may be None). Any extra key is an error.
    """
    missing = set(FEATURE_COLUMNS) - set(raw)
    if missing:
        raise ValueError(f"Application is missing fields: {sorted(missing)}")
    extra = set(raw) - set(FEATURE_COLUMNS)
    if extra:
        raise ValueError(f"Application has unexpected fields: {sorted(extra)}")

    row = pd.DataFrame([{col: raw[col] for col in FEATURE_COLUMNS}])
    row = clean_data(row, stats=clean_stats)
    row = engineer_features(row)

    aligned = row.reindex(columns=list(feature_names), fill_value=0)
    missing_features = set(feature_names) - set(row.columns)
    unexpected = set(row.columns) - set(feature_names)
    if unexpected:
        raise ValueError(
            f"Prepared row produced features the model does not expect: "
            f"{sorted(unexpected)}"
        )
    # `missing_features` are the absent age-bin dummies; reindex filled them
    # with 0, which is correct (the applicant is not in those buckets).
    _ = missing_features
    return aligned.astype("float64")
