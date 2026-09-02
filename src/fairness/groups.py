"""Group labels for the fairness audit: age bands and decision bands.

Both helpers deliberately reuse the constants the rest of the pipeline
already defines -- the age-bin edges from ``src.features.engineer`` and the
cutoffs / band names from ``src.model.decision`` -- so the audit groups
applicants exactly the way the model's own features and decision policy do,
and cannot drift out of sync with them.

Protected attribute
-------------------
The Kaggle "Give Me Some Credit" dataset carries no race, sex, ethnicity,
national-origin, or marital-status fields. ``age`` is the only demographic
attribute in it, and age is an ECOA-protected basis (Regulation B,
12 CFR 1002.6(b)(2)). So this audit uses **age bands** as its protected
grouping. That is a real limitation, not a design choice: a production
fair-lending audit would repeat every metric here for race, sex, national
origin, marital status, receipt of public assistance, and the exercise of
Consumer Credit Protection Act rights -- typically using proxy methods such
as BISG when those attributes are not collected directly.
"""

from __future__ import annotations

import pandas as pd

from src.features.engineer import AGE_BIN_EDGES, AGE_BIN_LABELS
from src.model.decision import (
    APPROVE_BELOW,
    APPROVED,
    DENIED,
    DENY_AT_OR_ABOVE,
    REFERRED,
    decide,
)

AGE_BAND_LIMITATION = (
    "Age is the only demographic attribute in the Give Me Some Credit "
    "dataset. It is an ECOA-protected basis, so it is used here as the "
    "protected grouping, but a production audit would also cover race, sex, "
    "national origin, marital status, age (>=62), and receipt of public "
    "assistance -- typically via a proxy method (e.g. BISG) where those "
    "attributes are not directly collected."
)


def assign_age_band(age) -> "pd.Categorical | str":
    """Map age(s) to the feature layer's age-bin label(s).

    Accepts a scalar (returns a ``str``) or any 1-D sequence / Series
    (returns a ``pandas.Categorical`` with the six ordered band labels).
    Ages outside the bin edges (``<=0`` or ``>120``) become ``NaN`` /
    ``None`` -- the caller decides whether to drop them.
    """
    scalar = pd.api.types.is_scalar(age)
    series = pd.Series([age] if scalar else age)
    banded = pd.cut(series, bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS)
    if scalar:
        value = banded.iloc[0]
        return None if pd.isna(value) else str(value)
    return pd.Categorical(banded, categories=AGE_BIN_LABELS, ordered=True)


def assign_decision_band(
    probability,
    approve_below: float = APPROVE_BELOW,
    deny_at_or_above: float = DENY_AT_OR_ABOVE,
) -> "pd.Series | str":
    """Map P(default) to ``"approved"`` / ``"referred"`` / ``"denied"``.

    Scalar in -> ``str`` out; sequence in -> ``pandas.Series`` of str.
    Uses ``src.model.decision.decide`` for a scalar so there is exactly one
    banding rule in the codebase; the vectorised path applies the identical
    cutoffs to a whole column at once.
    """
    if pd.api.types.is_scalar(probability):
        return decide(probability, approve_below, deny_at_or_above)["decision"]

    if not 0.0 < approve_below <= deny_at_or_above < 1.0:
        raise ValueError(
            "thresholds must satisfy 0 < approve_below <= deny_at_or_above < 1"
        )
    prob = pd.Series(probability, dtype="float64").reset_index(drop=True)
    if not prob.between(0.0, 1.0).all():
        raise ValueError("probabilities must all be in [0, 1]")

    band = pd.Series(REFERRED, index=prob.index, dtype="object")
    band[prob < approve_below] = APPROVED
    band[prob >= deny_at_or_above] = DENIED
    return band


__all__ = [
    "assign_age_band",
    "assign_decision_band",
    "AGE_BAND_LIMITATION",
    "APPROVED",
    "REFERRED",
    "DENIED",
]
