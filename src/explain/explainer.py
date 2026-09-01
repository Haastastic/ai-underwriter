"""Per-application SHAP explanations as plain structured data.

This layer sits between the model and the LLM adverse-action layer. It
answers "why did the model score this application the way it did?" and
returns the answer as a JSON-serialisable dict of per-feature
contributions -- never prose, never a SHAP object. The LLM layer consumes
this dict and nothing else from the model side, so the model can be
retrained or swapped without the LLM layer noticing.

SHAP values are in log-odds (margin) space -- the native output of
TreeExplainer on an XGBoost binary classifier -- so they are exactly
additive:

    base_value + sum(contribution.shap_value) == model log-odds margin

The positive class is ``SeriousDlqin2yrs = 1`` (serious delinquency), so a
positive SHAP value pushed the application toward *higher* risk. That sign
is what ``direction`` records and it does not depend on the output space.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import shap

INCREASES_RISK = "increases_risk"
DECREASES_RISK = "decreases_risk"


def build_explainer(model) -> shap.TreeExplainer:
    """Wrap a fitted tree model in a SHAP TreeExplainer.

    Split out so a caller scoring many applications builds the explainer
    once and passes it back into :func:`explain_row`.
    """
    return shap.TreeExplainer(model)


def explain_row(
    model,
    row: dict | pd.Series | pd.DataFrame,
    feature_names: list[str] | None = None,
    explainer: shap.TreeExplainer | None = None,
) -> dict[str, Any]:
    """Return a structured SHAP explanation for one application row.

    Parameters
    ----------
    model:
        Fitted XGBoost classifier (or compatible tree model).
    row:
        Feature name -> value, as a dict, a pandas Series, or a 1-row
        DataFrame. Must contain *exactly* the model's features -- a missing
        or unexpected feature is an error, not something to paper over.
    feature_names:
        Overrides the model's own feature-name list.
    explainer:
        Reuse a prebuilt explainer (see :func:`build_explainer`).

    Returns
    -------
    dict with keys:
        ``predicted_probability`` -- float, P(serious delinquency)
        ``base_value``            -- float, population log-odds (SHAP expected value)
        ``base_rate``             -- float, sigmoid(base_value) ~ population default rate
        ``logodds_margin``        -- float, base_value + sum(shap_value)
        ``contributions``         -- list[dict], sorted by |shap_value| desc, each:
            ``feature``    -- str
            ``value``      -- int | float, the applicant's value for this feature
            ``shap_value`` -- float, signed log-odds contribution
            ``direction``  -- "increases_risk" | "decreases_risk"
    """
    names = _model_feature_names(model, feature_names)
    values = _validate_row(row, names)

    frame = pd.DataFrame([[values[name] for name in names]], columns=names).astype(
        "float64"
    )

    explainer = explainer or build_explainer(model)
    result = explainer(frame)

    shap_values = [float(v) for v in result.values[0]]
    base_value = float(result.base_values[0])

    contributions = [
        {
            "feature": name,
            "value": _py_scalar(values[name]),
            "shap_value": sv,
            "direction": INCREASES_RISK if sv > 0 else DECREASES_RISK,
        }
        for name, sv in zip(names, shap_values)
    ]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

    return {
        "predicted_probability": float(model.predict_proba(frame)[0, 1]),
        "base_value": base_value,
        "base_rate": _sigmoid(base_value),
        "logodds_margin": base_value + sum(shap_values),
        "contributions": contributions,
    }


def top_contributors(
    explanation: dict[str, Any],
    k: int = 4,
    direction: str | None = INCREASES_RISK,
) -> list[dict]:
    """The ``k`` largest-magnitude contributions in ``direction``.

    ECOA adverse-action notices must state the specific principal reasons
    for a denial; this is the shortlist the LLM layer turns into that
    language. Pass ``direction=None`` to keep contributions of both signs.
    """
    items = explanation["contributions"]
    if direction is not None:
        items = [c for c in items if c["direction"] == direction]
    return items[:k]


def _model_feature_names(model, explicit: list[str] | None) -> list[str]:
    if explicit is not None:
        return list(explicit)
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    names = getattr(booster, "feature_names", None)
    if not names:
        raise ValueError(
            "Model exposes no feature names; pass feature_names= explicitly."
        )
    return list(names)


def _validate_row(
    row: dict | pd.Series | pd.DataFrame, feature_names: list[str]
) -> dict:
    """Normalise the row to a plain dict and check it against the contract."""
    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            raise ValueError(f"Expected exactly one row, got {len(row)}")
        provided = row.iloc[0].to_dict()
    elif isinstance(row, pd.Series):
        provided = row.to_dict()
    elif isinstance(row, dict):
        provided = dict(row)
    else:
        raise TypeError(f"Unsupported row type: {type(row)!r}")

    expected = set(feature_names)
    got = set(provided)
    if missing := expected - got:
        raise ValueError(f"Row is missing required features: {sorted(missing)}")
    if extra := got - expected:
        raise ValueError(f"Row has unexpected features: {sorted(extra)}")

    return provided


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _py_scalar(v):
    """Cast a numpy/pandas scalar to a plain Python number for JSON safety."""
    return v.item() if hasattr(v, "item") else v
