"""Group fairness metrics: selection rates, the four-fifths rule, denial ratios.

Pure arithmetic over a table of ``(group, decision)`` pairs. No model, no
dataset loading, no framework-specific objects across the boundary -- a
DataFrame in, plain dicts / DataFrames out -- so every function here is
unit-testable on a hand-built table and reusable for any protected
attribute, not just the age bands this demo happens to audit.

Nothing in this module can influence a credit decision: it is handed
decisions the model layer already made and only measures how they fall
across groups.

Definitions
-----------
selection / approval rate
    Fraction of a group placed in the favourable band. Two favourable
    definitions are reported: ``approval_rate`` (band == "approved") and
    ``acceptance_rate`` (band in {"approved", "referred"} -- i.e. "not
    denied"), since a referral is not itself an adverse action.
denial rate
    Fraction of a group placed in the "denied" band.
adverse-impact ratio (AIR)
    group selection rate / selection rate of the *most*-selected group.
    The EEOC "four-fifths rule" treats ``AIR < 0.80`` for any group as
    evidence of adverse impact worth investigating.
denial-rate ratio
    group denial rate / denial rate of the *least*-denied group.
    Flagged here at ``> 1.25`` (= 1 / 0.80), the four-fifths rule applied
    to the unfavourable outcome.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.model.decision import APPROVED, DENIED, REFERRED

DECISION_BANDS = (APPROVED, REFERRED, DENIED)

FOUR_FIFTHS = 0.80
DENIAL_RATIO_FLAG = 1.25  # 1 / FOUR_FIFTHS


def group_rates(
    frame: pd.DataFrame,
    group_col: str = "group",
    decision_col: str = "decision",
) -> pd.DataFrame:
    """Per-group counts and rates for the three decision bands.

    ``frame`` needs one row per decided application, with a group label and
    a decision band (``"approved"`` / ``"referred"`` / ``"denied"``). Rows
    with a null group label are dropped (e.g. an age that fell outside the
    bin edges). Returns a DataFrame indexed by group, ordered by the group
    label, with columns::

        n
        approved  referred  denied            (counts)
        approved_rate  referred_rate  denied_rate
        approval_rate                          (== approved_rate)
        acceptance_rate                        (approved_rate + referred_rate)
        denial_rate                            (== denied_rate)
    """
    for col in (group_col, decision_col):
        if col not in frame.columns:
            raise ValueError(f"frame is missing required column {col!r}")

    data = frame[[group_col, decision_col]].dropna(subset=[group_col]).copy()
    # Normalise the group label to plain strings so a categorical column
    # cannot smuggle in empty, all-zero rows for unobserved categories.
    data[group_col] = data[group_col].astype(str)
    bad = set(data[decision_col].unique()) - set(DECISION_BANDS)
    if bad:
        raise ValueError(
            f"unexpected decision labels {sorted(bad)}; expected a subset of "
            f"{list(DECISION_BANDS)}"
        )
    if len(data) == 0:
        raise ValueError("no rows left to audit after dropping null group labels")

    counts = (
        data.assign(_one=1)
        .groupby([group_col, decision_col], observed=True)["_one"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=list(DECISION_BANDS), fill_value=0)
    )
    counts.columns.name = None
    counts.index.name = "group"

    out = counts.sort_index()
    out = out[out[list(DECISION_BANDS)].sum(axis=1) > 0]
    out.insert(0, "n", out[list(DECISION_BANDS)].sum(axis=1))
    for band in DECISION_BANDS:
        out[f"{band}_rate"] = out[band] / out["n"]
    out["approval_rate"] = out["approved_rate"]
    out["acceptance_rate"] = out["approved_rate"] + out["referred_rate"]
    out["denial_rate"] = out["denied_rate"]
    return out


def disparate_impact(
    rates: pd.DataFrame,
    favorable: str = "approval_rate",
) -> dict[str, Any]:
    """Adverse-impact ratios and the four-fifths-rule verdict.

    ``rates`` is the output of :func:`group_rates`. ``favorable`` selects
    the rate column to compare -- ``"approval_rate"`` (strict) or
    ``"acceptance_rate"`` ("not denied"). The reference group is the one
    with the highest favourable rate; every group's ratio is its rate over
    that reference rate.

    Returns::

        {
          "favorable_outcome": favorable,
          "reference_group": str,
          "reference_rate": float,
          "ratios": {group: {"rate": float,
                             "adverse_impact_ratio": float | None,
                             "passes_four_fifths": bool}},
          "min_ratio": float | None,
          "min_ratio_group": str | None,
          "passes_four_fifths": bool,          # all groups >= 0.80
          "threshold": 0.80,
        }

    ``adverse_impact_ratio`` is ``None`` only in the degenerate case where
    the reference rate is 0 (no group was ever selected); ``passes`` is then
    ``False`` for any group and the overall verdict is ``False``.
    """
    if favorable not in rates.columns:
        raise ValueError(
            f"{favorable!r} is not a column of the rates table; expected one "
            "of 'approval_rate' / 'acceptance_rate'"
        )
    if len(rates) == 0:
        raise ValueError("rates table is empty; nothing to compare")

    series = rates[favorable]
    ref_group = str(series.idxmax())
    ref_rate = float(series.max())

    ratios: dict[str, Any] = {}
    for group, rate in series.items():
        rate = float(rate)
        air = None if ref_rate == 0.0 else rate / ref_rate
        ratios[str(group)] = {
            "rate": rate,
            "adverse_impact_ratio": air,
            "passes_four_fifths": air is not None and air >= FOUR_FIFTHS,
        }

    finite = {g: r["adverse_impact_ratio"] for g, r in ratios.items()
              if r["adverse_impact_ratio"] is not None}
    min_group = min(finite, key=finite.get) if finite else None
    min_ratio = finite[min_group] if min_group is not None else None

    return {
        "favorable_outcome": favorable,
        "reference_group": ref_group,
        "reference_rate": ref_rate,
        "ratios": ratios,
        "min_ratio": min_ratio,
        "min_ratio_group": min_group,
        "passes_four_fifths": all(r["passes_four_fifths"] for r in ratios.values()),
        "threshold": FOUR_FIFTHS,
    }


def denial_rate_disparity(rates: pd.DataFrame) -> dict[str, Any]:
    """Denial-rate ratios against the least-denied group (four-fifths, inverted).

    ``rates`` is the output of :func:`group_rates`. The reference group is
    the one with the *lowest* denial rate; every group's ratio is its denial
    rate over that reference. A ratio above :data:`DENIAL_RATIO_FLAG`
    (1.25 = 1 / 0.80) is flagged.

    Returns the same shape as :func:`disparate_impact` with
    ``denial_rate`` / ``denial_rate_ratio`` / ``flagged`` keys, plus
    ``max_ratio`` / ``max_ratio_group`` and an overall ``passes`` (no group
    flagged). If the reference denial rate is 0, a group with any denials
    has ratio ``None`` (undefined but disparate) and is flagged.
    """
    if len(rates) == 0:
        raise ValueError("rates table is empty; nothing to compare")

    series = rates["denial_rate"]
    ref_group = str(series.idxmin())
    ref_rate = float(series.min())

    ratios: dict[str, Any] = {}
    for group, rate in series.items():
        rate = float(rate)
        if ref_rate > 0.0:
            ratio = rate / ref_rate
            flagged = ratio > DENIAL_RATIO_FLAG
        elif rate == 0.0:
            ratio, flagged = 1.0, False
        else:
            ratio, flagged = None, True
        ratios[str(group)] = {
            "denial_rate": rate,
            "denial_rate_ratio": ratio,
            "flagged": flagged,
        }

    finite = {g: r["denial_rate_ratio"] for g, r in ratios.items()
              if r["denial_rate_ratio"] is not None}
    max_group = max(finite, key=finite.get) if finite else None
    return {
        "reference_group": ref_group,
        "reference_rate": ref_rate,
        "ratios": ratios,
        "max_ratio": finite[max_group] if max_group is not None else None,
        "max_ratio_group": max_group,
        "passes": not any(r["flagged"] for r in ratios.values()),
        "threshold": DENIAL_RATIO_FLAG,
    }


__all__ = [
    "group_rates",
    "disparate_impact",
    "denial_rate_disparity",
    "FOUR_FIFTHS",
    "DENIAL_RATIO_FLAG",
    "DECISION_BANDS",
]
