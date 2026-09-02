"""Derive recommended three-band cutoffs for a model version from its validation split.

`src.model.decision` fixes the *shape* of the policy -- two cutoffs, three
bands -- and the portfolio's risk appetite was set with the original v1
policy:

    approved  P < 0.08       ~80% of applicants, ~2% observed default rate
    referred                 the ambiguous middle band
    denied    P >= 0.30      ~6%  of applicants, ~47% observed default rate

A new version's probabilities can sit on a different scale, so the same
numbers are not automatically the same policy. This module re-derives the
cutoffs for a version by the reasoning above -- keep the approve band and
the deny band the same *size* (how many applicants are auto-approved and
auto-declined), then report the observed default rate each band actually
carries. The result is recorded in the version's ``metadata.json`` as
``recommended_cutoffs``. Training never rewrites the code defaults in
``src.model.decision``; those are promoted by hand when a version becomes
the default (v2's 0.08 / 0.28 are the current ones), and any other version
is served with its own numbers through ``AIU_APPROVE_BELOW`` /
``AIU_DENY_AT_OR_ABOVE``.

Nothing here decides an application: it is offline arithmetic over the
validation split's labels and probabilities, run once at training time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.model.decision import APPROVE_BELOW, APPROVED, DENIED, DENY_AT_OR_ABOVE, REFERRED

# The original v1 policy's band shares are the recommendation targets, so
# every version is sized to the same risk appetite.
TARGET_APPROVE_SHARE = 0.80
TARGET_DENY_SHARE = 0.06

# Cutoffs are quoted to two decimals, like the v1 policy, so a recommended
# value is something a person can read off and set in an environment
# variable rather than an eight-digit float.
_GRID_STEP = 0.01


def band_summary(
    y_true, y_prob, approve_below: float, deny_at_or_above: float
) -> dict[str, dict[str, float | int]]:
    """Share of applicants and observed default rate in each band.

    Returns ``{"approved" | "referred" | "denied": {"n", "share",
    "observed_default_rate"}}``. A band with no applicants reports a
    default rate of ``None``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n_total = len(y_prob)
    masks = {
        APPROVED: y_prob < approve_below,
        REFERRED: (y_prob >= approve_below) & (y_prob < deny_at_or_above),
        DENIED: y_prob >= deny_at_or_above,
    }
    out: dict[str, dict[str, Any]] = {}
    for band, mask in masks.items():
        n = int(mask.sum())
        out[band] = {
            "n": n,
            "share": float(n / n_total) if n_total else 0.0,
            "observed_default_rate": float(y_true[mask].mean()) if n else None,
        }
    return out


def _grid() -> np.ndarray:
    return np.round(np.arange(_GRID_STEP, 1.0, _GRID_STEP), 2)


def _closest_cutoff(y_prob: np.ndarray, share_of, target: float) -> float:
    """The grid cutoff whose band share (per `share_of`) is closest to `target`.

    Ties go to the lower cutoff, which is the conservative direction for
    both bands (approve slightly fewer, deny slightly more).
    """
    grid = _grid()
    shares = np.array([share_of(y_prob, c) for c in grid])
    return float(grid[int(np.argmin(np.abs(shares - target)))])


def _approve_share(y_prob: np.ndarray, cutoff: float) -> float:
    return float((y_prob < cutoff).mean())


def _deny_share(y_prob: np.ndarray, cutoff: float) -> float:
    return float((y_prob >= cutoff).mean())


def recommend_cutoffs(
    y_true,
    y_prob,
    *,
    target_approve_share: float = TARGET_APPROVE_SHARE,
    target_deny_share: float = TARGET_DENY_SHARE,
) -> dict[str, Any]:
    """Recommend (approve_below, deny_at_or_above) for a model's probabilities.

    ``approve_below`` is the two-decimal cutoff whose auto-approve share is
    closest to ``target_approve_share``; ``deny_at_or_above`` the one whose
    auto-deny share is closest to ``target_deny_share``. Both are reported
    with the band shares and observed default rates they actually produce,
    alongside the same summary under the code-default cutoffs, so a reviewer
    can see whether the numbers moved and what the move buys.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    if not 0.0 < target_approve_share < 1.0 or not 0.0 < target_deny_share < 1.0:
        raise ValueError("target shares must be in (0, 1)")

    approve_below = _closest_cutoff(y_prob, _approve_share, target_approve_share)
    deny_at_or_above = _closest_cutoff(y_prob, _deny_share, target_deny_share)

    if not approve_below <= deny_at_or_above:
        raise ValueError(
            "recommended cutoffs crossed; the probabilities are too compressed "
            "for the target band shares"
        )

    return {
        "approve_below": approve_below,
        "deny_at_or_above": deny_at_or_above,
        "method": (
            "two-decimal cutoffs matching the v1 policy's band shares "
            f"(approve >= {target_approve_share:.0%} of applicants, "
            f"deny >= {target_deny_share:.0%}) on the validation split"
        ),
        "bands": band_summary(y_true, y_prob, approve_below, deny_at_or_above),
        "code_defaults": {
            "approve_below": APPROVE_BELOW,
            "deny_at_or_above": DENY_AT_OR_ABOVE,
            "bands": band_summary(y_true, y_prob, APPROVE_BELOW, DENY_AT_OR_ABOVE),
        },
    }


def resolve_cutoffs(
    model_dir: str | Path,
    approve_below: float | None = None,
    deny_at_or_above: float | None = None,
) -> tuple[float, float, str]:
    """Pick the cutoffs to apply for a saved version and say where they came from.

    Explicit values win; otherwise the version's ``metadata.json``
    ``recommended_cutoffs`` (recorded by ``src.model.train`` for every
    version from v2 on); otherwise the code defaults in
    ``src.model.decision``. Returns ``(approve_below, deny_at_or_above,
    source)``. Used by the offline tools (fairness audit, CLI pipeline);
    the API takes its cutoffs from ``AIU_APPROVE_BELOW`` /
    ``AIU_DENY_AT_OR_ABOVE`` so that policy stays an explicit deployment
    setting.
    """
    if approve_below is not None and deny_at_or_above is not None:
        return approve_below, deny_at_or_above, "command line"

    metadata_path = Path(model_dir) / "metadata.json"
    recommended: dict | None = None
    if metadata_path.is_file():
        recommended = json.loads(metadata_path.read_text()).get("recommended_cutoffs")

    if recommended is not None:
        a = approve_below if approve_below is not None else float(recommended["approve_below"])
        d = deny_at_or_above if deny_at_or_above is not None else float(recommended["deny_at_or_above"])
        return a, d, f"{metadata_path} (recommended_cutoffs)"

    a = approve_below if approve_below is not None else APPROVE_BELOW
    d = deny_at_or_above if deny_at_or_above is not None else DENY_AT_OR_ABOVE
    return a, d, "src.model.decision defaults"


__all__ = [
    "band_summary",
    "recommend_cutoffs",
    "resolve_cutoffs",
    "TARGET_APPROVE_SHARE",
    "TARGET_DENY_SHARE",
]
