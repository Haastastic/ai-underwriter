"""Map a calibrated model probability to an underwriting decision.

The model produces P(serious delinquency in 2 years). This module applies
the portfolio's two cutoffs to place an application in one of three bands.
It lives next to the model layer -- not in the app -- so the CLI and the API
share exactly one policy, and it is kept well away from ``src.llm``: the LLM
layer is handed the decision this produces and never the reverse.

Cutoffs (chosen from the v1 30k-row validation split; see models/v1):

    P < 0.08            approved   ~80% of applicants, ~2.1% observed default rate
    0.08 <= P < 0.30    referred   ~14% of applicants, the ambiguous middle band
    P >= 0.30           denied     ~6%  of applicants, ~47% observed default rate

The `referred` band is what the loan-officer review UI (Phase 7) works.
"""

from __future__ import annotations

from typing import Any

APPROVED = "approved"
REFERRED = "referred"
DENIED = "denied"

APPROVE_BELOW = 0.08
DENY_AT_OR_ABOVE = 0.30


def decide(
    probability: float,
    approve_below: float = APPROVE_BELOW,
    deny_at_or_above: float = DENY_AT_OR_ABOVE,
) -> dict[str, Any]:
    """Return the decision band for a probability.

    Output::

        {"decision": "approved" | "referred" | "denied",
         "probability": float,
         "thresholds": {"approve_below": float, "deny_at_or_above": float}}
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if not 0.0 < approve_below <= deny_at_or_above < 1.0:
        raise ValueError(
            "thresholds must satisfy 0 < approve_below <= deny_at_or_above < 1"
        )

    if probability < approve_below:
        decision = APPROVED
    elif probability >= deny_at_or_above:
        decision = DENIED
    else:
        decision = REFERRED

    return {
        "decision": decision,
        "probability": float(probability),
        "thresholds": {
            "approve_below": approve_below,
            "deny_at_or_above": deny_at_or_above,
        },
    }
