"""Three-band decision policy tests."""

import pytest

from src.model.decision import (
    APPROVE_BELOW,
    APPROVED,
    DENIED,
    DENY_AT_OR_ABOVE,
    REFERRED,
    decide,
)


@pytest.mark.parametrize(
    "probability, expected",
    [
        (0.0, APPROVED),
        (APPROVE_BELOW - 1e-9, APPROVED),
        (APPROVE_BELOW, REFERRED),          # boundary is inclusive of referred
        (0.15, REFERRED),
        (DENY_AT_OR_ABOVE - 1e-9, REFERRED),
        (DENY_AT_OR_ABOVE, DENIED),         # boundary is inclusive of denied
        (1.0, DENIED),
    ],
)
def test_band_boundaries(probability, expected):
    assert decide(probability)["decision"] == expected


def test_output_shape():
    out = decide(0.5)
    assert set(out) == {"decision", "probability", "thresholds"}
    assert out["probability"] == 0.5
    assert out["thresholds"] == {
        "approve_below": APPROVE_BELOW,
        "deny_at_or_above": DENY_AT_OR_ABOVE,
    }


def test_decision_is_monotonic_in_probability():
    order = {APPROVED: 0, REFERRED: 1, DENIED: 2}
    seen = [order[decide(p / 100)["decision"]] for p in range(0, 101)]
    assert seen == sorted(seen)


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_probability_out_of_range_raises(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        decide(bad)


def test_custom_thresholds_validated():
    with pytest.raises(ValueError, match="approve_below"):
        decide(0.5, approve_below=0.4, deny_at_or_above=0.2)


def test_custom_thresholds_applied():
    assert decide(0.25, approve_below=0.1, deny_at_or_above=0.2)["decision"] == DENIED
    assert decide(0.05, approve_below=0.1, deny_at_or_above=0.2)["decision"] == APPROVED
