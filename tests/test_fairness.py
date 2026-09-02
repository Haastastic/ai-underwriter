"""Fairness-audit layer tests.

Two priorities, matching the layer's job:

1. The four-fifths / adverse-impact-ratio arithmetic, checked against a
   hand-worked example (``_HAND_TABLE`` below).
2. The output contract of :func:`audit_decisions` -- plain, JSON-serialisable
   dicts, no numpy or DataFrame leakage -- so a report artifact and any
   downstream reader can depend on its shape.

The pure-metric tests build a tiny table by hand. One integration test
trains a small real model on a synthetic raw CSV (the
``tests/test_backend.py`` pattern) and runs the whole audit through it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.schema import RAW_COLUMNS
from src.features.engineer import engineer_features
from src.fairness.audit import audit_decisions, build_audit_frame, run_audit, score_frame
from src.fairness.groups import assign_age_band, assign_decision_band
from src.fairness.metrics import (
    DENIAL_RATIO_FLAG,
    FOUR_FIFTHS,
    denial_rate_disparity,
    disparate_impact,
    group_rates,
)
from src.fairness.report import default_out_path, main as report_main
from src.model.train import run_training


# --------------------------------------------------------------------------
# Hand-worked table
#
#   group  n  approved referred denied  approval_rate  acceptance  denial_rate
#   A      10    8        1        1        0.80          0.90        0.10
#   B      10    4        3        3        0.40          0.70        0.30
#   C      20   10        5        5        0.50          0.75        0.25
#
# approval reference = A (0.80). AIR: A 1.000, B 0.500, C 0.625  -> B, C fail 0.80
# denial reference   = A (0.10). ratio: A 1.000, B 3.000, C 2.500 -> B, C flagged
# --------------------------------------------------------------------------
def _hand_table(group_col: str = "group") -> pd.DataFrame:
    rows = (
        [("A", "approved")] * 8 + [("A", "referred")] * 1 + [("A", "denied")] * 1
        + [("B", "approved")] * 4 + [("B", "referred")] * 3 + [("B", "denied")] * 3
        + [("C", "approved")] * 10 + [("C", "referred")] * 5 + [("C", "denied")] * 5
    )
    return pd.DataFrame(rows, columns=[group_col, "decision"])


@pytest.fixture
def hand_rates() -> pd.DataFrame:
    return group_rates(_hand_table())


# --- group_rates ----------------------------------------------------------


def test_group_rates_counts_and_rates(hand_rates):
    assert list(hand_rates.index) == ["A", "B", "C"]
    assert hand_rates.loc["A", "n"] == 10
    assert hand_rates.loc["C", "n"] == 20
    assert hand_rates.loc["A", "approval_rate"] == pytest.approx(0.80)
    assert hand_rates.loc["B", "approval_rate"] == pytest.approx(0.40)
    assert hand_rates.loc["C", "denial_rate"] == pytest.approx(0.25)
    assert hand_rates.loc["A", "acceptance_rate"] == pytest.approx(0.90)


def test_group_rates_bands_sum_to_one(hand_rates):
    total = (
        hand_rates["approved_rate"]
        + hand_rates["referred_rate"]
        + hand_rates["denied_rate"]
    )
    assert np.allclose(total.to_numpy(), 1.0)


def test_group_rates_drops_null_group_rows():
    frame = pd.DataFrame(
        {"group": ["A", "A", None, np.nan], "decision": ["approved"] * 4}
    )
    rates = group_rates(frame)
    assert list(rates.index) == ["A"]
    assert rates.loc["A", "n"] == 2


def test_group_rates_rejects_unknown_decision_label():
    frame = pd.DataFrame({"group": ["A", "B"], "decision": ["approved", "maybe"]})
    with pytest.raises(ValueError, match="unexpected decision labels"):
        group_rates(frame)


# --- disparate_impact: the four-fifths calculation ----------------------


def test_disparate_impact_matches_hand_calculation(hand_rates):
    di = disparate_impact(hand_rates, favorable="approval_rate")

    assert di["reference_group"] == "A"
    assert di["reference_rate"] == pytest.approx(0.80)
    assert di["threshold"] == FOUR_FIFTHS

    ratios = di["ratios"]
    assert ratios["A"]["adverse_impact_ratio"] == pytest.approx(1.000)
    assert ratios["B"]["adverse_impact_ratio"] == pytest.approx(0.500)
    assert ratios["C"]["adverse_impact_ratio"] == pytest.approx(0.625)

    assert ratios["A"]["passes_four_fifths"] is True
    assert ratios["B"]["passes_four_fifths"] is False
    assert ratios["C"]["passes_four_fifths"] is False

    assert di["min_ratio"] == pytest.approx(0.500)
    assert di["min_ratio_group"] == "B"
    assert di["passes_four_fifths"] is False


def test_disparate_impact_acceptance_view_uses_not_denied(hand_rates):
    di = disparate_impact(hand_rates, favorable="acceptance_rate")
    # acceptance reference = A (0.90); B 0.70/0.90 = 0.778 (< 0.80, fails),
    # C 0.75/0.90 = 0.833 (>= 0.80, passes)
    assert di["reference_group"] == "A"
    assert di["ratios"]["B"]["adverse_impact_ratio"] == pytest.approx(0.7 / 0.9)
    assert di["ratios"]["B"]["passes_four_fifths"] is False
    assert di["ratios"]["C"]["passes_four_fifths"] is True


def test_disparate_impact_all_pass_when_groups_are_even():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "decision": (["approved"] * 8 + ["denied"] * 2) * 2,
        }
    )
    di = disparate_impact(group_rates(frame))
    assert di["passes_four_fifths"] is True
    assert di["min_ratio"] == pytest.approx(1.0)


def test_disparate_impact_degenerate_zero_reference_rate():
    frame = pd.DataFrame(
        {"group": ["A", "A", "B", "B"], "decision": ["denied"] * 4}
    )
    di = disparate_impact(group_rates(frame), favorable="approval_rate")
    assert di["reference_rate"] == 0.0
    assert di["ratios"]["A"]["adverse_impact_ratio"] is None
    assert di["passes_four_fifths"] is False


# --- denial_rate_disparity --------------------------------------------


def test_denial_rate_disparity_matches_hand_calculation(hand_rates):
    dd = denial_rate_disparity(hand_rates)
    assert dd["reference_group"] == "A"
    assert dd["reference_rate"] == pytest.approx(0.10)
    assert dd["threshold"] == DENIAL_RATIO_FLAG

    assert dd["ratios"]["A"]["denial_rate_ratio"] == pytest.approx(1.0)
    assert dd["ratios"]["B"]["denial_rate_ratio"] == pytest.approx(3.0)
    assert dd["ratios"]["C"]["denial_rate_ratio"] == pytest.approx(2.5)

    assert dd["ratios"]["A"]["flagged"] is False
    assert dd["ratios"]["B"]["flagged"] is True
    assert dd["ratios"]["C"]["flagged"] is True

    assert dd["max_ratio"] == pytest.approx(3.0)
    assert dd["max_ratio_group"] == "B"
    assert dd["passes"] is False


# --- assign_age_band -------------------------------------------------


@pytest.mark.parametrize(
    "age, band",
    [
        (18, "18-24"),
        (24, "18-24"),
        (25, "18-24"),   # right-closed bin edge, matches src.features.engineer
        (26, "25-34"),
        (35, "25-34"),
        (44, "35-44"),
        (54, "45-54"),
        (64, "55-64"),
        (65, "55-64"),
        (66, "65+"),
        (95, "65+"),
    ],
)
def test_assign_age_band_scalar_matches_feature_layer_bins(age, band):
    assert assign_age_band(age) == band


def test_assign_age_band_out_of_range_is_none():
    assert assign_age_band(0) is None
    assert assign_age_band(200) is None


def test_assign_age_band_vector_matches_engineer_dummies():
    ages = pd.DataFrame({"age": [22, 30, 40, 50, 60, 70]})
    banded = assign_age_band(ages["age"].to_numpy())

    # cross-check against the one-hot columns the feature layer produces
    eng = engineer_features(
        pd.DataFrame(
            {
                "age": ages["age"],
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfTimes90DaysLate": 0,
                "MonthlyIncome": 5000.0,
                "NumberOfDependents": 0.0,
                "NumberOfOpenCreditLinesAndLoans": 5,
            }
        )
    )
    for i, label in enumerate(banded):
        assert eng.iloc[i][f"age_bin_{label}"] == 1


# --- assign_decision_band -------------------------------------------


def test_assign_decision_band_scalar():
    assert assign_decision_band(0.05, 0.08, 0.30) == "approved"
    assert assign_decision_band(0.08, 0.08, 0.30) == "referred"
    assert assign_decision_band(0.20, 0.08, 0.30) == "referred"
    assert assign_decision_band(0.30, 0.08, 0.30) == "denied"


def test_assign_decision_band_vector_matches_scalar():
    probs = np.array([0.01, 0.08, 0.15, 0.30, 0.9])
    vec = assign_decision_band(probs, 0.08, 0.30).tolist()
    scal = [assign_decision_band(float(p), 0.08, 0.30) for p in probs]
    assert vec == scal == ["approved", "referred", "referred", "denied", "denied"]


def test_assign_decision_band_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        assign_decision_band(np.array([0.1, 0.2]), 0.5, 0.3)


# --- audit_decisions output contract -------------------------------


@pytest.fixture
def hand_report() -> dict:
    frame = _hand_table("age_band")
    return audit_decisions(frame, approve_below=0.08, deny_at_or_above=0.30)


def test_audit_decisions_top_level_contract(hand_report):
    assert set(hand_report) >= {
        "protected_attribute",
        "decision_bands",
        "thresholds",
        "n",
        "n_excluded_missing_group",
        "groups",
        "disparate_impact",
        "denial_rate_disparity",
        "four_fifths_rule",
        "limitations",
    }
    assert hand_report["protected_attribute"] == "age_band"
    assert hand_report["n"] == 40
    assert hand_report["decision_bands"] == ["approved", "referred", "denied"]
    assert "race" in hand_report["limitations"].lower()


def test_audit_decisions_four_fifths_rollup(hand_report):
    ff = hand_report["four_fifths_rule"]
    assert ff["passes"] is False
    assert ff["approval_air_min"] == pytest.approx(0.5)
    assert ff["approval_air_min_group"] == "B"
    assert ff["denial_ratio_max"] == pytest.approx(3.0)
    assert ff["denial_ratio_max_group"] == "B"


def test_audit_decisions_is_json_serialisable_and_plain(hand_report):
    dumped = json.dumps(hand_report)  # raises if numpy / DataFrame leaked
    round_tripped = json.loads(dumped)
    assert round_tripped["groups"][0]["group"] == "A"
    for g in hand_report["groups"]:
        assert type(g["n"]) is int
        assert type(g["approval_rate"]) is float


def test_audit_decisions_counts_excluded_rows():
    frame = _hand_table("age_band")
    frame.loc[len(frame)] = [None, "approved"]
    report = audit_decisions(frame)
    assert report["n"] == 41
    assert report["n_excluded_missing_group"] == 1


# --- integration: real model, synthetic dataset ----------------------


def _write_synthetic_raw_csv(path, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    util = rng.beta(2, 5, n)
    age = rng.integers(21, 80, n)
    past_due_30 = rng.poisson(0.4, n)
    debt_ratio = rng.gamma(2.0, 0.2, n)
    income = rng.lognormal(8.6, 0.6, n)
    open_lines = rng.poisson(8, n)
    late_90 = rng.poisson(0.15, n)
    real_estate = rng.poisson(1.0, n)
    past_due_60 = rng.poisson(0.1, n)
    dependents = rng.poisson(0.8, n)

    logit = (
        -4.6
        + 3.2 * util
        + 0.9 * past_due_30
        + 1.1 * late_90
        + 0.8 * debt_ratio
        - 1.5e-5 * income
        - 0.03 * (age - 50)  # a real age signal, so the audit has something to find
    )
    target = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    df = pd.DataFrame(
        {
            "SeriousDlqin2yrs": target,
            "RevolvingUtilizationOfUnsecuredLines": util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": past_due_30,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": past_due_60,
            "NumberOfDependents": dependents.astype(float),
        }
    )
    assert list(df.columns) == RAW_COLUMNS
    df.insert(0, "Unnamed: 0", range(1, n + 1))
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("fairness")
    csv_path = root / "cs-training.csv"
    _write_synthetic_raw_csv(csv_path)
    models_root = root / "models"
    run_training(csv_path, models_root=models_root, version_dir=models_root / "v1")
    return {"csv": csv_path, "models_root": models_root}


def test_build_audit_frame_has_expected_columns(trained):
    frame = build_audit_frame(
        "v1", trained["csv"], models_root=trained["models_root"], split="all"
    )
    assert set(frame.columns) == {"age", "age_band", "probability", "decision", "defaulted"}
    assert frame["decision"].isin(["approved", "referred", "denied"]).all()
    assert frame["probability"].between(0.0, 1.0).all()
    assert len(frame) == 6000


def test_score_frame_is_deterministic_and_read_only(trained):
    from src.model.artifacts import load_model
    from src.model.dataset import build_model_frame, split_xy

    model, feature_names = load_model(trained["models_root"] / "v1")
    X, y = split_xy(build_model_frame(trained["csv"]))

    probe = model.predict_proba(X.iloc[[0]])[0, 1]
    a = score_frame(model, feature_names, X, y_true=y)
    b = score_frame(model, feature_names, X, y_true=y)
    after = model.predict_proba(X.iloc[[0]])[0, 1]

    pd.testing.assert_frame_equal(a, b)
    assert probe == after  # auditing did not perturb the model


def test_run_audit_end_to_end_structure(trained):
    report = run_audit(
        "v1", trained["csv"], models_root=trained["models_root"], split="val"
    )
    json.dumps(report)  # serialisable
    assert report["model_version"] == "v1"
    assert report["split"] == "val"
    assert 0.0 <= report["observed_default_rate"] <= 1.0
    assert {g["group"] for g in report["groups"]}.issubset(
        {"18-24", "25-34", "35-44", "45-54", "55-64", "65+"}
    )
    # every group rate is a probability
    for g in report["groups"]:
        assert 0.0 <= g["approval_rate"] <= 1.0
        assert 0.0 <= g["denial_rate"] <= 1.0
    assert isinstance(report["four_fifths_rule"]["passes"], bool)


# --- CLI ------------------------------------------------------------


def test_report_cli_writes_artifact_and_refuses_overwrite(trained, capsys):
    out = default_out_path("v1", trained["models_root"])
    argv = [
        "v1",
        "--data", str(trained["csv"]),
        "--models-root", str(trained["models_root"]),
        "--split", "all",
    ]
    report_main(argv)
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["protected_attribute"] == "age_band"

    with pytest.raises(SystemExit, match="already exists"):
        report_main(argv)

    report_main(argv + ["--force"])  # --force is allowed to replace it


def test_report_cli_rejects_immutable_target(trained):
    with pytest.raises(SystemExit, match="immutable"):
        report_main(
            [
                "v1",
                "--data", str(trained["csv"]),
                "--models-root", str(trained["models_root"]),
                "--out", str(trained["models_root"] / "v1" / "eval_report.json"),
            ]
        )
