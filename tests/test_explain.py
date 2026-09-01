"""Explainability-layer tests.

The SHAP -> LLM boundary is the priority here: these lock the shape and the
types of :func:`explain_row`'s output so the (not-yet-built) LLM layer can
depend on it, and they check the feature-contract guards that keep a
malformed application row from silently producing a wrong explanation.

A small XGBoost model is trained on a synthetic frame in-fixture so the
suite needs neither the 150k-row CSV nor a persisted model artifact.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.explain.explainer import (
    DECREASES_RISK,
    INCREASES_RISK,
    build_explainer,
    explain_row,
    top_contributors,
)

FEATURES = ["util", "age", "past_due", "income", "noise_flag"]


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(
        {
            "util": rng.beta(2, 5, n),
            "age": rng.integers(21, 80, n),
            "past_due": rng.poisson(0.3, n),
            "income": rng.lognormal(8.5, 0.6, n),
            "noise_flag": rng.integers(0, 2, n),
        }
    )
    logit = -2.3 + 3.0 * X["util"] + 0.9 * X["past_due"] - 2e-5 * X["income"]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    clf = xgb.XGBClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.1, random_state=0
    )
    clf.fit(X, y)
    return clf


@pytest.fixture
def sample_row():
    return {
        "util": 0.85,
        "age": 27,
        "past_due": 2,
        "income": 3200.0,
        "noise_flag": 1,
    }


@pytest.fixture
def explanation(model, sample_row):
    return explain_row(model, sample_row)


# --- output contract (SHAP -> LLM boundary) --------------------------------


def test_top_level_keys_are_exactly_the_contract(explanation):
    assert set(explanation) == {
        "predicted_probability",
        "base_value",
        "base_rate",
        "logodds_margin",
        "contributions",
    }


def test_output_is_json_serialisable(explanation):
    round_tripped = json.loads(json.dumps(explanation))
    assert round_tripped["contributions"][0]["feature"] in FEATURES


def test_each_contribution_has_the_contract_fields_and_types(explanation):
    for c in explanation["contributions"]:
        assert set(c) == {"feature", "value", "shap_value", "direction"}
        assert isinstance(c["feature"], str)
        assert isinstance(c["value"], (int, float)) and not isinstance(c["value"], bool)
        assert isinstance(c["shap_value"], float)
        assert c["direction"] in (INCREASES_RISK, DECREASES_RISK)


def test_no_numpy_types_leak(explanation):
    for c in explanation["contributions"]:
        assert type(c["shap_value"]) is float
        assert type(c["value"]) in (int, float)
    for key in ("predicted_probability", "base_value", "base_rate", "logodds_margin"):
        assert type(explanation[key]) is float


def test_contributions_cover_every_feature_once(explanation):
    seen = [c["feature"] for c in explanation["contributions"]]
    assert sorted(seen) == sorted(FEATURES)


def test_contributions_sorted_by_absolute_shap_descending(explanation):
    mags = [abs(c["shap_value"]) for c in explanation["contributions"]]
    assert mags == sorted(mags, reverse=True)


def test_direction_matches_shap_sign(explanation):
    for c in explanation["contributions"]:
        expected = INCREASES_RISK if c["shap_value"] > 0 else DECREASES_RISK
        assert c["direction"] == expected


# --- correctness ---------------------------------------------------------


def test_shap_contributions_are_additive_to_the_model_margin(model, sample_row, explanation):
    frame = pd.DataFrame([sample_row])[FEATURES].astype("float64")
    margin = float(model.predict(frame, output_margin=True)[0])

    recon = explanation["base_value"] + sum(
        c["shap_value"] for c in explanation["contributions"]
    )
    assert recon == pytest.approx(margin, abs=1e-4)
    assert explanation["logodds_margin"] == pytest.approx(margin, abs=1e-4)


def test_predicted_probability_matches_the_model(model, sample_row, explanation):
    frame = pd.DataFrame([sample_row])[FEATURES].astype("float64")
    assert explanation["predicted_probability"] == pytest.approx(
        float(model.predict_proba(frame)[0, 1]), abs=1e-6
    )


def test_base_rate_is_sigmoid_of_base_value(explanation):
    bv = explanation["base_value"]
    assert explanation["base_rate"] == pytest.approx(1.0 / (1.0 + np.exp(-bv)))


def test_values_echo_the_applicant_input(explanation, sample_row):
    by_feature = {c["feature"]: c["value"] for c in explanation["contributions"]}
    for name, val in sample_row.items():
        assert by_feature[name] == pytest.approx(val)


# --- input forms are interchangeable ------------------------------------


def test_dict_series_and_dataframe_give_the_same_explanation(model, sample_row):
    from_dict = explain_row(model, sample_row)
    from_series = explain_row(model, pd.Series(sample_row))
    from_frame = explain_row(model, pd.DataFrame([sample_row]))

    for other in (from_series, from_frame):
        assert other["contributions"] == from_dict["contributions"]
        assert other["predicted_probability"] == pytest.approx(
            from_dict["predicted_probability"]
        )


# --- feature-contract guards ------------------------------------------


def test_missing_feature_raises(model, sample_row):
    del sample_row["income"]
    with pytest.raises(ValueError, match="missing required features"):
        explain_row(model, sample_row)


def test_unexpected_feature_raises(model, sample_row):
    sample_row["employer_zip"] = 90210
    with pytest.raises(ValueError, match="unexpected features"):
        explain_row(model, sample_row)


def test_multi_row_dataframe_raises(model, sample_row):
    two = pd.DataFrame([sample_row, sample_row])
    with pytest.raises(ValueError, match="exactly one row"):
        explain_row(model, two)


# --- top_contributors --------------------------------------------------


def test_top_contributors_filters_direction_and_truncates(explanation):
    top = top_contributors(explanation, k=2, direction=INCREASES_RISK)
    assert len(top) <= 2
    assert all(c["direction"] == INCREASES_RISK for c in top)
    mags = [abs(c["shap_value"]) for c in top]
    assert mags == sorted(mags, reverse=True)


def test_top_contributors_direction_none_keeps_both_signs(explanation):
    top = top_contributors(explanation, k=99, direction=None)
    assert top == explanation["contributions"]


# --- explainer reuse ------------------------------------------------


def test_prebuilt_explainer_can_be_reused_across_rows(model, sample_row):
    explainer = build_explainer(model)
    a = explain_row(model, sample_row, explainer=explainer)
    other = {**sample_row, "util": 0.1, "past_due": 0}
    b = explain_row(model, other, explainer=explainer)

    fresh_a = explain_row(model, sample_row)
    assert a["contributions"] == fresh_a["contributions"]
    assert a["contributions"] != b["contributions"]
