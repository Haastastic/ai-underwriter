"""Per-version model config tests: feature selection, v2 artifacts, cutoffs.

The point of `src.model.config` is that model v2 (no age features) is a
*config* difference from v1, not a fork of the pipeline. These tests lock
that:

* the v1 config still selects every pipeline column (and matches the
  committed ``models/v1/feature_names.json`` when the artifact is present);
* the v2 config excludes ``age`` and every age-derived feature -- and the
  list it excludes is the one the feature layer itself publishes;
* training with the v2 config writes a ``feature_names.json`` with no
  age-derived feature, and the serving path (``prepare_application``) and
  the fairness audit both work against that artifact with no other change;
* the recommended-cutoff derivation behaves as documented.

Everything runs on a synthetic raw CSV (the ``tests/test_backend.py``
pattern); nothing depends on the real dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean import fit_clean_stats
from src.data.load import load_raw_data
from src.data.schema import FEATURE_COLUMNS, RAW_COLUMNS
from src.explain import explain_row
from src.fairness.audit import run_audit
from src.fairness.report import main as fairness_report_main
from src.features.engineer import AGE_BIN_LABELS, AGE_DERIVED_FEATURES
from src.features.prepare import prepare_application
from src.model.artifacts import load_model
from src.model.config import (
    DEFAULT_CONFIG,
    MODEL_CONFIGS,
    XGB_PARAMS,
    ModelConfig,
    align_features,
    get_config,
)
from src.model.cutoffs import band_summary, recommend_cutoffs, resolve_cutoffs
from src.model.dataset import build_model_frame, split_xy
from src.model.report import build_report
from src.model.train import run_training

REPO_ROOT = Path(__file__).resolve().parent.parent
V1_FEATURE_NAMES = REPO_ROOT / "models" / "v1" / "feature_names.json"


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
        - 0.03 * (age - 50)  # an age signal v1 can use and v2 must not see
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
def synthetic_csv(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("cfg") / "cs-training.csv"
    _write_synthetic_raw_csv(path)
    return path


@pytest.fixture(scope="module")
def pipeline_columns(synthetic_csv) -> list[str]:
    X, _ = split_xy(build_model_frame(synthetic_csv))
    return list(X.columns)


@pytest.fixture(scope="module")
def trained_both(synthetic_csv, tmp_path_factory) -> dict:
    """v1 and v2 trained side by side from the same CSV into one models root."""
    models_root = tmp_path_factory.mktemp("models")
    run_training(synthetic_csv, models_root=models_root, version_dir=models_root / "v1", config="v1")
    run_training(synthetic_csv, models_root=models_root, version_dir=models_root / "v2", config="v2")
    return {"csv": synthetic_csv, "models_root": models_root}


# --- the age-derived list is owned by the feature layer -----------------


def test_age_derived_features_cover_raw_age_bins_and_ratio():
    assert "age" in AGE_DERIVED_FEATURES
    assert "credit_lines_per_year_of_age" in AGE_DERIVED_FEATURES
    for label in AGE_BIN_LABELS:
        assert f"age_bin_{label}" in AGE_DERIVED_FEATURES
    assert len(AGE_DERIVED_FEATURES) == 2 + len(AGE_BIN_LABELS)


def test_every_age_derived_name_is_a_real_pipeline_column(pipeline_columns):
    # If the feature layer renames a column, this list must move with it.
    assert set(AGE_DERIVED_FEATURES) <= set(pipeline_columns)


# --- config registry ----------------------------------------------------


def test_registry_has_v1_and_v2_and_v1_is_default():
    assert {"v1", "v2"} <= set(MODEL_CONFIGS)
    assert DEFAULT_CONFIG == "v1"
    assert get_config() is MODEL_CONFIGS["v1"]
    assert get_config("v2") is MODEL_CONFIGS["v2"]
    cfg = ModelConfig(name="x", description="ad hoc")
    assert get_config(cfg) is cfg


def test_get_config_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model config"):
        get_config("v99")


def test_v1_config_selects_every_pipeline_column(pipeline_columns):
    assert get_config("v1").select_feature_names(pipeline_columns) == pipeline_columns
    assert get_config("v1").excluded_features == ()
    assert get_config("v1").xgb_params == XGB_PARAMS


@pytest.mark.skipif(
    not V1_FEATURE_NAMES.exists(), reason="models/v1 artifact not present (gitignored)"
)
def test_v1_config_still_reproduces_committed_v1_feature_list(pipeline_columns):
    committed = json.loads(V1_FEATURE_NAMES.read_text())
    assert get_config("v1").select_feature_names(pipeline_columns) == committed


def test_v2_config_excludes_all_age_derived_names(pipeline_columns):
    v2 = get_config("v2")
    selected = v2.select_feature_names(pipeline_columns)

    assert set(v2.excluded_features) == set(AGE_DERIVED_FEATURES)
    assert set(selected).isdisjoint(AGE_DERIVED_FEATURES)
    assert not any(name.startswith("age") for name in selected)
    assert not any("age" in name.lower() for name in selected)
    # everything else survives, in pipeline order
    assert selected == [c for c in pipeline_columns if c not in AGE_DERIVED_FEATURES]
    assert len(selected) == len(pipeline_columns) - len(AGE_DERIVED_FEATURES)


def test_v2_params_keep_natural_class_ratio():
    # No reweighting trick in any version: calibration is a reported metric.
    for cfg in MODEL_CONFIGS.values():
        assert "scale_pos_weight" not in cfg.xgb_params
        assert cfg.xgb_params["objective"] == "binary:logistic"


def test_config_metadata_is_json_serialisable():
    for cfg in MODEL_CONFIGS.values():
        meta = json.loads(json.dumps(cfg.to_metadata()))
        assert meta["config"] == cfg.name
        assert meta["excluded_features"] == list(cfg.excluded_features)


def test_align_features_selects_subset_and_rejects_missing():
    X = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    pd.testing.assert_frame_equal(align_features(X, ["c", "a"]), X[["c", "a"]])
    with pytest.raises(ValueError, match="missing: \\['z'\\]"):
        align_features(X, ["a", "z"])


# --- training with the v2 config ---------------------------------------


def test_v2_artifacts_have_no_age_derived_feature(trained_both):
    v2_dir = trained_both["models_root"] / "v2"
    names = json.loads((v2_dir / "feature_names.json").read_text())
    assert set(names).isdisjoint(AGE_DERIVED_FEATURES)
    assert not any("age" in n.lower() for n in names)

    # the model itself agrees with the file
    model, loaded_names = load_model(v2_dir)
    assert loaded_names == names
    assert list(model.get_booster().feature_names) == names

    for artifact in ("model.json", "feature_names.json", "eval_report.json",
                     "calibration.png", "metadata.json"):
        assert (v2_dir / artifact).exists()


def test_v1_and_v2_share_split_and_differ_only_by_config(trained_both):
    root = trained_both["models_root"]
    m1 = json.loads((root / "v1" / "metadata.json").read_text())
    m2 = json.loads((root / "v2" / "metadata.json").read_text())

    assert m1["config"] == "v1" and m2["config"] == "v2"
    for key in ("random_seed", "val_fraction", "n_train", "n_val",
                "positive_rate_train", "positive_rate_val"):
        assert m1[key] == m2[key]
    assert m1["excluded_features"] == []
    assert set(m2["excluded_features"]) == set(AGE_DERIVED_FEATURES)
    assert m2["n_features"] == m1["n_features"] - len(AGE_DERIVED_FEATURES)
    assert m2["params"] == get_config("v2").xgb_params

    v1_names = json.loads((root / "v1" / "feature_names.json").read_text())
    assert len(v1_names) == m1["n_features"]
    assert set(AGE_DERIVED_FEATURES) <= set(v1_names)


def test_metadata_records_recommended_cutoffs(trained_both):
    meta = json.loads((trained_both["models_root"] / "v2" / "metadata.json").read_text())
    rec = meta["recommended_cutoffs"]
    assert 0.0 < rec["approve_below"] <= rec["deny_at_or_above"] < 1.0
    assert set(rec["bands"]) == {"approved", "referred", "denied"}
    assert set(rec["code_defaults"]) == {"approve_below", "deny_at_or_above", "bands"}


def test_report_scores_a_subset_feature_version(trained_both):
    # src.model.report used to require pipeline columns == feature_names;
    # v2 uses a subset, and must still be re-evaluable.
    metrics = build_report("v2", trained_both["csv"], models_root=trained_both["models_root"])
    saved = json.loads((trained_both["models_root"] / "v2" / "eval_report.json").read_text())
    assert metrics == pytest.approx(saved, rel=1e-6)


# --- serving path against the v2 artifact ------------------------------


APPLICATION = {
    "RevolvingUtilizationOfUnsecuredLines": 0.4,
    "age": 23,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "DebtRatio": 0.5,
    "MonthlyIncome": 3000,
    "NumberOfOpenCreditLinesAndLoans": 4,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}


def test_prepare_application_aligns_to_v2_feature_list(trained_both):
    model, names = load_model(trained_both["models_root"] / "v2")
    stats = fit_clean_stats(load_raw_data(trained_both["csv"]))

    row = prepare_application(APPLICATION, stats, names)
    assert list(row.columns) == names
    assert "age" not in row.columns
    assert set(FEATURE_COLUMNS) - {"age"} <= set(row.columns)

    prob = float(model.predict_proba(row)[0, 1])
    assert 0.0 <= prob <= 1.0

    # SHAP explanation covers exactly the v2 features, and no age feature
    explanation = explain_row(model, row.iloc[0].to_dict(), feature_names=names)
    feats = [c["feature"] for c in explanation["contributions"]]
    assert feats and set(feats) == set(names)
    assert set(feats).isdisjoint(AGE_DERIVED_FEATURES)


def test_prepare_application_ignores_age_for_v2(trained_both):
    # Two applicants differing only in age get the identical v2 score.
    model, names = load_model(trained_both["models_root"] / "v2")
    stats = fit_clean_stats(load_raw_data(trained_both["csv"]))
    young = prepare_application({**APPLICATION, "age": 22}, stats, names)
    old = prepare_application({**APPLICATION, "age": 71}, stats, names)
    pd.testing.assert_frame_equal(young, old)
    assert model.predict_proba(young)[0, 1] == model.predict_proba(old)[0, 1]


def test_prepare_application_raises_when_a_non_age_feature_is_missing(trained_both):
    _, names = load_model(trained_both["models_root"] / "v2")
    stats = fit_clean_stats(load_raw_data(trained_both["csv"]))
    with pytest.raises(ValueError, match="did not produce features the model expects"):
        prepare_application(APPLICATION, stats, names + ["brand_new_feature"])


def test_prepare_application_still_tolerates_missing_age_bins_for_v1(trained_both):
    _, names = load_model(trained_both["models_root"] / "v1")
    stats = fit_clean_stats(load_raw_data(trained_both["csv"]))
    row = prepare_application(APPLICATION, stats, names)
    assert list(row.columns) == names
    assert row["age_bin_18-24"].iloc[0] == 1
    assert row[[f"age_bin_{l}" for l in AGE_BIN_LABELS]].sum(axis=1).iloc[0] == 1


# --- fairness audit still groups on age for the age-blind model ---------


def test_fairness_audit_runs_on_v2_and_groups_by_age(trained_both):
    report = run_audit(
        "v2", trained_both["csv"], models_root=trained_both["models_root"], split="val"
    )
    json.dumps(report)
    assert report["model_version"] == "v2"
    assert report["protected_attribute"] == "age_band"
    groups = {g["group"] for g in report["groups"]}
    assert groups <= set(AGE_BIN_LABELS)
    assert len(groups) >= 4  # the synthetic ages 21-79 fill most bands
    assert sum(g["n"] for g in report["groups"]) == report["n"]


def test_fairness_cli_uses_recorded_cutoffs_for_v2(trained_both, tmp_path, capsys):
    out = tmp_path / "v2_fairness.json"
    fairness_report_main(
        [
            "v2",
            "--data", str(trained_both["csv"]),
            "--models-root", str(trained_both["models_root"]),
            "--out", str(out),
        ]
    )
    saved = json.loads(out.read_text())
    meta = json.loads((trained_both["models_root"] / "v2" / "metadata.json").read_text())
    rec = meta["recommended_cutoffs"]
    assert saved["thresholds"] == {
        "approve_below": rec["approve_below"],
        "deny_at_or_above": rec["deny_at_or_above"],
    }
    assert "recommended_cutoffs" in saved["cutoffs_source"]
    assert "recommended_cutoffs" in capsys.readouterr().out


def test_resolve_cutoffs_precedence(trained_both):
    v2_dir = trained_both["models_root"] / "v2"
    rec = json.loads((v2_dir / "metadata.json").read_text())["recommended_cutoffs"]

    a, d, src = resolve_cutoffs(v2_dir)
    assert (a, d) == (rec["approve_below"], rec["deny_at_or_above"])
    assert "recommended_cutoffs" in src

    a, d, src = resolve_cutoffs(v2_dir, 0.05, 0.25)
    assert (a, d, src) == (0.05, 0.25, "command line")

    # A directory with no metadata falls back to the code defaults.
    from src.model.decision import APPROVE_BELOW, DENY_AT_OR_ABOVE

    a, d, src = resolve_cutoffs(v2_dir.parent / "nope")
    assert (a, d) == (APPROVE_BELOW, DENY_AT_OR_ABOVE)
    assert "defaults" in src


# --- cutoff derivation --------------------------------------------------


def test_band_summary_partitions_and_reports_default_rates():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    p = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.60, 0.90])
    bands = band_summary(y, p, 0.08, 0.30)
    assert bands["approved"]["n"] == 3 and bands["approved"]["observed_default_rate"] == 0.0
    assert bands["referred"]["n"] == 2 and bands["referred"]["observed_default_rate"] == 0.5
    assert bands["denied"]["n"] == 3 and bands["denied"]["observed_default_rate"] == 1.0
    assert sum(b["share"] for b in bands.values()) == pytest.approx(1.0)


def test_band_summary_empty_band_has_no_default_rate():
    bands = band_summary([0, 1], [0.01, 0.02], 0.08, 0.30)
    assert bands["denied"]["n"] == 0
    assert bands["denied"]["observed_default_rate"] is None


def test_recommend_cutoffs_hits_target_band_shares():
    rng = np.random.default_rng(0)
    p = rng.beta(0.5, 6.0, 50_000)  # skewed low, like a default-probability model
    y = rng.binomial(1, p)
    rec = recommend_cutoffs(y, p, target_approve_share=0.80, target_deny_share=0.06)

    assert round(rec["approve_below"], 2) == rec["approve_below"]  # two decimals
    assert rec["approve_below"] <= rec["deny_at_or_above"]
    assert rec["bands"]["approved"]["share"] == pytest.approx(0.80, abs=0.03)
    assert rec["bands"]["denied"]["share"] == pytest.approx(0.06, abs=0.02)
    # a calibrated model's approve band is low-risk and its deny band high-risk
    assert rec["bands"]["approved"]["observed_default_rate"] < rec["bands"]["denied"]["observed_default_rate"]
    assert "code_defaults" in rec


def test_recommend_cutoffs_rejects_bad_targets():
    with pytest.raises(ValueError, match="target shares"):
        recommend_cutoffs([0, 1], [0.1, 0.9], target_approve_share=1.5)
