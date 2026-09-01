"""Model-layer tests.

These use a small synthetic frame with a known signal rather than the real
150k-row CSV, so the suite stays fast and runs without the dataset present.
The data-layer -> model-layer boundary (`build_model_frame`) is exercised
separately in the data/feature tests plus the end-to-end training run.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.schema import TARGET_COLUMN
from src.model.artifacts import load_model, next_version_dir, save_artifacts
from src.model.dataset import split_xy, train_val_split
from src.model.evaluate import compute_metrics, ks_statistic, save_calibration_plot
from src.model.train import train_model


@pytest.fixture
def synthetic_frame():
    rng = np.random.default_rng(0)
    n = 3000
    age = rng.integers(21, 80, n)
    util = rng.beta(2, 5, n)
    past_due = rng.poisson(0.3, n)
    income = rng.lognormal(8.5, 0.6, n)

    logit = (
        -2.3
        + 3.0 * util
        + 0.9 * past_due
        - 2e-5 * income
        - 0.015 * (age - 50)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, p)

    return pd.DataFrame(
        {
            TARGET_COLUMN: y,
            "age": age,
            "RevolvingUtilizationOfUnsecuredLines": util,
            "NumberOfTime30-59DaysPastDueNotWorse": past_due,
            "MonthlyIncome": income,
            "noise_flag": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def split(synthetic_frame):
    X, y = split_xy(synthetic_frame)
    return train_val_split(X, y, 0.25, 42)


# --- dataset -----------------------------------------------------------------


def test_split_xy_separates_target(synthetic_frame):
    X, y = split_xy(synthetic_frame)
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y) == len(synthetic_frame)
    assert set(y.unique()) <= {0, 1}


def test_split_xy_requires_target():
    with pytest.raises(ValueError, match="target"):
        split_xy(pd.DataFrame({"x": [1, 2, 3]}))


def test_split_xy_rejects_non_numeric_features():
    df = pd.DataFrame({TARGET_COLUMN: [0, 1], "grade": ["A", "B"]})
    with pytest.raises(ValueError, match="[Nn]on-numeric"):
        split_xy(df)


def test_train_val_split_is_reproducible_and_stratified(synthetic_frame):
    X, y = split_xy(synthetic_frame)
    _, X_val_a, _, y_val_a = train_val_split(X, y, 0.25, 42)
    _, X_val_b, _, y_val_b = train_val_split(X, y, 0.25, 42)

    pd.testing.assert_frame_equal(X_val_a, X_val_b)
    assert len(X_val_a) == pytest.approx(0.25 * len(X), abs=1)
    assert abs(y_val_a.mean() - y.mean()) < 0.01  # stratified


# --- metrics ---------------------------------------------------------------


def test_ks_statistic_perfect_separation_is_one():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_ks_statistic_random_scores_near_zero():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.5, 5000)
    prob = rng.random(5000)
    assert 0.0 <= ks_statistic(y, prob) < 0.1


def test_compute_metrics_reports_all_three_with_sane_values(split):
    X_train, X_val, y_train, y_val = split
    model = train_model(X_train, y_train, X_val, y_val)
    prob = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, prob)
    assert set(metrics) == {"auc_roc", "ks_statistic", "brier_score"}
    assert 0.5 < metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["ks_statistic"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 0.25


# --- training ------------------------------------------------------------


def test_train_model_predict_proba_in_unit_interval(split):
    X_train, X_val, y_train, _ = split
    model = train_model(X_train, y_train)
    prob = model.predict_proba(X_val)[:, 1]
    assert prob.min() >= 0.0
    assert prob.max() <= 1.0


def test_train_model_with_val_set_uses_early_stopping(split):
    X_train, X_val, y_train, y_val = split
    model = train_model(X_train, y_train, X_val, y_val)
    # best_iteration is only set when early stopping ran
    assert getattr(model, "best_iteration", None) is not None


# --- evaluate plot ----------------------------------------------------------


def test_save_calibration_plot_writes_a_png(tmp_path, split):
    X_train, X_val, y_train, y_val = split
    model = train_model(X_train, y_train)
    prob = model.predict_proba(X_val)[:, 1]

    out = tmp_path / "calibration.png"
    save_calibration_plot(y_val, prob, out)
    assert out.exists() and out.stat().st_size > 0


# --- artifacts ------------------------------------------------------------


def test_next_version_dir_increments_past_highest(tmp_path):
    assert next_version_dir(tmp_path).name == "v1"
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    (tmp_path / "scratch").mkdir()  # ignored: not a vN dir
    assert next_version_dir(tmp_path).name == "v3"


def test_save_artifacts_roundtrips_and_refuses_overwrite(tmp_path, split):
    X_train, X_val, y_train, y_val = split
    model = train_model(X_train, y_train)
    prob = model.predict_proba(X_val)[:, 1]
    feature_names = list(X_train.columns)

    version_dir = next_version_dir(tmp_path)
    _, report = save_artifacts(
        version_dir,
        model=model,
        feature_names=feature_names,
        y_val=y_val,
        y_val_prob=prob,
        metadata={"model_library": "xgboost"},
    )

    for name in (
        "model.json",
        "feature_names.json",
        "eval_report.json",
        "calibration.png",
        "metadata.json",
    ):
        assert (version_dir / name).exists()
    assert set(report) == {"auc_roc", "ks_statistic", "brier_score"}

    loaded, loaded_features = load_model(version_dir)
    assert loaded_features == feature_names
    np.testing.assert_allclose(
        loaded.predict_proba(X_val)[:, 1], prob, rtol=1e-5, atol=1e-6
    )

    with pytest.raises(FileExistsError):
        save_artifacts(
            version_dir,
            model=model,
            feature_names=feature_names,
            y_val=y_val,
            y_val_prob=prob,
            metadata={},
        )
