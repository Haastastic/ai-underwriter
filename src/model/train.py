"""Train the XGBoost underwriting model and persist a versioned artifact.

    python -m src.model.train
    python -m src.model.train --data data/raw/cs-training.csv

Runs the data-layer pipeline, makes a stratified train/validation split,
fits XGBoost with early stopping on the validation split, then writes
models/vN/ (model, feature list, eval report, calibration plot, metadata).
Existing version directories are never touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import xgboost as xgb

from src.model.artifacts import next_version_dir, save_artifacts
from src.model.config import (
    EARLY_STOPPING_ROUNDS,
    RANDOM_SEED,
    VAL_FRACTION,
    XGB_PARAMS,
)
from src.model.dataset import build_model_frame, split_xy, train_val_split

DEFAULT_DATA_PATH = Path("data/raw/cs-training.csv")


def train_model(X_train, y_train, X_val=None, y_val=None, params: dict | None = None):
    """Fit an XGBClassifier.

    If a validation set is supplied, early stopping is enabled and the
    returned model predicts with its best iteration.
    """
    params = dict(XGB_PARAMS if params is None else params)
    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def run_training(
    data_path: str | Path = DEFAULT_DATA_PATH,
    models_root: str | Path = "models",
    version_dir: str | Path | None = None,
) -> tuple[Path, dict]:
    """End-to-end: CSV -> trained model + artifacts. Returns (dir, eval_report)."""
    df = build_model_frame(data_path)
    X, y = split_xy(df)
    X_train, X_val, y_train, y_val = train_val_split(
        X, y, VAL_FRACTION, RANDOM_SEED
    )

    model = train_model(X_train, y_train, X_val, y_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    target_dir = (
        Path(version_dir)
        if version_dir is not None
        else next_version_dir(models_root)
    )
    metadata = {
        "model_library": "xgboost",
        "xgboost_version": xgb.__version__,
        "random_seed": RANDOM_SEED,
        "val_fraction": VAL_FRACTION,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_features": int(X.shape[1]),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_val": float(y_val.mean()),
        "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
        "params": {k: v for k, v in XGB_PARAMS.items()},
    }

    return save_artifacts(
        target_dir,
        model=model,
        feature_names=list(X.columns),
        y_val=y_val,
        y_val_prob=y_val_prob,
        metadata=metadata,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    args = parser.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(f"Training data not found: {args.data}")

    target_dir, report = run_training(args.data, models_root=args.models_root)
    print(f"Saved model artifacts to {target_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
