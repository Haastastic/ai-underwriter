"""Small, principled hyperparameter search for a model-version config.

    python -m scripts.tune_hparams --config v2
    python -m scripts.tune_hparams --config v2 --out /tmp/v2_grid.csv

Provenance for the hyperparameters recorded in ``src.model.config``. Runs a
3-fold stratified CV over a modest grid (depth / learning rate /
min_child_weight / L2) on the **training split only** -- the validation
split is never touched here, so it remains a clean hold-out when two
versions' eval reports are compared. Each fold fits with early stopping, so
``n_estimators`` is not a grid axis; the mean best iteration is reported
instead.

Writes a CSV of every combination (sorted by mean CV AUC) somewhere outside
``models/``; it does not train or save a model version. Copy the winning row
into the config by hand -- the point is that the chosen numbers are in
version control with the reasoning next to them, not that a script decides.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.model.config import DEFAULT_CONFIG, MODEL_CONFIGS, RANDOM_SEED, VAL_FRACTION, get_config
from src.model.dataset import build_model_frame, split_xy, train_val_split
from src.model.evaluate import compute_metrics
from src.model.train import DEFAULT_DATA_PATH, train_model

GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05],
    "min_child_weight": [1.0, 5.0, 20.0],
    "reg_lambda": [1.0, 5.0],
}
N_FOLDS = 3
N_ESTIMATORS_CAP = 2000  # early stopping picks the real length


def run_grid(
    data_path: str | Path = DEFAULT_DATA_PATH,
    config: str = DEFAULT_CONFIG,
    grid: dict | None = None,
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    cfg = get_config(config)
    grid = GRID if grid is None else grid

    X, y = split_xy(build_model_frame(data_path))
    X = X[cfg.select_feature_names(X.columns)]
    X_train, _, y_train, _ = train_val_split(X, y, VAL_FRACTION, RANDOM_SEED)

    folds = list(
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        .split(X_train, y_train)
    )

    rows = []
    t0 = time.time()
    for combo in itertools.product(*grid.values()):
        overrides = dict(zip(grid.keys(), combo))
        params = {**cfg.xgb_params, **overrides, "n_estimators": N_ESTIMATORS_CAP}
        aucs, iters = [], []
        for fit_idx, eval_idx in folds:
            model = train_model(
                X_train.iloc[fit_idx],
                y_train.iloc[fit_idx],
                X_train.iloc[eval_idx],
                y_train.iloc[eval_idx],
                params=params,
                early_stopping_rounds=cfg.early_stopping_rounds,
            )
            prob = model.predict_proba(X_train.iloc[eval_idx])[:, 1]
            aucs.append(compute_metrics(y_train.iloc[eval_idx], prob)["auc_roc"])
            iters.append(int(model.best_iteration))
        row = {
            **overrides,
            "cv_auc_mean": float(np.mean(aucs)),
            "cv_auc_std": float(np.std(aucs)),
            "best_iteration_mean": float(np.mean(iters)),
        }
        rows.append(row)
        print(json.dumps(row), f"[{time.time() - t0:.0f}s]", flush=True)

    return pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--config", choices=sorted(MODEL_CONFIGS), default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None, help="CSV of all results")
    args = parser.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(f"Training data not found: {args.data}")
    if args.out is not None and "models" in args.out.resolve().parts:
        raise SystemExit("--out must not point inside models/ (version dirs are immutable)")

    results = run_grid(args.data, config=args.config)
    print()
    print(results.head(10).to_string(index=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
