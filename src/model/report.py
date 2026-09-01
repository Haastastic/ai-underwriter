"""Re-evaluate a saved model version and print (optionally rewrite) its report.

    python -m src.model.report v1
    python -m src.model.report v1 --data data/raw/cs-training.csv --write

Loads models/<version>/, rebuilds the same stratified validation split from
the data pipeline, and recomputes AUC / KS / Brier. Useful for reproducing a
version's numbers or checking an old model against a refreshed dataset.

`--write` refreshes eval_report.json and calibration.png in place; without
it, nothing on disk changes. The model file itself is never rewritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.model.artifacts import load_model
from src.model.config import RANDOM_SEED, VAL_FRACTION
from src.model.dataset import build_model_frame, split_xy, train_val_split
from src.model.evaluate import evaluate_predictions
from src.model.train import DEFAULT_DATA_PATH


def build_report(
    version: str,
    data_path: str | Path = DEFAULT_DATA_PATH,
    models_root: str | Path = "models",
    write: bool = False,
) -> dict:
    model_dir = Path(models_root) / version
    model, feature_names = load_model(model_dir)

    df = build_model_frame(data_path)
    X, y = split_xy(df)
    if list(X.columns) != list(feature_names):
        raise ValueError(
            "Feature columns from the pipeline no longer match the saved model's "
            f"feature_names.json for {version}"
        )
    _, X_val, _, y_val = train_val_split(X, y, VAL_FRACTION, RANDOM_SEED)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    plot_path = model_dir / "calibration.png" if write else None
    report = evaluate_predictions(y_val, y_val_prob, plot_path=plot_path)
    if write:
        (model_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="model version dir name, e.g. v1")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--write",
        action="store_true",
        help="refresh eval_report.json and calibration.png in place",
    )
    args = parser.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(f"Data not found: {args.data}")

    report = build_report(
        args.version, args.data, models_root=args.models_root, write=args.write
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
