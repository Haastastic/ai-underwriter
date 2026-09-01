"""Re-evaluate a saved model version against a dataset and print its metrics.

    python -m src.model.report v1
    python -m src.model.report v1 --data data/raw/cs-training.csv
    python -m src.model.report v1 --plot /tmp/v1_calibration.png

Loads models/<version>/, rebuilds the same stratified validation split from
the data pipeline, recomputes AUC / KS / Brier, and prints them as JSON.

This command is read-only with respect to models/: a version directory is
immutable (that is what lets its committed eval_report.json always describe
the model sitting next to it), so nothing here is ever written back into it.
To get a refreshed report, train a new version. `--plot` may write a
reliability diagram, but only to a path outside models/.
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

MODELS_ROOT = Path("models")


def _reject_path_under_models(path: Path) -> None:
    """Guard: refuse to write a report artifact into any versioned model dir."""
    resolved = path.resolve()
    models_resolved = MODELS_ROOT.resolve()
    if resolved == models_resolved or models_resolved in resolved.parents:
        raise ValueError(
            f"{path} is under {MODELS_ROOT}/; model version directories are "
            "immutable. Point --plot somewhere else."
        )


def build_report(
    version: str,
    data_path: str | Path = DEFAULT_DATA_PATH,
    models_root: str | Path = "models",
    plot_path: str | Path | None = None,
) -> dict:
    """Recompute AUC / KS / Brier for a saved version on `data_path`'s val split.

    `plot_path`, if given, is where a fresh calibration plot is written; it
    must not resolve to a location inside models/.
    """
    if plot_path is not None:
        _reject_path_under_models(Path(plot_path))

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

    return evaluate_predictions(y_val, y_val_prob, plot_path=plot_path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="model version dir name, e.g. v1")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="write a fresh calibration plot here (must be outside models/)",
    )
    args = parser.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(f"Data not found: {args.data}")

    try:
        report = build_report(
            args.version,
            args.data,
            models_root=args.models_root,
            plot_path=args.plot,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
