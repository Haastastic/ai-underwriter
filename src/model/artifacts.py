"""Persist and reload a versioned model artifact directory.

Layout (a directory is never overwritten -- a new run takes the next vN):

    models/v1/
        model.json           XGBoost native format (portable, no pickle)
        feature_names.json    ordered feature list the model expects
        eval_report.json      AUC / KS / Brier on the validation split
        calibration.png       reliability diagram
        metadata.json         params, seed, split sizes, versions, timestamp
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import xgboost as xgb

from src.model.evaluate import evaluate_predictions

MODELS_ROOT = Path("models")
_VERSION_RE = re.compile(r"^v(\d+)$")

_ARTIFACT_FILES = (
    "model.json",
    "feature_names.json",
    "eval_report.json",
    "calibration.png",
    "metadata.json",
)


def next_version_dir(models_root: str | Path = MODELS_ROOT) -> Path:
    """Return `<models_root>/v{N+1}` where vN is the highest existing version."""
    models_root = Path(models_root)
    versions = []
    if models_root.exists():
        for p in models_root.iterdir():
            m = _VERSION_RE.match(p.name)
            if p.is_dir() and m:
                versions.append(int(m.group(1)))
    return models_root / f"v{max(versions, default=0) + 1}"


def save_artifacts(
    target_dir: str | Path,
    *,
    model: xgb.XGBClassifier,
    feature_names,
    y_val,
    y_val_prob,
    metadata: dict,
) -> tuple[Path, dict]:
    """Write all artifact files into `target_dir` and return (dir, eval_report).

    Refuses to write into an existing directory: model versions are
    immutable so that an eval report always describes the model sitting
    next to it.
    """
    target_dir = Path(target_dir)
    if target_dir.exists():
        raise FileExistsError(
            f"{target_dir} already exists; model versions are never overwritten"
        )
    target_dir.mkdir(parents=True)

    model.save_model(target_dir / "model.json")
    _write_json(target_dir / "feature_names.json", list(feature_names))

    eval_report = evaluate_predictions(
        y_val, y_val_prob, plot_path=target_dir / "calibration.png"
    )
    _write_json(target_dir / "eval_report.json", eval_report)
    _write_json(
        target_dir / "metadata.json",
        {**metadata, "saved_at": datetime.now(timezone.utc).isoformat()},
    )
    return target_dir, eval_report


def load_model(model_dir: str | Path) -> tuple[xgb.XGBClassifier, list[str]]:
    """Rehydrate the classifier and its feature-name list from `model_dir`."""
    model_dir = Path(model_dir)
    feature_names = json.loads((model_dir / "feature_names.json").read_text())
    model = xgb.XGBClassifier()
    model.load_model(model_dir / "model.json")
    return model, feature_names


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))
