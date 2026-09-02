"""Per-version training configuration for the XGBoost underwriting model.

A model version is fully described by (a) its :class:`ModelConfig` entry in
:data:`MODEL_CONFIGS` and (b) the data-layer pipeline output, so any version
is reproducible from the CSV alone::

    python -m src.model.train --config v1
    python -m src.model.train --config v2      # the default

The two things a config controls:

* **Which engineered columns the model sees.** The data layer always emits
  the same full column set; a config *excludes* names from it. v1 uses
  everything. v2 excludes ``age`` and every age-derived feature -- the
  less-discriminatory-alternative search the Phase 8 fairness audit called
  for -- so it is a config difference from v1, not a fork of the pipeline.
* **XGBoost hyperparameters.** v2's were chosen by a small 3-fold CV grid on
  the *training* split only (``scripts/tune_hparams.py``), so the validation
  split stays a clean hold-out for the v1-vs-v2 comparison.

The train/validation split (``RANDOM_SEED`` / ``VAL_FRACTION``) is shared by
every version on purpose: it is what makes the eval reports comparable.

No calibration-distorting tricks are used in any version. ``scale_pos_weight``
(or any other reweighting of the rare positive class) would inflate predicted
probabilities; probability calibration is one of the three metrics every
model version must report, so each model is trained on the natural class
ratio and its raw probabilities are what get evaluated.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.features.engineer import AGE_DERIVED_FEATURES

RANDOM_SEED = 42
VAL_FRACTION = 0.2

# v1: deliberately shallow and regularised. The value of this demo is the
# SHAP / adverse-action story, which reads better off a modestly sized tree
# ensemble than off a deep, high-variance one.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_estimators": 600,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

EARLY_STOPPING_ROUNDS = 40


@dataclass(frozen=True)
class ModelConfig:
    """Everything that distinguishes one model version's training run."""

    name: str
    description: str
    # Engineered column names the model must not see. Applied as an
    # exclusion (rather than an allow-list) so a version does not have to
    # enumerate every column it *does* use, and a name listed here that the
    # pipeline no longer produces is simply a no-op.
    excluded_features: tuple[str, ...] = ()
    xgb_params: dict[str, Any] = field(default_factory=lambda: dict(XGB_PARAMS))
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS

    def select_feature_names(self, columns: Iterable[str]) -> list[str]:
        """Return `columns` minus the excluded names, order preserved."""
        excluded = set(self.excluded_features)
        return [c for c in columns if c not in excluded]

    def to_metadata(self) -> dict[str, Any]:
        """Plain-dict form for a version's ``metadata.json``."""
        return {
            "config": self.name,
            "description": self.description,
            "excluded_features": list(self.excluded_features),
            "early_stopping_rounds": self.early_stopping_rounds,
            "params": dict(self.xgb_params),
        }


V1 = ModelConfig(
    name="v1",
    description=(
        "Baseline: every engineered column, including age and the "
        "age-derived features."
    ),
    excluded_features=(),
    xgb_params=dict(XGB_PARAMS),
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)

# v2 hyperparameters, from `scripts/tune_hparams.py --config v2`: a 3-fold
# stratified CV on the training split over max_depth {3,4,5} x
# learning_rate {0.03,0.05} x min_child_weight {1,5,20} x reg_lambda {1,5},
# early stopping per fold. The surface is flat -- the whole grid spans
# 0.0006 AUC against a fold-to-fold std of ~0.004 -- so the choice is
# "slower and more regularised than v1" rather than a decisive winner:
# the best row (depth 5) and this depth-4 row differ by 0.00003 AUC, and
# depth 4 is kept because a shallow ensemble is what keeps the SHAP /
# adverse-action story readable (the same reasoning as v1).
V2_XGB_PARAMS = {
    **XGB_PARAMS,
    "n_estimators": 2000,  # early stopping picks the actual length
    "max_depth": 4,
    "learning_rate": 0.03,
    "min_child_weight": 20.0,
    "reg_lambda": 5.0,
}

V2 = ModelConfig(
    name="v2",
    description=(
        "Less-discriminatory alternative: age and every age-derived feature "
        "removed from the model; hyperparameters re-tuned by CV on the "
        "training split. The fairness audit still groups by age -- it reads "
        "age from the cleaned data, not from the model's features."
    ),
    excluded_features=tuple(AGE_DERIVED_FEATURES),
    xgb_params=V2_XGB_PARAMS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)

MODEL_CONFIGS: dict[str, ModelConfig] = {V1.name: V1, V2.name: V2}
# v2 is the shipped default: what `python -m src.model.train` builds and what
# the API serves unless AIU_MODEL_VERSION says otherwise.
DEFAULT_CONFIG = V2.name


def get_config(name_or_config: str | ModelConfig = DEFAULT_CONFIG) -> ModelConfig:
    """Resolve a config by name (``"v1"``, ``"v2"``) or pass one through."""
    if isinstance(name_or_config, ModelConfig):
        return name_or_config
    try:
        return MODEL_CONFIGS[name_or_config]
    except KeyError:
        raise ValueError(
            f"Unknown model config {name_or_config!r}; "
            f"expected one of {sorted(MODEL_CONFIGS)}"
        ) from None


def align_features(X, feature_names: Sequence[str]):
    """Return ``X[feature_names]`` after checking every name is present.

    The shared guard for every code path that scores a *saved* version
    against pipeline output (`src.model.report`, `src.fairness.audit`): the
    pipeline may emit more columns than a version uses (v2 leaves age out),
    which is fine, but a column the model expects and the pipeline no longer
    produces is an error, not something to fill in.
    """
    missing = [name for name in feature_names if name not in X.columns]
    if missing:
        raise ValueError(
            "feature columns from the data pipeline no longer include every "
            f"feature the saved model expects; missing: {missing}"
        )
    return X[list(feature_names)]


__all__ = [
    "RANDOM_SEED",
    "VAL_FRACTION",
    "XGB_PARAMS",
    "EARLY_STOPPING_ROUNDS",
    "ModelConfig",
    "MODEL_CONFIGS",
    "DEFAULT_CONFIG",
    "V1",
    "V2",
    "get_config",
    "align_features",
]
