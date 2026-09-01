"""Training configuration for the XGBoost underwriting model.

A model version is fully described by (a) this config and (b) the data-layer
pipeline output, so a run is reproducible from the CSV alone.

No calibration-distorting tricks are used. `scale_pos_weight` (or any other
reweighting of the rare positive class) would inflate predicted
probabilities; probability calibration is one of the three metrics every
model version must report, so the model is trained on the natural class
ratio and its raw probabilities are what get evaluated.
"""

RANDOM_SEED = 42
VAL_FRACTION = 0.2

# Deliberately shallow and regularised: the value of this demo is the
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
