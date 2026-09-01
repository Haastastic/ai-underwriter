"""Model-agnostic evaluation for a binary risk model.

Every model version reports all three of:
  - AUC-ROC        -- overall rank ordering
  - KS statistic   -- the standard credit-risk separation metric
  - Brier score    -- probability calibration / sharpness

plus a saved calibration (reliability) plot.

These functions take `y_true` and predicted probabilities only -- no model
object, no feature matrix -- so they are reusable across model libraries and
for post-hoc recalibration experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed to write a PNG

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov separation: max gap between the TPR and FPR curves.

    This is the maximum vertical distance between the cumulative score
    distributions of defaulters and non-defaulters -- the rank-ordering
    metric underwriting teams usually quote alongside AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def compute_metrics(y_true, y_prob) -> dict[str, float]:
    """Return {'auc_roc', 'ks_statistic', 'brier_score'} for the predictions."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "ks_statistic": ks_statistic(y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def save_calibration_plot(y_true, y_prob, path: str | Path, n_bins: int = 10) -> Path:
    """Write a reliability diagram + predicted-probability histogram to `path`.

    Quantile bins (equal count per bin) are used so the diagram is not
    dominated by the empty high-probability region that a low-default-rate
    model produces.
    """
    path = Path(path)
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 7), height_ratios=[3, 1]
    )

    ax_cal.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax_cal.plot(mean_pred, frac_pos, "o-", label="model")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xlabel("mean predicted probability (quantile bins)")
    ax_cal.set_ylabel("observed default rate")
    ax_cal.set_title("Calibration")
    ax_cal.legend(loc="upper left")

    ax_hist.hist(y_prob, bins=30, range=(0, 1))
    ax_hist.set_xlabel("predicted probability")
    ax_hist.set_ylabel("count (log)")
    ax_hist.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def evaluate_predictions(y_true, y_prob, plot_path: str | Path | None = None) -> dict:
    """Compute the metric dict and, if `plot_path` is given, save the plot."""
    metrics = compute_metrics(y_true, y_prob)
    if plot_path is not None:
        save_calibration_plot(y_true, y_prob, plot_path)
    return metrics
