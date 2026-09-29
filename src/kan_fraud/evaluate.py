"""Honest evaluation for an imbalanced detection task.

Accuracy is reported for completeness only: with a 2.25 % positive rate, always
predicting "no fraud" scores ~97.75 %. The numbers that decide anything here are
precision, recall, F1 and PR-AUC, each shown next to that trivial baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc")

# Select a production threshold on a validation split. A sweep over the test set is a
# diagnostic curve, never a tuned parameter.
SWEEP_NOTE = (
    "diagnostic only: this sweep is computed on the test split, so its optimum must not "
    "be adopted as a production threshold. Select the operating point on a validation "
    "split instead."
)


def probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax over the two output logits -> probability of the fraud class."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True))[:, 1]


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "n": int(y_true.size),
        "positives": int(y_true.sum()),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metrics


def majority_baseline(y_true: np.ndarray) -> dict[str, Any]:
    """The 'always predict no fraud' classifier - the bar any model must clear.

    Built from ``compute_metrics`` on a constant predictor so the two dicts cannot drift
    apart: PR-AUC of a constant predictor is exactly the positive rate, and ROC-AUC 0.5.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    return compute_metrics(y_true, np.zeros_like(y_true, dtype=np.float64))


def threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray, step: float = 0.02
) -> dict[str, Any]:
    """Show what the 0.5 default costs, instead of accepting it silently.

    A missed fraud usually costs more than a false alarm, so the useful question is not
    "what is the accuracy at 0.5" but "which threshold maximises F1, and what recall
    does it buy". See ``SWEEP_NOTE``: the optimum here is a diagnostic, not a decision.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    curve = []
    for threshold in np.round(np.arange(step, 1.0, step), 4):
        metrics = compute_metrics(y_true, y_prob, threshold=float(threshold))
        curve.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["confusion"]["tp"],
                "fp": metrics["confusion"]["fp"],
            }
        )

    return {
        "note": SWEEP_NOTE,
        "at_0.5": next((p for p in curve if abs(p["threshold"] - 0.5) < step / 2), None),
        "best_f1": max(curve, key=lambda point: point["f1"]),
        "best_balanced": max(curve, key=lambda point: point["precision"] + point["recall"]),
        "curve": curve,
    }


def format_report(
    title: str, metrics: dict[str, Any], baseline: dict[str, Any] | None = None
) -> str:
    """Plain-text metric block: no accuracy-only headlines."""
    rows = []
    for key in METRIC_KEYS:
        value = metrics.get(key)
        line = f"  {key:<10} {value:.4f}" if isinstance(value, float) else f"  {key:<10} n/a"
        if baseline and isinstance(value, float):
            base = baseline.get(key)
            if isinstance(base, float):
                line += f"   (majority baseline {base:.4f}, lift {value - base:+.4f})"
        rows.append(line)
    cm = metrics.get("confusion")
    if cm:
        rows.append(f"  confusion  tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}")
    header = f"{title} (n={metrics.get('n')}, positives={metrics.get('positives')})"
    return "\n".join([header, *rows])
