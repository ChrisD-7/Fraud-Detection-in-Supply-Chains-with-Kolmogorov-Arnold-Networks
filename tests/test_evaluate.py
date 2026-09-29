"""Metrics: imbalance-aware numbers and the trivial baseline they must beat."""

from __future__ import annotations

import numpy as np
import sympy as sp

from kan_fraud.evaluate import (
    compute_metrics,
    format_report,
    majority_baseline,
    probabilities,
    threshold_sweep,
)
from kan_fraud.symbolic import describe, formula_accuracy


def test_probabilities_are_softmax_of_the_fraud_logit():
    logits = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    probs = probabilities(logits)
    assert np.allclose(probs, [0.5, 0.0, 1.0], atol=1e-4)
    assert probs.min() >= 0 and probs.max() <= 1


def test_metrics_recover_a_known_confusion_matrix():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.6, 0.4])
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["confusion"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert abs(metrics["accuracy"] - 0.6) < 1e-9


def test_majority_baseline_is_the_accuracy_trap():
    y_true = np.array([0] * 97 + [1] * 3)
    baseline = majority_baseline(y_true)
    assert abs(baseline["accuracy"] - 0.97) < 1e-9
    assert baseline["recall"] == 0.0
    assert baseline["f1"] == 0.0
    assert abs(baseline["pr_auc"] - 0.03) < 1e-9


def test_perfect_classifier_beats_the_baseline():
    y_true = np.array([0] * 97 + [1] * 3)
    perfect = compute_metrics(y_true, y_true.astype(float))
    assert perfect["f1"] == 1.0
    assert perfect["pr_auc"] == 1.0
    assert perfect["accuracy"] > majority_baseline(y_true)["accuracy"]


def test_threshold_sweep_finds_a_better_operating_point_than_0_5():
    """A well-ranked but badly calibrated model: 0.5 is not the best threshold."""
    y_true = np.array([0] * 90 + [1] * 10)
    # every positive scores 0.4, every negative 0.1 -> ranking perfect, 0.5 predicts nothing
    y_prob = np.concatenate([np.full(90, 0.1), np.full(10, 0.4)])
    sweep = threshold_sweep(y_true, y_prob)
    assert sweep["at_0.5"]["tp"] == 0
    assert sweep["best_f1"]["f1"] > 0.9
    assert sweep["best_f1"]["threshold"] < 0.5
    assert len(sweep["curve"]) > 10


def test_threshold_sweep_reports_the_50_point():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.4, 0.6])
    sweep = threshold_sweep(y_true, y_prob, step=0.1)
    assert sweep["at_0.5"]["threshold"] == 0.5
    assert sweep["best_f1"]["f1"] == 1.0


def test_report_prints_the_baseline_lift():
    y_true = np.array([0] * 97 + [1] * 3)
    text = format_report("test", compute_metrics(y_true, y_true.astype(float)), majority_baseline(y_true))
    assert "majority baseline" in text
    assert "confusion" in text


def test_formula_accuracy_takes_the_argmax_of_both_outputs():
    x = np.array([[-1.0], [1.0], [-2.0], [2.0]])
    y = np.array([0, 1, 0, 1])
    clean, fraud = sp.symbols("x_1"), sp.symbols("x_1")
    exprs = [sp.Integer(0), fraud]  # fraud logit = x_1 -> positive input means fraud
    assert formula_accuracy(exprs, x, y) == 1.0


def test_formula_accuracy_rejects_wrong_output_count():
    import pytest

    with pytest.raises(ValueError):
        formula_accuracy([sp.Integer(0)], np.zeros((2, 1)), np.zeros(2))


def test_compile_formulas_is_reusable_across_splits():
    """Lambdifying once and re-scoring must work, including a constant output."""
    from kan_fraud.symbolic import compile_formulas, formula_accuracy_compiled

    compiled = compile_formulas([sp.Integer(0), sp.Symbol("x_1")], 1)
    assert formula_accuracy_compiled(compiled, np.array([[-1.0], [1.0]]), np.array([0, 1])) == 1.0
    # Same callables, a different split: no recompilation, still correct.
    assert formula_accuracy_compiled(compiled, np.array([[-2.0], [3.0]]), np.array([0, 1])) == 1.0


def test_formula_describe_maps_symbols_to_feature_names():
    exprs = [sp.Integer(0), sp.Symbol("x_2")]
    text = describe(exprs, ["Days shipping", "Sales"])
    assert "x_2=Sales" in text
