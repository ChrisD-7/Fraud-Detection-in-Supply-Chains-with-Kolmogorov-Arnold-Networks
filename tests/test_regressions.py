"""Regression tests for the issues raised in the pre-merge review (2026-09-29).

Each test pins a defect that was found by an independent reviewer, so it cannot come back
unnoticed. The frame builder and fixtures come from conftest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kan_fraud import evaluate, kan_model
from kan_fraud.config import Config
from kan_fraud.data import curate
from kan_fraud.run import _safe_run_name, render_summary, run
from conftest import make_frame


# --- the single-feature baseline: one column can be the whole model -------------


def test_single_feature_baseline_finds_the_informative_column():
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10)
    informative = np.where(y == 1, 1.0, 0.0) + rng.normal(0, 0.01, size=len(y))
    noise = rng.normal(size=(len(y), 3))
    x = np.column_stack([noise[:, 0], informative, noise[:, 1], noise[:, 2]])

    result = evaluate.single_feature_baseline(x, y, ["a", "b", "c", "d"])

    assert result["best_feature"] == "b"
    assert result["best_roc_auc"] > 0.95
    assert result["n_considered"] == 4
    assert result["at_chance"] == 3  # only the informative column is away from 0.5
    assert result["chance_band"] > 0.02  # small sample -> wide band
    assert result["note"] == evaluate.SINGLE_FEATURE_NOTE
    assert [r["feature"] for r in result["ranking"]][0] == "b"


def test_single_feature_baseline_handles_a_matrix_with_no_signal():
    rng = np.random.default_rng(1)
    y = np.array([0] * 50 + [1] * 50)
    x = rng.normal(size=(100, 2))

    result = evaluate.single_feature_baseline(x, y, ["a", "b"])

    assert result["best_feature"] in {"a", "b"}
    assert result["at_chance"] == 2


def test_single_feature_baseline_skips_constant_and_nan_columns():
    y = np.array([0, 1, 0, 1])
    x = np.array([
        [1.0, 5.0, np.nan],
        [2.0, 5.0, np.nan],
        [4.0, 5.0, np.nan],
        [5.0, 5.0, np.nan],
    ])

    result = evaluate.single_feature_baseline(x, y, ["varying", "constant", "with_nan"])

    assert result["n_considered"] == 1
    assert result["best_feature"] == "varying"


def test_chance_band_widens_when_the_test_set_is_small():
    # 50/50 with pure noise: a fixed 0.02 window would call noise a signal.
    rng = np.random.default_rng(7)
    y = np.array([0] * 50 + [1] * 50)
    x = rng.normal(size=(100, 4))

    result = evaluate.single_feature_baseline(x, y, list("abcd"))

    assert result["chance_band"] > 0.02
    assert result["at_chance"] == 4


# --- curation must not let NaN reach the model ----------------------------------


def test_curate_raises_when_nan_survive_with_drop_na_rows_off():
    frame = make_frame()
    frame.loc[frame.index[0], "Sales"] = np.nan  # a kept feature, not a droppable column
    cfg = Config()
    cfg.data.drop_na_rows = False

    with pytest.raises(ValueError, match="NaN cells survived curation"):
        curate(frame, cfg)


def test_curate_drops_nan_when_asked():
    frame = make_frame()
    frame.loc[frame.index[0], "Sales"] = np.nan
    cfg = Config()
    cfg.data.drop_na_rows = True

    curated = curate(frame, cfg)

    assert not curated.isna().any().any()


# --- a free-text run name must not escape results_dir ---------------------------


@pytest.mark.parametrize("name", ["../evil", "a/b", "..", ".", "", "/tmp/abs", "a\\b"])
def test_unsafe_run_names_are_rejected(name):
    with pytest.raises(ValueError, match="unsafe run_name"):
        _safe_run_name(name)


@pytest.mark.parametrize("name", ["smoke", "default50k", "run-2026_09_29", "a.b"])
def test_plain_run_names_are_accepted(name):
    assert _safe_run_name(name) == name


# --- identical configs must produce identical numbers ---------------------------


def test_set_determinism_seeds_numpy_and_torch():
    import os
    import random

    import torch

    info = kan_model.set_determinism(1234, log=lambda *_: None)

    assert info["seed"] == 1234
    assert info["deterministic_algorithms"] is True
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == kan_model.CUBLAS_WORKSPACE_CONFIG
    assert info["cublas_workspace_config"] == kan_model.CUBLAS_WORKSPACE_CONFIG
    assert info["note"] == kan_model.DETERMINISM_NOTE

    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    kan_model.set_determinism(1234, log=lambda *_: None)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second


def test_run_records_its_reproducibility_settings(smoke_config, synthetic_frame, tmp_path):
    cfg = smoke_config
    cfg.data.n_sample = None
    cfg.data.balance = "none"
    cfg.model.steps = 1
    cfg.output.results_dir = str(tmp_path)
    cfg.output.run_name = "repro"
    cfg.symbolic.enabled = False

    results = run(cfg, log=lambda *_: None, frame=synthetic_frame)

    assert results["reproducibility"]["seed"] == cfg.model.seed
    assert results["reproducibility"]["note"] == kan_model.DETERMINISM_NOTE


# --- the saved summary must show the single-feature baseline --------------------


def test_summary_reports_the_single_feature_baseline():
    results = {
        "device": "cuda",
        "train_seconds": 1.0,
        "data": {
            "rows_used": 100, "split_strategy": "grouped", "n_features": 2,
            "balance": "none", "test_base_rate": 0.1,
        },
        "test": {
            "accuracy": 0.9, "precision": 0.2, "recall": 0.3, "f1": 0.24,
            "pr_auc": 0.2, "roc_auc": 0.88, "n": 20, "positives": 2,
            "confusion": {"tn": 17, "fp": 1, "fn": 1, "tp": 1},
        },
        "majority_baseline": {"accuracy": 0.9, "precision": 0.0, "recall": 0.0,
                              "f1": 0.0, "pr_auc": 0.1, "roc_auc": 0.5},
        "single_feature_baseline": {
            "best_feature": "Type", "best_roc_auc": 0.87, "n_considered": 2,
            "at_chance": 1, "chance_band": 0.045, "ranking": [],
            "note": evaluate.SINGLE_FEATURE_NOTE,
        },
    }

    summary = render_summary(results)

    assert "## Best single feature" in summary
    assert "`Type` on its own: ROC-AUC 0.8700" in summary
    assert "1 of 2 features are within 0.045 of chance" in summary
    assert evaluate.SINGLE_FEATURE_NOTE in summary


def test_summary_survives_a_run_without_the_baseline():
    results = {
        "device": "cpu", "train_seconds": 0.1,
        "data": {"rows_used": 10, "split_strategy": "grouped", "n_features": 1,
                 "balance": "none", "test_base_rate": 0.5},
        "test": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5,
                 "pr_auc": 0.5, "roc_auc": 0.5, "n": 2, "positives": 1,
                 "confusion": {"tn": 1, "fp": 0, "fn": 0, "tp": 1}},
        "majority_baseline": {"accuracy": 0.5, "precision": 0.0, "recall": 0.0,
                              "f1": 0.0, "pr_auc": 0.5, "roc_auc": 0.5},
    }

    summary = render_summary(results)

    assert "# Run summary" in summary
    assert "Best single feature" not in summary
