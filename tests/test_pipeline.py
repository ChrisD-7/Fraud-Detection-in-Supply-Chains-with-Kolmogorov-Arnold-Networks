"""End-to-end: build, train one step, evaluate, save - on synthetic data.

Runs a real KAN (small: 1 step, 2 hidden neurons) so the pykan API itself is
exercised without needing the 18 MB dataset or a GPU in CI.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from conftest import make_frame
from kan_fraud import evaluate, kan_model
from kan_fraud.config import Config
from kan_fraud.run import run


def test_train_and_evaluate_produces_metrics(smoke_config):
    frame = make_frame(n_rows=300, fraud_rate=0.08, seed=3)
    results = run(smoke_config, log=lambda *_: None, frame=frame)

    assert results["test"]["n"] > 0
    assert results["test"]["positives"] > 0
    for key in evaluate.METRIC_KEYS:
        assert key in results["test"]
    assert results["majority_baseline"]["recall"] == 0.0
    assert results["threshold_sweep"]["at_0.5"] is not None
    assert results["threshold_sweep"]["best_f1"]["f1"] >= results["threshold_sweep"]["at_0.5"]["f1"]
    assert "test split" in results["threshold_sweep"]["note"]
    out = Path(smoke_config.output.results_dir) / smoke_config.output.run_name
    assert (out / "metrics.json").exists()
    assert (out / "summary.md").exists()
    assert json.loads((out / "metrics.json").read_text())["test"]["n"] == results["test"]["n"]


def test_pipeline_skips_symbolic_when_disabled(smoke_config):
    results = run(smoke_config, log=lambda *_: None, frame=make_frame(n_rows=240, seed=7))
    assert "symbolic" not in results


def test_device_resolution():
    assert kan_model.resolve_device("cpu") == "cpu"
    assert kan_model.resolve_device("auto") in {"cpu", "cuda"}


def test_prune_uses_current_pykan_signature(smoke_config):
    """pykan 0.2.8 prune() is prune(node_th, edge_th); threshold= raises TypeError."""
    data = _trained_model_and_data(smoke_config, symbolic=True)
    pruned = kan_model.prune_kan(*data)
    assert pruned is not None


def test_prune_refuses_a_speed_mode_model(smoke_config):
    """Documented upstream bug: prune() after speed() raises IndexError in pykan 0.2.8."""
    data = _trained_model_and_data(smoke_config, symbolic=False)
    with pytest.raises(ValueError, match="speed mode"):
        kan_model.prune_kan(*data)


def _trained_model_and_data(cfg: Config, symbolic: bool = True):
    from kan_fraud.data import prepare

    cfg = copy.deepcopy(cfg)
    cfg.symbolic.enabled = symbolic
    cfg.symbolic.prune = False
    prepared = prepare(cfg, make_frame(n_rows=240, seed=11))
    dataset = prepared.to_torch("cpu")
    ckpt = Path(cfg.output.results_dir) / cfg.output.run_name / "kan_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    model = kan_model.build_kan(cfg, prepared.n_features, "cpu", ckpt_path=str(ckpt))
    kan_model.train_kan(model, dataset, cfg)
    return model, dataset["train_input"]


def test_unknown_config_key_is_rejected(tmp_path):
    from kan_fraud.config import load_config

    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  nope: 1\n")
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(path)


@pytest.mark.parametrize("config_name", ["smoke", "default", "full"])
def test_shipped_configs_load_and_validate(config_name):
    """A shipped config must never reference a key the dataclasses dropped."""
    from pathlib import Path

    from kan_fraud.config import load_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / f"{config_name}.yaml")
    assert cfg.model.steps >= 1
    assert cfg.symbolic.enabled in (True, False)
