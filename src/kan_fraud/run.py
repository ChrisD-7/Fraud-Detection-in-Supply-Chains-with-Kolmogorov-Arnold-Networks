"""Command-line entry point: prepare -> train -> evaluate -> symbolic -> save.

Usage::

    python -m kan_fraud.run --config configs/smoke.yaml
    python -m kan_fraud.run --config configs/default.yaml --n-sample 50000 --steps 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from . import evaluate, kan_model, symbolic
from .config import Config, load_config
from .data import UNKNOWN_CATEGORY_CODE, prepare


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KAN fraud detection on the DataCo dataset")
    parser.add_argument("--config", default="configs/smoke.yaml", help="YAML config path")
    parser.add_argument("--n-sample", type=int, default=None, help="override data.n_sample")
    parser.add_argument("--steps", type=int, default=None, help="override model.steps")
    parser.add_argument("--device", default=None, help="override model.device (auto|cpu|cuda)")
    parser.add_argument("--run-name", default=None, help="override output.run_name")
    parser.add_argument("--results-dir", default=None, help="override output.results_dir")
    parser.add_argument("--no-symbolic", action="store_true", help="skip symbolic regression")
    parser.add_argument(
        "--simplify-formulas",
        action="store_true",
        help="sympy-simplify the symbolic formula in the report (~100 s per output)",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.n_sample is not None:
        cfg.data.n_sample = args.n_sample
    if args.steps is not None:
        cfg.model.steps = args.steps
    if args.device is not None:
        cfg.model.device = args.device
    if args.run_name is not None:
        cfg.output.run_name = args.run_name
    if args.results_dir is not None:
        cfg.output.results_dir = args.results_dir
    if args.no_symbolic:
        # One switch: turning symbolic off also enables pykan's speed mode.
        cfg.symbolic.enabled = False
    if args.simplify_formulas:
        cfg.symbolic.simplify_formulas = True
    return cfg.validate()


def _to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _safe_run_name(name: str) -> str:
    """Reject a run name that would escape results_dir: config and --run-name are free text."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name or Path(name).is_absolute():
        raise ValueError(f"unsafe run_name {name!r}: use a plain directory name")
    return name


def run(cfg: Config, log: Any = print, frame: Any = None) -> dict[str, Any]:
    started = time.time()
    reproducibility = kan_model.set_determinism(cfg.model.seed, log)
    device = kan_model.resolve_device(cfg.model.device)
    log(f"device: {device}")

    log("preparing data ...")
    data = prepare(cfg, frame)
    meta = data.meta
    log(
        f"  rows {meta['rows_used']} -> train {meta['train_rows']} / test {meta['test_rows']}, "
        f"features {meta['n_features']}, test base rate {meta['test_base_rate']:.4f}, "
        f"balance {meta['balance']}"
    )
    if meta.get("unseen_test_categories"):
        # The sentinel code, not the dict of per-column counts (that is printed below).
        log(
            f"  unseen test categories (encoded as {UNKNOWN_CATEGORY_CODE}): "
            f"{meta['unseen_test_categories']}"
        )

    dataset = data.to_torch(device)
    out_dir = Path(cfg.output.results_dir) / _safe_run_name(cfg.output.run_name)
    ckpt_path = out_dir / "kan_ckpt"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model = kan_model.build_kan(cfg, data.n_features, device, ckpt_path=str(ckpt_path))

    log(f"training ({cfg.model.optimiser}, {cfg.model.steps} steps) ...")
    train_started = time.time()
    # No per-step metric closures: they would re-forward the whole train and test split
    # at every logged step. Losses come from the history, everything else is measured
    # once, below, with the metrics that actually matter for this problem.
    history = kan_model.train_kan(model, dataset, cfg, metrics=None, log=log)
    train_seconds = time.time() - train_started

    train_logits = _to_numpy(kan_model.forward_logits(model, dataset["train_input"]))
    test_logits = _to_numpy(kan_model.forward_logits(model, dataset["test_input"]))
    train_metrics = evaluate.compute_metrics(data.y_train, evaluate.probabilities(train_logits))
    test_probs = evaluate.probabilities(test_logits)
    test_metrics = evaluate.compute_metrics(data.y_test, test_probs)
    baseline = evaluate.majority_baseline(data.y_test)
    sweep = evaluate.threshold_sweep(data.y_test, test_probs)
    single_baseline = evaluate.single_feature_baseline(data.x_test, data.y_test, data.feature_names)

    log(evaluate.format_report("test set - trained KAN", test_metrics, baseline))
    log(evaluate.format_report("train set - trained KAN", train_metrics))
    best = sweep["best_f1"]
    log(
        f"  threshold sweep: best F1 {best['f1']:.4f} at {best['threshold']:.2f} "
        f"(precision {best['precision']:.4f}, recall {best['recall']:.4f}, "
        f"tp {best['tp']}, fp {best['fp']})"
    )
    log(f"  note: {evaluate.SWEEP_NOTE}")
    if single_baseline["best_feature"]:
        log(
            f"  best single feature: {single_baseline['best_feature']} ROC-AUC "
            f"{single_baseline['best_roc_auc']:.4f} vs model {test_metrics['roc_auc']:.4f}; "
            f"{single_baseline['at_chance']}/{single_baseline['n_considered']} features "
            f"within {single_baseline.get('chance_band', 0.02)} of chance"
        )
        log(f"  note: {single_baseline['note']}")

    results: dict[str, Any] = {
        "config": cfg.to_dict(),
        "data": meta,
        "train": train_metrics,
        "train_note": (
            "measured on the ADASYN-resampled train split, so it includes synthetic rows "
            "and is not directly comparable to the test figures"
        ),
        "test": test_metrics,
        "majority_baseline": baseline,
        "threshold_sweep": sweep,
        "single_feature_baseline": single_baseline,
        "history": {
            key: [_to_numpy(v) if hasattr(v, "detach") else v for v in values]
            for key, values in (history or {}).items()
            if isinstance(values, list)
        },
        "train_seconds": train_seconds,
        "device": device,
        "reproducibility": reproducibility,
    }

    if cfg.symbolic.enabled:
        log("symbolic regression ...")
        if cfg.symbolic.prune:
            model = kan_model.prune_kan(model, dataset["train_input"])
        model, r2_values, r2_log = symbolic.auto_symbolic(model, cfg)
        exprs = symbolic.formulas(model)
        compiled = symbolic.compile_formulas(exprs, data.n_features)
        results["symbolic"] = {
            "r2_threshold": cfg.symbolic.r2_threshold,
            "edges_fitted": len(r2_values),
            "edges_fitted_note": "lower bound: parsed from pykan's printed edge fittings",
            "r2_mean": float(np.mean(r2_values)) if r2_values else None,
            "r2_min": float(np.min(r2_values)) if r2_values else None,
            "formula_train_accuracy": symbolic.formula_accuracy_compiled(
                compiled, data.x_train, data.y_train
            ),
            "formula_test_accuracy": symbolic.formula_accuracy_compiled(
                compiled, data.x_test, data.y_test
            ),
            "formula_train_accuracy_note": (
                "measured on the ADASYN-resampled train split, so it includes synthetic rows "
                "and is not directly comparable to the test figure"
            ),
            "edge_fit_log": r2_log,
            "formulas": [str(expr) for expr in exprs],
        }
        log(
            f"  formula test accuracy {results['symbolic']['formula_test_accuracy']:.4f} "
            f"(model test accuracy {test_metrics['accuracy']:.4f})"
        )
        log(symbolic.describe(exprs, data.feature_names, simplify=cfg.symbolic.simplify_formulas))

    results["elapsed_seconds"] = time.time() - started
    save_results(cfg, results, log)
    return results


def save_results(cfg: Config, results: dict[str, Any], log: Any = print) -> Path:
    out_dir = Path(cfg.output.results_dir) / _safe_run_name(cfg.output.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "summary.md").write_text(render_summary(results))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(results["config"], sort_keys=False))
    log(f"wrote {out_dir}/metrics.json and summary.md")
    return out_dir


def render_summary(results: dict[str, Any]) -> str:
    data, test, baseline = results["data"], results["test"], results["majority_baseline"]
    symbolic_results = results.get("symbolic", {})
    lines = [
        "# Run summary",
        "",
        f"- device: `{results['device']}`, train time: {results['train_seconds']:.1f}s",
        f"- rows used: {data['rows_used']} ({data['split_strategy']} split), "
        f"features: {data['n_features']}, balance: `{data['balance']}`",
        f"- test set: n={test['n']}, positives={test['positives']} "
        f"(base rate {data['test_base_rate']:.4f})",
        "",
        "| metric | KAN (test) | majority baseline | lift |",
        "|---|---|---|---|",
    ]
    for key in evaluate.METRIC_KEYS:
        value, base = test.get(key), baseline.get(key, 0.0)
        lines.append(f"| {key} | {value:.4f} | {base:.4f} | {value - base:+.4f} |")
    lines += ["", f"confusion matrix: {test['confusion']}", ""]

    single = results.get("single_feature_baseline") or {}
    if single.get("best_feature"):
        lines += [
            "## Best single feature",
            "",
            f"- `{single['best_feature']}` on its own: ROC-AUC {single['best_roc_auc']:.4f} "
            f"(model {test['roc_auc']:.4f})",
            f"- {single['at_chance']} of {single['n_considered']} features are within "
            f"{single.get('chance_band', 0.02)} of chance (2 SE of AUC under the null)",
            "",
            single.get("note", ""),
            "",
        ]

    sweep = results.get("threshold_sweep") or {}
    if sweep.get("best_f1"):
        best = sweep["best_f1"]
        at_half = sweep.get("at_0.5") or {}
        lines += [
            "## Threshold",
            "",
            "| threshold | precision | recall | f1 | tp | fp |",
            "|---|---|---|---|---|---|",
            f"| 0.50 (default) | {at_half.get('precision', 0):.4f} | {at_half.get('recall', 0):.4f} "
            f"| {at_half.get('f1', 0):.4f} | {at_half.get('tp', 0)} | {at_half.get('fp', 0)} |",
            f"| {best['threshold']:.2f} (best F1) | {best['precision']:.4f} | {best['recall']:.4f} "
            f"| {best['f1']:.4f} | {best['tp']} | {best['fp']} |",
            "",
            sweep.get("note", ""),
            "",
        ]
    if symbolic_results:
        lines += [
            "## Symbolic form",
            "",
            f"- edges fitted: {symbolic_results['edges_fitted']} "
            f"({symbolic_results.get('edges_fitted_note', '')})",
            f"- mean r2 {symbolic_results['r2_mean']}, min r2 {symbolic_results['r2_min']}",
            f"- formula accuracy: train {symbolic_results['formula_train_accuracy']:.4f} "
            f"({symbolic_results.get('formula_train_accuracy_note', '')}), "
            f"test {symbolic_results['formula_test_accuracy']:.4f}",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)
    run(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
