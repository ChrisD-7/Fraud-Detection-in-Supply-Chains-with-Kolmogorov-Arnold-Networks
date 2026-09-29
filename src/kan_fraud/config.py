"""Configuration objects and YAML loading for the KAN fraud-detection pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class DataConfig:
    """How the dataset is loaded, curated and split."""

    zip_path: str = "DataCoSupplyChainDataset.zip"
    encoding: str = "latin-1"
    n_sample: int | None = 20_000  # None = use every row (slow on CPU)
    seed: int = 42
    test_size: float = 0.2
    # "grouped" keeps every order (Order Id) wholly inside train or test, which closes
    # the group-leakage channel: one order holds several line items.
    split_strategy: Literal["grouped", "random"] = "grouped"
    group_column: str = "Order Id"
    # Oversampling is applied to the TRAIN split only.
    balance: Literal["adasyn", "smote", "none"] = "adasyn"
    min_minority_for_oversample: int = 50
    drop_na_rows: bool = True


@dataclass
class ModelConfig:
    hidden: int = 6
    grid: int = 3
    k: int = 3
    seed: int = 1
    device: Literal["auto", "cpu", "cuda"] = "auto"
    optimiser: Literal["LBFGS", "Adam"] = "LBFGS"
    steps: int = 10
    lamb: float = 0.0
    loss: Literal["cross_entropy", "mse"] = "cross_entropy"
    # pykan re-fits the spline grids to the data every `grid_update_num` steps. It is
    # skipped automatically if the solve is rank-deficient on CUDA.
    update_grid: bool = True


@dataclass
class SymbolicConfig:
    """The single switch for interpretability - it drives pykan's speed mode too.

    Off means ``model.speed()``, no pruning and no symbolic regression, which is what
    makes an accuracy-only run fast. There is deliberately no second flag to keep in
    sync with this one.
    """

    enabled: bool = True
    # pykan >= 0.2.8: edges whose best fit scores below r2_threshold stay splines.
    r2_threshold: float = 0.8
    weight_simple: float = 0.8
    lib: list[str] = field(
        default_factory=lambda: [
            "x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "tanh", "sin", "tan", "abs",
        ]
    )
    prune: bool = True
    # sympy simplify on a 40+ edge expression tree costs ~100 s per formula and buys
    # nothing in this report, so it is opt-in.
    simplify_formulas: bool = False


@dataclass
class OutputConfig:
    results_dir: str = "results"
    run_name: str = "smoke"
    save_figures: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> Config:
        """Reject nonsense once, at load time, instead of deep inside a run."""
        if not 0.0 < self.data.test_size < 1.0:
            raise ValueError(f"data.test_size must be in (0, 1), got {self.data.test_size}")
        if self.data.n_sample is not None and self.data.n_sample < 10:
            raise ValueError(f"data.n_sample must be None or >= 10, got {self.data.n_sample}")
        if self.model.hidden < 1 or self.model.grid < 1 or self.model.k < 1:
            raise ValueError("model.hidden, model.grid and model.k must all be >= 1")
        if self.model.steps < 1:
            raise ValueError(f"model.steps must be >= 1, got {self.model.steps}")
        if not self.symbolic.enabled:
            # Pruning is unreachable with symbolic off; state that instead of leaving a
            # combination that looks meaningful but is silently ignored.
            self.symbolic.prune = False
        return self


def _merge(target: Any, overrides: dict[str, Any]) -> Any:
    """Merge a nested dict into a dataclass instance, rejecting unknown keys."""
    valid = {f.name: f for f in fields(target)}
    for key, value in overrides.items():
        if key not in valid:
            raise ValueError(f"unknown config key {key!r} for {type(target).__name__}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, key, value)
    return target


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config file, layered on the dataclass defaults, then validate it."""
    cfg = Config()
    _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        _merge(cfg, overrides)
    return cfg.validate()
