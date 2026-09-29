"""Symbolic regression on a trained KAN, with a valid accuracy definition.

The original notebook rounded a single output score and compared it to a 0/1 label.
A two-output KAN is a classifier: the decision is ``logit_fraud > logit_clean``, i.e.
the argmax over the two symbolic expressions - exactly what the pykan classification
example does.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from typing import Any, Callable

import numpy as np
import sympy as sp

from .config import Config

_R2_PATTERN = re.compile(r"r2=([0-9.eE+-]+)")


def auto_symbolic(model: Any, cfg: Config) -> tuple[Any, list[float], str]:
    """Fit symbolic functions per edge; keep edges whose best fit misses r2_threshold.

    ``r2_threshold`` is a pykan >= 0.2.8 knob: below it the edge stays a spline instead
    of being forced onto a wrong closed form.

    pykan returns ``None`` and only *prints* the per-edge fits, so r² is parsed out of
    that output. The parse sees only the fittings it announces, so ``len(r2_values)`` is
    a lower bound on the number of symbolically fitted edges; the raw log is returned
    alongside for that reason.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.auto_symbolic(
            lib=cfg.symbolic.lib,
            weight_simple=cfg.symbolic.weight_simple,
            r2_threshold=cfg.symbolic.r2_threshold,
        )
    captured = buffer.getvalue()
    r2 = [float(m) for m in _R2_PATTERN.findall(captured)]
    return model, r2, captured


def formulas(model: Any) -> list[sp.Expr]:
    """The two output expressions (clean logit, fraud logit)."""
    return list(model.symbolic_formula()[0])


def compile_formulas(exprs: list[sp.Expr], n_features: int) -> list[Callable]:
    """Lambdify each output formula once, then reuse the callables across splits."""
    if len(exprs) != 2:
        raise ValueError(f"expected 2 output formulas, got {len(exprs)}")
    symbols = sp.symbols(f"x_1:{n_features + 1}")

    def compile_one(expr: sp.Expr) -> Callable:
        function = sp.lambdify(symbols, expr, "numpy")

        def run(x: np.ndarray) -> np.ndarray:
            values = np.asarray(function(*x.T), dtype=np.float64)
            if values.ndim == 0:  # a constant output still needs one value per row
                values = np.full(x.shape[0], float(values), dtype=np.float64)
            return values

        return run

    return [compile_one(expr) for expr in exprs]


def formula_accuracy_compiled(compiled: list[Callable], x: np.ndarray, y: np.ndarray) -> float:
    """Accuracy of a compiled closed form, decided by comparing the output logits."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y).astype(int).ravel()
    logits = np.stack([function(x) for function in compiled], axis=1)
    predictions = np.argmax(np.nan_to_num(logits, nan=-np.inf), axis=1)
    return float((predictions == y).mean())


def formula_accuracy(exprs: list[sp.Expr], x: np.ndarray, y: np.ndarray) -> float:
    """Compile-and-score in one call, for a single split."""
    x = np.asarray(x, dtype=np.float64)
    return formula_accuracy_compiled(compile_formulas(exprs, x.shape[1]), x, y)


def describe(exprs: list[sp.Expr], names: list[str] | None = None, simplify: bool = False) -> str:
    """Readable formulas, with the input symbols mapped back to feature names.

    ``simplify`` is off by default: sympy simplification of a pruned KAN formula costs
    ~100 s per output and changes nothing else in the report.
    """
    lines = [
        f"  out{index}: {sp.simplify(expr) if simplify else expr}"
        for index, expr in enumerate(exprs)
    ]
    if names:
        mapping = {}
        for symbol in sorted({s for e in exprs for s in e.free_symbols}, key=lambda s: s.name):
            if not symbol.name.startswith("x_"):
                continue
            position = int(symbol.name.split("_")[1]) - 1
            if 0 <= position < len(names):
                mapping[symbol.name] = names[position]
        if mapping:
            lines.append("  symbols: " + ", ".join(f"{k}={v}" for k, v in mapping.items()))
    return "\n".join(lines)
