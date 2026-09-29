"""KAN construction and training, using the current pykan API.

API notes (pykan 0.2.8, the version this repo now targets):

* ``model.train()`` was renamed ``fit()`` - ``train`` collides with
  ``nn.Module.train``. ``fit`` also takes ``loss_fn``, so no hand-rolled loop.
* ``model.prune()`` takes ``node_th`` / ``edge_th``; the old ``threshold=``
  kwarg raises TypeError.
* ``model.speed()`` disables the symbolic branch (worth it for accuracy-only runs).
* Classification labels are long class indices, used with ``CrossEntropyLoss``.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from typing import Any

import numpy as np
import torch

from .config import Config

# pykan prints this before failing the grid update with UnboundLocalError; the message
# is the only signal that distinguishes it from any other UnboundLocalError.
_LSTSQ_FAILURE = "lstsq failed"


def resolve_device(preference: str = "auto") -> str:
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
    return preference


def build_kan(cfg: Config, n_features: int, device: str, ckpt_path: str = "./model") -> Any:
    from kan import KAN

    hidden = cfg.model.hidden
    width = [n_features, hidden, 2] if hidden else [n_features, 2]
    model = KAN(
        width=width,
        grid=cfg.model.grid,
        k=cfg.model.k,
        seed=cfg.model.seed,
        device=device,
        # auto_save must stay on: prune()/fix_symbolic() write history.txt into the
        # checkpoint directory. ckpt_path is pointed at the run's results folder so
        # nothing lands in the current working directory.
        auto_save=True,
        ckpt_path=ckpt_path,
    )
    if not cfg.symbolic.enabled:
        model = model.speed()
    return model


def loss_function(name: str) -> torch.nn.Module:
    if name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    if name == "mse":
        return torch.nn.MSELoss()
    raise ValueError(f"unknown loss {name!r}")


def _fit(
    model: Any, dataset: dict[str, Any], cfg: Config, metrics: Any, update_grid: bool
) -> tuple[dict[str, Any], str]:
    """Run one fit, capturing pykan's stdout so its failure message can be inspected."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = model.fit(
            dataset,
            opt=cfg.model.optimiser,
            steps=cfg.model.steps,
            log=1,
            lamb=cfg.model.lamb,
            loss_fn=loss_function(cfg.model.loss),
            metrics=metrics,
            update_grid=update_grid,
        )
    return result, buffer.getvalue()


def train_kan(
    model: Any,
    dataset: dict[str, Any],
    cfg: Config,
    metrics: Any = None,
    log: Any = print,
) -> dict[str, Any]:
    """Train through pykan's own ``fit`` and return its metric history.

    On CUDA, pykan's grid update solves a batched spline least-squares problem with
    torch's ``gels`` driver, which needs a full-rank matrix. A rank-deficient spline
    matrix (several standardised features near-constant is enough) makes pykan print
    "lstsq failed" and then raise ``UnboundLocalError`` from an unassigned local. Only
    that specific failure is retried, with the grid update off; anything else propagates.
    """
    try:
        result, captured = _fit(model, dataset, cfg, metrics, update_grid=cfg.model.update_grid)
    except UnboundLocalError:
        if not cfg.model.update_grid or model.device == "cpu":
            raise
        log(
            "  pykan grid update failed (rank-deficient lstsq on "
            f"{model.device}); retrying with update_grid=False"
        )
        return _fit(model, dataset, cfg, metrics, update_grid=False)[0]
    sys.stdout.write(captured)
    return result


def forward_logits(model: Any, inputs: Any) -> torch.Tensor:
    """Raw model output (one logit pair per row), evaluated without gradients."""
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(np.asarray(inputs), dtype=torch.get_default_dtype())
    inputs = inputs.to(model.device)
    with torch.no_grad():
        return model(inputs)


def prune_kan(model: Any, x: Any) -> Any:
    """Prune dead nodes/edges before symbolic regression (pykan 0.2.8 signature).

    Refuses a model in speed mode: pykan 0.2.8's ``prune()`` runs a forward pass with
    activations disabled (``speed()`` sets ``save_act=False``) and dies with
    ``IndexError: list index out of range`` inside MultKAN.forward, so pruning and speed
    mode are mutually exclusive.
    """
    if not model.symbolic_enabled or not model.save_act:
        raise ValueError(
            "cannot prune a model in speed mode (speed mode exists for runs that skip "
            "pruning and symbolic regression); pykan 0.2.8 prune() raises IndexError there"
        )
    model(x)  # prune reads cached activations
    return model.prune(node_th=1e-2, edge_th=3e-2)
