"""KAN fraud detection on the DataCo supply-chain dataset.

Leak-free preparation, current pykan API, and honest metrics for an imbalanced
detection task. See README.md for the leakage the original notebooks had.
"""

from .config import Config, load_config
from .data import PreparedData, prepare
from .evaluate import compute_metrics, majority_baseline, threshold_sweep
from .kan_model import build_kan, resolve_device, train_kan

__all__ = [
    "Config",
    "PreparedData",
    "build_kan",
    "compute_metrics",
    "load_config",
    "majority_baseline",
    "prepare",
    "resolve_device",
    "threshold_sweep",
    "train_kan",
]
