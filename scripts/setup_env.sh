#!/usr/bin/env bash
# Create the conda environment this project is verified against and run the tests.
#
#   bash scripts/setup_env.sh            # CUDA build of torch (default here)
#   TORCH_INDEX=cpu bash scripts/setup_env.sh   # CPU-only machine
#
# Verified stack: python 3.11, pykan 0.2.8, torch 2.6.0, numpy 2.4, pandas 3.0.
set -euo pipefail

ENV="${ENV:-kan-fraud}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
PIP="$HOME/miniconda3/envs/$ENV/bin/pip"
PY="$HOME/miniconda3/envs/$ENV/bin/python"

# If PYTHONPATH points at another interpreter's site-packages (Hermes does this), it
# shadows this env and imports/pip silently report the wrong environment. Strip it.
if [ -n "${PYTHONPATH:-}" ]; then
  echo "note: ignoring PYTHONPATH=$PYTHONPATH for this environment"
  unset PYTHONPATH
fi

"$CONDA" create -y -n "$ENV" python=3.11
"$PIP" install --upgrade pip wheel setuptools

if [ "${TORCH_INDEX:-cu124}" = "cpu" ]; then
  "$PIP" install torch --index-url https://download.pytorch.org/whl/cpu
else
  "$PIP" install torch --index-url "https://download.pytorch.org/whl/${TORCH_INDEX:-cu124}"
fi

"$PIP" install -e "$ROOT"
"$PIP" install pytest
"$PY" -m pytest -q
echo "environment '$ENV' ready - run: conda activate $ENV && python -m kan_fraud.run --config configs/smoke.yaml"
