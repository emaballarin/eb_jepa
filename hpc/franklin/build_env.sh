#!/bin/bash
# Build the in-home uv environment for the EB-JEPA Moving MNIST benchmark.
#
# Run this ON THE FRANKLIN LOGIN NODE (it has internet; compute nodes do not).
# Requires `uv` on PATH:  curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Result:
#   * venv at $EBJEPA_VENV_HOME/.venv  (default: ~/venvs/eb_jepa/.venv)
#   * nightly PyTorch + torchvision from the cu126 channel
#   * eb_jepa installed editable (--no-deps) from this repo checkout
#
# Override the venv location by exporting EBJEPA_VENV_HOME before running.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_HOME="${EBJEPA_VENV_HOME:-$HOME/venvs/eb_jepa}"

echo ">> Repo:       $REPO_DIR"
echo ">> Venv home:  $VENV_HOME"

mkdir -p "$VENV_HOME"
cp "$REPO_DIR/hpc/franklin/eb_jepa-env.pyproject.toml" "$VENV_HOME/pyproject.toml"

# Resolve + install the (minimal) third-party deps, with uv-managed CPython 3.13.
uv sync --project "$VENV_HOME" --python 3.13

# Make `eb_jepa` importable without re-resolving its heavy dependency list.
# This relies on the repo pyproject's requires-python being relaxed to allow 3.13.
uv pip install --python "$VENV_HOME/.venv/bin/python" -e "$REPO_DIR" --no-deps

echo
echo ">> Build complete. Sanity check:"
"$VENV_HOME/.venv/bin/python" - <<'PY'
import torch, torchvision
print(f"   torch        {torch.__version__}")
print(f"   torchvision  {torchvision.__version__}")
print(f"   torch.version.cuda  {torch.version.cuda}")
PY
echo ">> Venv: $VENV_HOME/.venv"
