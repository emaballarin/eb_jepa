#!/bin/bash
# End-to-end smoke test on the `debug` partition (<=15 min walltime).
# Runs ONE epoch with a tiny rollout to validate the env, GPU, dataset staging,
# and checkpoint write before launching the real benchmark runs.
#
# Submit from this directory (so slurm_logs/ resolves):
#   module load intel/slurm
#   cd hpc/franklin && mkdir -p slurm_logs && sbatch smoke_debug.sbatch
#SBATCH --job-name=vjepa-mmnist-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:14:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail

# --- Locations (override by exporting before `sbatch`) ---
REPO_DIR="${EBJEPA_REPO:-$HOME/repositories/eb_jepa}"
VENV="${EBJEPA_VENV:-$HOME/venvs/eb_jepa/.venv}"
export EBJEPA_DSETS="${EBJEPA_DSETS:-$HOME/datasets/eb_jepa}"        # holds mnist_test_seq.npy
export EBJEPA_CKPTS="${EBJEPA_CKPTS:-$HOME/eb_jepa_runs/checkpoints}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# --- Offline guards (compute nodes have no internet) ---
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$EBJEPA_CKPTS"
source "$VENV/bin/activate"
cd "$REPO_DIR"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun python -m examples.video_jepa.main \
  --fname examples/video_jepa/cfgs/default.yaml \
  logging.log_wandb=false \
  logging.tqdm_silent=true \
  optim.epochs=1 \
  model.steps=2 \
  data.batch_size=16 \
  data.num_workers=2
