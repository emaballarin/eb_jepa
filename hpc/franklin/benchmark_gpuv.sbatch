#!/bin/bash
# Single-GPU Moving MNIST (video_jepa) benchmark on the `gpuv` partition
# (NVIDIA Tesla V100 SXM2 16/32GB). Most plentiful GPU partition on Franklin;
# smaller VRAM, so a more modest batch size and longer walltime budget.
#
# Submit from this directory (so slurm_logs/ resolves):
#   module load intel/slurm
#   cd hpc/franklin && mkdir -p slurm_logs && sbatch benchmark_gpuv.sbatch
#SBATCH --job-name=vjepa-mmnist-v100
#SBATCH --partition=gpuv
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:59:59
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

# Reduce allocator fragmentation (see the CUDA-OOM note in README)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$EBJEPA_CKPTS"
source "$VENV/bin/activate"
cd "$REPO_DIR"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# NOTE: every override is a Fire flag (`--key=value`); booleans must be
# Python-capitalized (`False`/`True`) so Fire parses them as bool, not str.
srun python -m examples.video_jepa.main \
  --fname=examples/video_jepa/cfgs/default.yaml \
  --logging.log_wandb=False \
  --logging.tqdm_silent=True \
  --data.batch_size=16 \
  --data.num_workers=4 \
  --optim.epochs=50
