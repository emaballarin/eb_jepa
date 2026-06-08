# Moving MNIST (video_jepa) benchmark — Franklin HPC

SLURM launchers + an in-home `uv` environment to run the EB-JEPA **Moving MNIST**
example (`examples/video_jepa`) as a single-GPU benchmark on the Franklin cluster
(IIT, SLURM scheduler).

## What's here

| File | Purpose |
|------|---------|
| `eb_jepa-env.pyproject.toml` | `uv` env spec — minimal deps, **nightly PyTorch on the `cu128` channel** |
| `build_env.sh` | Build the in-home venv (`~/venvs/eb_jepa`) + editable `eb_jepa` install |
| `smoke_debug.sbatch` | 1-epoch end-to-end smoke test on the `debug` partition (≤15 min) |
| `benchmark_gpua.sbatch` | Single-GPU benchmark on `gpua` (A100 80GB) |
| `benchmark_gpuv.sbatch` | Single-GPU benchmark on `gpuv` (V100 16/32GB) |

## Why `cu128` (CUDA 12.8.x)

This mirrors the maintainer's reference research env (nightly PyTorch, uv-managed
Python, in-home venv) **except** the CUDA channel is pinned to `cu128` instead of
`cu132`. Franklin's GPU-node kernel-level NVIDIA drivers are **incompatible with
the CUDA 13+ runtime**, so `cu128` is the newest nightly channel that loads. The
`cu128` wheels bundle a CUDA 12.8.x runtime — there is nothing else to pin.

## Key facts

- **Single GPU, no DDP.** `--gres=gpu:1` ⇒ `setup_distributed()` resolves to
  world_size 1; `main.py` maps the allocated GPU to index 0 via
  `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`. No code changes.
- **Compute nodes are offline.** Everything that needs the network (uv install,
  dataset download, weight caching) must happen on the **login node**. The
  sbatch scripts set `WANDB_MODE=offline`, `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1` and disable wandb (`logging.log_wandb=false`).
- **Dataset** (`mnist_test_seq.npy`, ~800 MB) is read from `$EBJEPA_DSETS`. It
  auto-downloads over HTTP **only if missing** — so it must be pre-staged.

## Setup & run (on the login node)

```bash
# 0. uv + repo in home (skip what you already have)
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo-url> ~/repositories/eb_jepa          # or rsync your working tree

# 1. Build the environment (downloads nightly cu128 wheels)
bash ~/repositories/eb_jepa/hpc/franklin/build_env.sh
#    -> prints e.g.  torch 2.x.0.devYYYYMMDD+cu128 / torch.version.cuda 12.8

# 2. Pre-stage the dataset (mandatory — nodes are offline)
mkdir -p ~/datasets/eb_jepa
wget https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy \
     -P ~/datasets/eb_jepa/

# 3. (insurance) pre-cache LPIPS/VGG weights — off the default path, but cheap
TORCH_HOME=~/.cache/torch ~/venvs/eb_jepa/.venv/bin/python \
     -c "import lpips; lpips.LPIPS(net='vgg')" || true

# 4. Submit — smoke test first, then the benchmarks
module load intel/slurm
cd ~/repositories/eb_jepa/hpc/franklin && mkdir -p slurm_logs
sbatch smoke_debug.sbatch          # gate: must pass before the real runs
sbatch benchmark_gpua.sbatch
sbatch benchmark_gpuv.sbatch
```

### Overridable locations

The sbatch scripts and `build_env.sh` read these env vars (sensible home
defaults if unset) — export them before submitting to relocate things (e.g. to a
scratch filesystem):

| Var | Default | Used by |
|-----|---------|---------|
| `EBJEPA_REPO` | `~/repositories/eb_jepa` | sbatch (repo checkout) |
| `EBJEPA_VENV` | `~/venvs/eb_jepa/.venv` | sbatch (venv to activate) |
| `EBJEPA_VENV_HOME` | `~/venvs/eb_jepa` | `build_env.sh` (venv parent) |
| `EBJEPA_DSETS` | `~/datasets/eb_jepa` | dataset dir (`mnist_test_seq.npy`) |
| `EBJEPA_CKPTS` | `~/eb_jepa_runs/checkpoints` | checkpoint output dir |
| `TORCH_HOME` | `~/.cache/torch` | torch-hub weight cache |

## Monitoring & results

```bash
squeue --me                                   # job states
tail -f slurm_logs/vjepa-mmnist-a100_<jobid>.out
```

- Checkpoints: `$EBJEPA_CKPTS/video_jepa/...` (`latest.pth.tar`, `epoch_*.pth.tar`).
- Benchmark signal: per-epoch wall-clock / throughput and `train/loss` from the
  job logs (wandb is disabled for offline runs).

## Tuning

Append any config override in dot notation to the `srun python -m
examples.video_jepa.main ...` line, e.g. `data.batch_size=256`, `optim.lr=5e-4`,
`optim.epochs=100`. Walltimes are generous (the model is tiny: `ResNet5` henc=32
+ `ResUNet` on 64×64 frames) and within each partition's max (debug 15 min,
gpua/gpuv 24 h).

## Troubleshooting

- **`torch.cuda.is_available()` is False / "no kernel image available"** — the
  node driver is too old for the CUDA 12.8 runtime (need ≥ 525.60.13). Check the
  `nvidia-smi` driver_version line printed at the top of the job log. Fallback:
  rebuild against `cu126` (edit the index URL in `eb_jepa-env.pyproject.toml`).
- **`uv sync` can't resolve a cp314 `cu128` wheel** — a given nightly day may be
  gapped. Retry, or pin `--python 3.13` in `build_env.sh`.
- **Job stalls at "Downloading"** — something tried to fetch over the network on
  an offline node. The dataset is the usual culprit: confirm
  `$EBJEPA_DSETS/mnist_test_seq.npy` exists and `EBJEPA_DSETS` is exported.
