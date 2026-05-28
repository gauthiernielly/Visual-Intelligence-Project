#!/bin/bash
#SBATCH --job-name=tsu_full
#SBATCH --time=18:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=tsu_full_%j.out
#SBATCH --error=tsu_full_%j.err

# Full-TSU TAD training on SCITAS Izar.
#
# Default: full pipeline. Useful env-var overrides:
#   FEATURES_ONLY=1            stop after CLIP feature extraction
#   SKIP_FEATURES=1            skip extraction, assume features already on disk
#   RESUME_CKPT=<path>         resume training from a previous checkpoint
#   DATASET_ROOT=<path>        parent of Videos_mp4, default /work/cs-503/sadgal

cd "${SLURM_SUBMIT_DIR:-.}"

set -euo pipefail

echo "=================================================="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "DATE=$(date)"
echo "=================================================="

# Source conda. Izar's non-interactive shells need an explicit hook.
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook 2>/dev/null)" || {
    echo "ERROR: cannot source conda. Run prepare_env.sh first." >&2
    exit 1
  }
fi

conda activate tsu || {
  echo "ERROR: conda env 'tsu' not found. Run 'bash prepare_env.sh' first." >&2
  exit 1
}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;8.0}"  # V100=sm_70, A100=sm_80

# OpenTAD's torchrun needs nproc_per_node to match the GPU count SLURM granted.
N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
export N_GPUS
echo "N_GPUS=$N_GPUS"

bash run_pipeline.sh
