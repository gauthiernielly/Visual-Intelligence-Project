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
# Usage:
#   sbatch submit_job.sh                          # default: full pipeline
#   FEATURES_ONLY=1 sbatch submit_job.sh          # skip training, just extract features
#   SKIP_FEATURES=1 sbatch submit_job.sh          # skip extraction (assume features done)
#   RESUME_CKPT=<path> sbatch submit_job.sh       # resume from a previous training checkpoint
#
# To use 2 GPUs instead, change `--gres=gpu:1` above to `--gres=gpu:2` and halve --time.

cd "${SLURM_SUBMIT_DIR:-.}"

set -euo pipefail

echo "=================================================="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "DATE=$(date)"
echo "=================================================="

# Conda activation (Izar non-interactive shells need explicit hook)
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
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;8.0}"  # V100 = sm_70, A100 = sm_80

# Detect how many GPUs SLURM gave us; OpenTAD's torchrun needs nproc_per_node to match.
N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
export N_GPUS
echo "N_GPUS=$N_GPUS"

bash run_pipeline.sh
