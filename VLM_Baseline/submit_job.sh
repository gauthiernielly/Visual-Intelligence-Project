#!/bin/bash
#SBATCH --job-name=qwen3vl_tsu_probe
#SBATCH --time=05:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=qwen3vl_tsu_probe_%j.out
#SBATCH --error=qwen3vl_tsu_probe_%j.err

# Batch-submit Qwen3VL probe training on SCITAS (Izar).
#
# Usage:
#   sbatch submit_job.sh <huggingface_hub_token>

cd "${SLURM_SUBMIT_DIR:-.}"

################# Hugging Face / cache setup #####################

# Use local per-user writable caches.
# Training data should now come from /work/cs-503/prepared/the_cauldron
# via load_from_disk(...), so no shared read-only HF cache is needed here.

CACHE_ROOT="${SCRATCH:-$HOME}/qwen3vl_tsu_probe_hf_cache"

export HF_HOME="${CACHE_ROOT}/hf_home"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf_datasets_cache"
export TMPDIR="${CACHE_ROOT}/tmp"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_DISABLE_TELEMETRY=1

###############################################################

export PYTHONUNBUFFERED=1

# Non-interactive batch shells often lack conda hook; common miniconda layout on Izar:
# shellcheck disable=SC1091
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook 2>/dev/null)" || true
fi

conda activate viproject

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "TMPDIR=$TMPDIR"

OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 qwen3vl_tsu_probe.py
