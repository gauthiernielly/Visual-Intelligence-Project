#!/bin/bash
# Create the pure_vlm conda environment and install all dependencies.
#
# Usage:
#   bash setup_env.sh
#
# Run once before the first: sbatch submit_job.sh

set -euo pipefail

CONDA_ENV="pure_vlm"
PYTHON_VERSION="3.10.20"
CUDA_TAG="cu128"

# ── Conda activation ─────────────────────────────────────────────────────────
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook 2>/dev/null)" || {
    echo "ERROR: conda not found. Install Miniconda first." >&2
    exit 1
  }
fi

# ── Create environment if needed ─────────────────────────────────────────────
if conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
  echo "Conda env '${CONDA_ENV}' already exists — skipping creation."
else
  echo "Creating conda env '${CONDA_ENV}' (Python ${PYTHON_VERSION}) ..."
  conda create -y -n "${CONDA_ENV}" python="${PYTHON_VERSION}"
  echo "Env created."
fi

conda activate "${CONDA_ENV}"

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# ── Install PyTorch with the correct CUDA wheels ─────────────────────────────
# Done first so the subsequent pip install -r does not pull a CPU-only build.
pip install --upgrade pip
echo "Installing PyTorch (CUDA tag: ${CUDA_TAG}) ..."
pip install \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  torch==2.10.0 torchvision==0.25.0

# ── Install remaining dependencies ───────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing remaining dependencies from requirements.txt ..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ── Rename Jupyter kernel for notebooks usage ────────────────────────────────
python -m ipykernel install --user \ 
  --name pure_vlm 
  --display-name " VLM baseline kernel (pure_vlm)"

echo ""
echo "Setup complete. Activate with:  conda activate ${CONDA_ENV}"
