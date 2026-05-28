#!/bin/bash
# One-time conda env setup for the TSU full-dataset run on Izar.
# Run this once on the cluster before sbatching submit_job.sh.

set -euo pipefail

# Source the conda hook so `conda activate` works in non-interactive shells.
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found. Install miniconda3 or anaconda3 in your home." >&2
  exit 1
fi

ENV_NAME="${ENV_NAME:-tsu}"

if conda env list | grep -q "^${ENV_NAME}\s"; then
  echo "Env '${ENV_NAME}' already exists. Activating."
else
  echo "Creating env '${ENV_NAME}' (Python 3.10)..."
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

echo "Installing PyTorch (CUDA 11.8) ..."
pip install -q torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# OpenTAD pulls in pytorchvideo and imgaug, which both fail to build on
# Python 3.12 and are not actually imported by the ActionFormer code path.
# We skip them and install the rest of the runtime deps explicitly.
# setuptools is pinned below 70 because PyTorch 2.1's cpp_extension.py still
# imports pkg_resources, removed in setuptools 80.
echo "Installing OpenTAD runtime deps..."
pip install -q \
  mmengine \
  wandb \
  scipy \
  einops \
  pandas \
  tqdm \
  ninja \
  numpy==1.23.5 \
  gdown==5.1.0 \
  "setuptools<70" \
  wheel

echo "Installing feature-extraction deps..."
pip install -q open_clip_torch decord pillow matplotlib

echo
echo "Done. Activate with: conda activate ${ENV_NAME}"
echo "Then: sbatch submit_job.sh"
