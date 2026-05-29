#!/bin/bash
#SBATCH --job-name=vlm_eval
#SBATCH --time=24:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/vlm_eval_%j.out
#SBATCH --error=logs/vlm_eval_%j.err

# Run the VLM generation or evaluation pipeline.
#
# Usage:
#   sbatch submit_job.sh generate <args for generate.py>   # generate predictions (default)
#   sbatch submit_job.sh evaluate <args for evaluate.py>   # evaluate generated_segments.json
#   sbatch submit_job.sh all <args for generate.py>
#                            <args for evaluate.py>        # generate then evaluate in one job
#
# All arguments after MODE are forwarded to generate.py and/or evaluate.py.

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "ARGS: $*"

# ── Conda activation ───────────────────────────────────────────────────────────
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

conda activate pure_vlm || {
  echo "ERROR: conda env 'pure_vlm' not found. Run 'bash setup_env.sh' first." >&2
  exit 1
}

# ── Mode dispatch ─────────────────────────────────────────────────────────────
MODE="${1:-all}"
shift || true   # remaining args forwarded to the selected script

case "$MODE" in
  generate)
    python src/generate.py "$@"
    ;;
  evaluate)
    python src/evaluate.py "$@"
    ;;
  all)
    python src/generate.py "$@"
    python src/evaluate.py "$@"
    ;;
  *)
    echo "ERROR: unknown mode '$MODE'. Use 'generate', 'evaluate', or 'all'." >&2
    exit 1
    ;;
esac
