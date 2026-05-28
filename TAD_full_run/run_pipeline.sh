#!/bin/bash
# End-to-end pipeline. Called by submit_job.sh inside the SLURM allocation.
#
# Env-var overrides:
#   FEATURES_ONLY=1   stop after step 2 (feature extraction)
#   SKIP_FEATURES=1   skip step 2 (assume features already on disk)
#   RESUME_CKPT=<p>   resume training from this checkpoint path
#   N_GPUS=<n>        auto-detected by submit_job.sh, falls back to 1

set -euo pipefail

ROOT="$(pwd)"
OUT="${ROOT}/outputs"
WORK="${ROOT}/work"
SCRIPTS="${ROOT}/scripts"
CFG="${ROOT}/configs"
mkdir -p "$OUT" "$WORK" "$OUT/figures"

# Inputs. Default is the SCITAS-Izar absolute path. Override with DATASET_ROOT
# if your dataset lives elsewhere. DATASET_ROOT must be the parent of Videos_mp4.
DATASET_ROOT="${DATASET_ROOT:-/work/cs-503/sadgal}"
VIDEOS_DIR="${DATASET_ROOT}/Videos_mp4"
DATA_CS_SPLIT="${ROOT}/data_cs_split.json"

if [[ ! -d "$VIDEOS_DIR" ]]; then
  echo "ERROR: videos directory not found: $VIDEOS_DIR" >&2
  echo "Override with: DATASET_ROOT=/path/containing/Videos_mp4 sbatch submit_job.sh" >&2
  exit 1
fi
if [[ ! -f "$DATA_CS_SPLIT" ]]; then
  echo "ERROR: data_cs_split.json missing in $(pwd)" >&2
  echo "Copy it from the project root before running." >&2
  exit 1
fi

# Outputs we produce.
SPLIT_JSON="${OUT}/tsu_cs_full.json"
CAT_TXT="${OUT}/category_idx.txt"
FEAT_DIR="${OUT}/features/clip_vitb32"
EXP_DIR="${OUT}/exps/tsu_full"
RAW_PRED="${EXP_DIR}/gpu1_id0/result_detection.json"     # OpenTAD writes here when save_dict=True
CANON_PRED="${OUT}/predictions_canonical.json"

mkdir -p "$FEAT_DIR" "$EXP_DIR"

# Step 1: build the train/val/test split annotation file.
echo
echo "=== [1/8] Building full split annotation =================="
python "${SCRIPTS}/build_full_split.py" \
  --input  "$DATA_CS_SPLIT" \
  --output "$SPLIT_JSON" \
  --cat-out "$CAT_TXT" \
  --val-subjects P25 \
  --fps 25.0

# Step 2: extract CLIP features for every video in the split. Resumable.
if [[ "${SKIP_FEATURES:-0}" == "1" ]]; then
  echo
  echo "=== [2/8] Skipping feature extraction (SKIP_FEATURES=1) ==="
else
  echo
  echo "=== [2/8] CLIP feature extraction =========================="
  python "${SCRIPTS}/extract_clip_features.py" \
    --video-dir   "$VIDEOS_DIR" \
    --output-dir  "$FEAT_DIR" \
    --ann-file    "$SPLIT_JSON" \
    --model       ViT-B-32 \
    --pretrained  laion2b_s34b_b79k \
    --fps 25 --feat-stride 16 \
    --batch-size 64 --device cuda
fi

if [[ "${FEATURES_ONLY:-0}" == "1" ]]; then
  echo "FEATURES_ONLY=1, stopping after extraction."
  exit 0
fi

# Step 3: clone OpenTAD, build its CUDA extensions, then patch eager imports
# so missing extensions or mmcv do not break the ActionFormer code path.
OPENTAD_DIR="${WORK}/OpenTAD"
echo
echo "=== [3/8] Setting up OpenTAD =============================="

# torch.utils.cpp_extension imports pkg_resources, which setuptools 80 removed.
# Pin setuptools below 70 and make sure wheel is present before any build.
pip install -q "setuptools<70" wheel

if [[ ! -d "${OPENTAD_DIR}" ]]; then
  git clone https://github.com/sming256/OpenTAD.git "${OPENTAD_DIR}"
fi

# Build the three CUDA extensions one at a time. --no-build-isolation is
# needed so the build subprocess can import torch from this conda env.
# Align1D may fail on the Izar CUDA arch, the patches that follow route
# around its absence. NMS is on the ActionFormer code path and must succeed.
for ext in \
  opentad/models/utils/post_processing/nms \
  opentad/models/roi_heads/roi_extractors/align1d \
  opentad/models/roi_heads/roi_extractors/boundary_pooling
do
  ( cd "${OPENTAD_DIR}/${ext}" && pip install -q --no-build-isolation . ) || \
    echo "  WARNING: build failed for ${ext}, patches will route around it."
done

python "${SCRIPTS}/apply_opentad_patches.py" --opentad "${OPENTAD_DIR}"

# Make OpenTAD importable for post-processing scripts that import from opentad.
export PYTHONPATH="${OPENTAD_DIR}:${PYTHONPATH:-}"

# Step 4: copy our TSU configs into the OpenTAD config tree, expanding the
# runtime path placeholders along the way.
echo
echo "=== [4/8] Writing TSU configs into OpenTAD ================"
TSU_BASE_DIR="${OPENTAD_DIR}/configs/_base_/datasets/tsu"
mkdir -p "$TSU_BASE_DIR"

# Substitute the path placeholders in the dataset config template.
SPLIT_JSON_ESC="${SPLIT_JSON//\//\\/}"
CAT_TXT_ESC="${CAT_TXT//\//\\/}"
FEAT_DIR_ESC="${FEAT_DIR//\//\\/}"
EXP_DIR_ESC="${EXP_DIR//\//\\/}"

sed \
  -e "s|@@ANN_FILE@@|${SPLIT_JSON}|g" \
  -e "s|@@CLASS_MAP@@|${CAT_TXT}|g" \
  -e "s|@@DATA_PATH@@|${FEAT_DIR}/|g" \
  "${CFG}/tsu_features_clip_full.py" > "${TSU_BASE_DIR}/features_clip_full.py"

sed \
  -e "s|@@WORK_DIR@@|${EXP_DIR}|g" \
  "${CFG}/tsu_clip_full.py" > "${OPENTAD_DIR}/configs/actionformer/tsu_clip_full.py"

# Sanity-parse via mmengine to catch syntax errors before training starts.
python -c "
from mmengine.config import Config
c = Config.fromfile('${OPENTAD_DIR}/configs/actionformer/tsu_clip_full.py')
print('  config parsed OK, num_classes=', c.model.rpn_head.num_classes,
      'in_channels=', c.model.projection.in_channels,
      'work_dir=', c.work_dir,
      'end_epoch=', c.workflow.end_epoch)
"

# Step 5: train ActionFormer.
echo
echo "=== [5/8] Training ActionFormer ============================"
NPP="${N_GPUS:-1}"
TRAIN_ARGS=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
  TRAIN_ARGS+=(--resume "${RESUME_CKPT}")
  echo "  resuming from: ${RESUME_CKPT}"
fi

(
  cd "${OPENTAD_DIR}"
  torchrun --nnodes=1 --nproc_per_node="${NPP}" \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py configs/actionformer/tsu_clip_full.py "${TRAIN_ARGS[@]}"
)

# Step 6: run inference on the test split.
echo
echo "=== [6/8] Inference ========================================"
BEST_CKPT=$(ls -t "${EXP_DIR}"/*/checkpoint/best.pth 2>/dev/null | head -1 || true)
if [[ -z "${BEST_CKPT}" ]]; then
  BEST_CKPT=$(ls -t "${EXP_DIR}"/*/checkpoint/epoch_*.pth | head -1)
fi
echo "  using checkpoint: ${BEST_CKPT}"

(
  cd "${OPENTAD_DIR}"
  # Same nproc_per_node as training so test.batch_size=2 stays valid.
  torchrun --nnodes=1 --nproc_per_node="${NPP}" \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/test.py configs/actionformer/tsu_clip_full.py \
    --checkpoint "${BEST_CKPT}" \
    --cfg-options evaluation.subset=testing
)

# With save_dict=True, OpenTAD writes the predictions JSON at work_dir/result_detection.json.
RAW_PRED_FOUND=$(find "${EXP_DIR}" -name "result_detection.json" | head -1)
echo "  raw predictions JSON: ${RAW_PRED_FOUND}"

# Step 7: post-process the raw predictions into the canonical JSON.
echo
echo "=== [7/8] Post-processing predictions ====================="
python "${SCRIPTS}/postprocess_predictions.py" \
  --raw     "${RAW_PRED_FOUND}" \
  --classes "${CAT_TXT}" \
  --output  "${CANON_PRED}" \
  --min-duration 0.1 \
  --min-score 0.05

python "${SCRIPTS}/verify_predictions.py" \
  --predictions  "${CANON_PRED}" \
  --ground-truth "${SPLIT_JSON}" \
  --num-classes 51 | tee "${OUT}/verify.txt"

# Step 7b: apply the matched prefilter on the 86-video evaluation subset.
VIDEO_LIST="${OUT}/hybrid_eval_86.txt"
TAD_86_JSON="${OUT}/tad_pipeline_results_86.json"
TAD_86_CSV="${OUT}/tad_complete_eval_86.csv"
GT_DIR="${DATASET_ROOT}/Annotation"

if [[ -f "${VIDEO_LIST}" ]]; then
  python "${SCRIPTS}/prefilter_tad_for_hybrid_eval.py" \
    --predictions "${CANON_PRED}" \
    --video-list  "${VIDEO_LIST}" \
    --output      "${TAD_86_JSON}"

  # Step 7c: run the shared evaluator to get the per-video LCS metrics CSV.
  if [[ -d "${GT_DIR}" ]]; then
    python "${SCRIPTS}/complete_eval.py" \
      --pred  "${TAD_86_JSON}" \
      --names TAD \
      --gt    "${GT_DIR}" \
      --fps   25 \
      --out   "${TAD_86_CSV}"
  else
    echo "  GT_DIR ${GT_DIR} not found, skipping complete_eval.py."
  fi
else
  echo "  ${VIDEO_LIST} missing, skipping prefilter and eval."
fi

# Step 8: render the figures the document uses. The showcase bars and 3-row
# Gantts also need the hybrid-pipeline JSON, pass --hybrid-pipeline-json
# and --hybrid-eval-csv on re-run when those are available.
echo
echo "=== [8/8] Generating figures =============================="
LOG_FILE=$(find "${EXP_DIR}" -name "log.json" | head -1)
VIS_ARGS=(
  --log         "${LOG_FILE}"
  --predictions "${CANON_PRED}"
  --annotations "${SPLIT_JSON}"
  --out-dir     "${OUT}/figures"
)
[[ -f "${VIDEO_LIST}" ]] && VIS_ARGS+=(--video-list "${VIDEO_LIST}")
[[ -d "${GT_DIR}" ]]      && VIS_ARGS+=(--gt-dir "${GT_DIR}")
[[ -f "${TAD_86_JSON}" ]] && VIS_ARGS+=(--tad-pipeline-json "${TAD_86_JSON}")
[[ -f "${TAD_86_CSV}" ]]  && VIS_ARGS+=(--tad-eval-csv "${TAD_86_CSV}")
python "${SCRIPTS}/visualize_results.py" "${VIS_ARGS[@]}"

# Copy the training log into outputs/ for easy retrieval.
cp "${LOG_FILE}" "${OUT}/log.json"

echo
echo "=================================================="
echo " DONE."
echo " outputs/         : $(ls -la ${OUT} | wc -l) entries"
echo " figures/         : $(ls -1 ${OUT}/figures 2>/dev/null | wc -l) PNGs"
echo " predictions_canonical.json : $(stat -c%s ${CANON_PRED} 2>/dev/null) bytes"
echo "=================================================="
