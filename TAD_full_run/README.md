# TSU full-dataset ActionFormer TAD training on SCITAS Izar

## What this bundle is, in one paragraph

This is the cluster-side pipeline that produces our **TAD baseline** for the
hybrid TAD+VLM project. It takes the Toyota Smarthome Untrimmed (TSU) dataset,
trains an ActionFormer model on the full Cross-Subject (CS) split, and outputs
a JSON of detected segments. Each detection is a `(start, end, action label,
confidence)` tuple. The hybrid pipeline consumes this JSON as the structural
anchor for VLM reasoning. Everything you need to reproduce the run, plus the
actual outputs we shipped, is in this folder.

## What we actually did, end-to-end

In plain language, the pipeline does this:

1. **Reads the TSU annotations** (`data_cs_split.json`) and reorganises them
   into a Cross-Subject train / validation / test split. Train uses 10 subjects
   (P03 through P19), validation uses subject P25 (held out from train), and
   test uses the official 7 test subjects (P02, P10, P11, P14, P16, P18, P20).
   The resulting counts are 315 / 36 / 185 videos respectively.
2. **Extracts visual features** for every video using CLIP ViT-B/32. We sample
   one frame every 0.64 s and pass each frame through CLIP's image encoder to
   get a 512-dimensional vector. A 21-minute video produces around 2,000
   vectors and one `.npy` file. This compresses 30 GB of video into about 2 GB
   of features and is what lets us train a temporal model on a single GPU. It
   is a one-time cost of around two hours.
3. **Sets up OpenTAD**, which is a unified open-source framework for temporal
   action detection, and patches it to work on Izar (more on this below). We
   wrote no custom dataset class for TSU. Instead we treat TSU as a
   Multi-THUMOS-shaped dataset and reuse OpenTAD's `ThumosPaddingDataset` via
   two new config files.
4. **Trains ActionFormer** for 40 epochs on the CLIP features. ActionFormer is
   a transformer-based TAD model that, given a sequence of feature vectors,
   outputs a list of segments. Each segment has a start time, an end time, an
   action label, and a confidence score. With 2 GPUs, training takes around
   18 minutes.
5. **Runs inference** on the 185 test videos and saves the raw predictions.
6. **Post-processes** the raw predictions. This step adds an integer
   `label_id` per segment (the canonical schema agreed with the hybrid
   pipeline), drops degenerate segments shorter than 0.1 s, discards noise
   below score 0.05, and deduplicates exact-duplicate predictions that arise
   from DDP test-time inference. The output is `predictions_canonical.json`,
   which is the artifact the hybrid pipeline consumes.
7. **Verifies** the schema and **plots** training curves, score / class /
   duration distributions, and per-video Gantt overlays for sanity checks.

## Why the install needed patches (the short version)

OpenTAD ships with three CUDA extensions and several optional model families
(TadTR, AFSD, VSGN, plus ViT / Swin / SlowFast backbones). None of these is
on the ActionFormer code path, but OpenTAD eagerly imports all of them at
module load. On Izar's environment:

- `Align1D` and `boundary_pooling` (CUDA extensions used by TadTR, AFSD, GTAD,
  and VSGN) fail to build because the conda PyTorch ships runtime libs but
  not `nvcc`.
- `mmcv` / `mmaction` (used by ViT / Swin / SlowFast backbones) are not
  installed because we do not need them for feature-based ActionFormer.

Rather than fight the install, `scripts/apply_opentad_patches.py` wraps the
unused eager imports in `try` / `except` so missing models silently fall back
to `None`. ActionFormer's required dependency `nms_1d_cpu` (the post-processing
NMS) does build correctly. You will see `[opentad-patch] X unavailable: ...`
lines at startup. They are informational, not errors.

## Headline results from the validated run

### Standard TAD metric (event-mAP)

Test set, 185 videos, 14,303 GT segments:

| Metric | Value |
|---|---|
| **Average-mAP (event-mAP across IoU {0.3, 0.5, 0.7})** | **12.52 %** |
| mAP @ IoU 0.30 | **18.38 %** |
| mAP @ IoU 0.50 | 13.15 % |
| mAP @ IoU 0.70 | 6.03 % |

⚠️ **Event-mAP, not frame-mAP.** Prior TSU work (PDAN around 32.7 %, MS-Temba
around 38 %) reports frame-mAP, which is strictly easier. The two numbers are
not directly comparable. Always disclose the metric difference when
presenting.

### Cross-pipeline metric (shared `hybrid_complete_eval.py`)

To allow a direct, apples-to-apples comparison between the TAD-only,
VLM-only, and Hybrid pipelines, all three pipelines are now scored by the
**same** `scripts/hybrid_complete_eval.py` script. It builds a label sequence
from each pipeline's segments (sorted by start time, consecutive duplicates
collapsed), aligns it with the GT label sequence via Longest Common
Subsequence, and derives **Recall, Precision and F1** from the LCS length.

Because the hybrid pipeline prefilters TAD segments before invoking the VLM
(deduplicate, `score >= 0.15`, per-class temporal NMS at IoU 0.3), the TAD
JSON has to go through the **same** prefilter before we score it. That is
exactly what `scripts/prefilter_tad_for_hybrid_eval.py` does: it reads
`predictions_canonical.json`, applies the three-step filter, restricts to
the agreed video subset, and writes a JSON in the format
`hybrid_complete_eval.py` consumes (`outputs/tad_pipeline_results.json`).

Results on the **68-video** cross-pipeline subset (6 test subjects: P02,
P10, P11, P16, P18, P20), GT mean 77 actions / video, TAD mean 212 actions
/ video after matched prefilter:

| Metric | TAD-only |
|---|---|
| Average **Recall**    | **69.6 %** |
| Average **Precision** | **27.9 %** |
| Average **F1-Score**  | **37.1 %** |

Per-video numbers are in `outputs/tad_complete_evaluation_results.csv`. The
68-video ID list is in `outputs/hybrid_eval_68.txt`.

The 68-video Recall / Precision / F1 numbers are the ones directly comparable
to the VLM-only and hybrid pipelines, since all three pipelines now run on
the same 68 videos with the same script.

A standalone LCS-recall script (`scripts/tad_lcs_eval.py`) is also kept in
this bundle for historical comparison. It evaluates LCS recall only, with
flexible filtering, and is the script that produced the earlier
`outputs/tad_lcs_full_test.csv` (185 videos, 84.57 %) and
`outputs/tad_lcs_hybrid_eval_50.csv` (older 50-video subset, 82.01 %).

## Folder layout

```
TAD_full_run/
├── README.md                       # this file
├── HANDOFF.md                      # interface contract for the hybrid pipeline
├── submit_job.sh                   # SLURM batch script
├── run_pipeline.sh                 # the actual end-to-end pipeline (called by SLURM)
├── prepare_env.sh                  # one-time conda env setup
├── data_cs_split.json              # master annotations (all 536 videos)
├── tsu_full_<jobid>.out / .err     # SLURM logs from the validated run
├── configs/
│   ├── tsu_features_clip_full.py   # OpenTAD dataset config (templated, paths injected)
│   └── tsu_clip_full.py            # ActionFormer training config
├── scripts/
│   ├── build_full_split.py             # step 1: tsu_cs_full.json with train/val/test
│   ├── extract_clip_features.py        # step 2: CLIP ViT-B/32 features (resumable)
│   ├── apply_opentad_patches.py        # step 3: defensive patches for missing extensions
│   ├── postprocess_predictions.py      # step 7: augment predictions, dedup, write canonical schema
│   ├── visualize_results.py            # step 8: produce all the figures
│   ├── verify_predictions.py           # step 7b: schema check
│   ├── tad_lcs_eval.py                 # standalone LCS-recall eval (185 + older 50 subsets)
│   ├── prefilter_tad_for_hybrid_eval.py # apply hybrid prefilter to predictions_canonical.json
│   └── hybrid_complete_eval.py         # shared Recall/Precision/F1 eval used by all 3 pipelines
├── outputs/
│   ├── tsu_cs_full.json                # train/val/test split annotations with `frame` field
│   ├── category_idx.txt                # 51-class map, alphabetical, line N is label_id N
│   ├── predictions_canonical.json      # the canonical output the hybrid pipeline consumes
│   ├── tad_pipeline_results.json       # predictions_canonical.json after the matched prefilter
│   ├── hybrid_eval_50.txt              # older 50-video subset
│   ├── hybrid_eval_68.txt              # current 68-video subset shared with VLM + Hybrid
│   ├── tad_lcs_full_test.csv           # per-video LCS recall on the 185 test videos
│   ├── tad_lcs_hybrid_eval_50.csv      # per-video LCS recall on the 50-video subset
│   ├── tad_complete_evaluation_results.csv # per-video Recall/Precision/F1 on the 68-video subset
│   ├── log.json                        # full training log
│   ├── verify.txt                      # output of verify_predictions.py
│   ├── figures/                        # training_curves, predictions_analysis, gantt_overlays
│   ├── exps/tsu_full/gpu2_id0/         # OpenTAD checkpoints + raw result_detection.json
│   └── features/clip_vitb32/           # per-video CLIP feature .npy files (~5 MB each)
└── work/                               # OpenTAD checkout (re-cloned by run_pipeline.sh)
```

## Quick start (running the bundle)

### 1. Copy this folder to the cluster

```bash
scp -r TAD_full_run/ <user>@izar.epfl.ch:~/tsu_full_run/
```

### 2. Create the conda env (one-time)

```bash
ssh <user>@izar.epfl.ch
cd ~/tsu_full_run
bash prepare_env.sh
```

Creates a `tsu` conda env with PyTorch 2.1.0+cu118, OpenTAD's runtime deps,
`open_clip_torch`, and `decord`. Pinned versions of `setuptools<70` and `wheel`
prevent the `pkg_resources` failures hit during initial bring-up. About 5
minutes.

### 3. Submit the job

```bash
sbatch submit_job.sh
```

The pipeline expects videos at `/work/cs-503/sadgal/Videos_mp4/`. If your
dataset path differs:

```bash
DATASET_ROOT=/some/other/path sbatch submit_job.sh
```

(`DATASET_ROOT` should point at the *parent* of `Videos_mp4/`, not at
`Videos_mp4/` itself.)

`submit_job.sh` writes logs to `tsu_full_<jobid>.out` and `tsu_full_<jobid>.err`
in the submission directory.

### 4. Watch progress

```bash
squeue -u $USER                     # is it running?
tail -f tsu_full_*.out              # live training log
ls -la outputs/                      # artifacts as they appear
```

## Cross-pipeline evaluation

After training + inference, two evaluation paths are available, both starting
from `outputs/predictions_canonical.json`.

### 1. Shared Recall / Precision / F1 on the 68-video subset (preferred)

This is the path that produces the **numbers directly comparable to the
VLM-only and Hybrid pipelines**, since all three pipelines run the same
`scripts/hybrid_complete_eval.py` script on the same 68 videos. Two steps:

**Step 1.** Prefilter the TAD JSON exactly as the hybrid pipeline would
before sending segments to the VLM (deduplicate, `score >= 0.15`, per-class
temporal NMS at IoU 0.3), and restrict to the 68-video subset:

```bash
python scripts/prefilter_tad_for_hybrid_eval.py \
    --predictions outputs/predictions_canonical.json \
    --video-list  outputs/hybrid_eval_68.txt \
    --output      outputs/tad_pipeline_results.json
```

**Step 2.** Point `hybrid_complete_eval.py` at the prefiltered JSON. The
shared script reads `JSON_PATH` and `ANNOTATIONS_DIR` from constants at the
top of the file. Either edit those two lines, or invoke it via a tiny wrapper
that overrides them:

```bash
JSON_PATH=outputs/tad_pipeline_results.json \
ANNOTATIONS_DIR=/work/cs-503/sadgal/Annotation \
OUTPUT_CSV=outputs/tad_complete_evaluation_results.csv \
python scripts/hybrid_complete_eval.py
```

(`ANNOTATIONS_DIR` is the per-subject CSV folder used by all three pipelines.
On Izar the canonical path is `/work/cs-503/sadgal/Annotation`.)

### 2. Standalone LCS-recall on the full 185 test set (historical)

For the standalone LCS-recall view on the full test split, use
`scripts/tad_lcs_eval.py`. It reads GT directly from `data_cs_split.json`
(no dependency on the cluster CSV folder) and exposes `--min-score`,
`--topk`, `--nms-iou` filters.

```bash
python scripts/tad_lcs_eval.py \
    --predictions outputs/predictions_canonical.json \
    --annotations data_cs_split.json \
    --subset testing \
    --min-score 0.1 \
    --out-csv outputs/tad_lcs_full_test.csv
```

Both scripts accept any predictions JSON that follows the canonical schema
`{"results": {video_id: [{segment, label, score}]}}`, so the VLM-only and
hybrid teams can reuse them directly on their own outputs.

## Resource configuration

`submit_job.sh` defaults are conservative:

```
--gres=gpu:1            # 1 GPU (V100 or A100)
--time=18:00:00         # 18 h wall clock, generous safety margin
--mem=32G
--cpus-per-task=8
```

The validated run used **2 GPUs** (edit `--gres=gpu:2` in `submit_job.sh`),
which is recommended. `run_pipeline.sh` reads the GPU count automatically and
configures `torchrun --nproc_per_node` accordingly.

### Actual wall-clock from the validated run (2 V100s, job 2879405)

| Step | Wall clock |
|---|---|
| [1/8] Build split | <1 s |
| [2/8] CLIP feature extraction (536 videos) | **about 2 h 19 min** (one-time. Reuse via `SKIP_FEATURES=1`) |
| [3/8] OpenTAD install + patches | about 5 min |
| [4/8] Configs | <1 s |
| [5/8] **Training** (40 epochs × 156 batches × batch=2) | **about 18 min** on 2 GPUs |
| [6/8] Inference (185 test videos) | about 1 min |
| [7/8] Post-process + verify | <1 min |
| [8/8] Visualizations | <1 min |

Once features are cached on disk, **a full re-run takes about 30 minutes**.
The 18 h budget exists for safety margin and queue jitter, not because
anything genuinely takes that long.

(`<N>` is the number of GPUs used. The validated run used 2, so the path is
`gpu2_id0/`.)

## How this differs from the smoke test notebook

The smoke notebook was the proof of concept (4 train videos × 5 epochs on
Colab). This bundle is the production version (315 train videos × 40 epochs on
SCITAS).

The smoke notebook was the proof of concept (4 train videos × 5 epochs on
Colab). This bundle is the production version (315 train videos × 40 epochs on
SCITAS).

| | Smoke (Colab) | Full (Izar) |
|---|---|---|
| Train videos | 4 hand-picked | **315** (10 subjects: P03 through P19) |
| Val videos | 1 | **36** (subject P25, held out) |
| Test videos | 2 | **185** (full CS test split, 7 subjects) |
| Epochs | 5 | 40 |
| Gradient steps | around 5 | **around 12,480** (156 batches × 40 epochs / 2 GPUs) |
| Test mAP @ IoU 0.3 | 0 % (4-video toy) | **18.38 %** |
| Wall clock | around 5 min | around 30 min on 2 GPUs (plus 2 h once for feature extraction) |
| Goal | "pipeline works" | "produce a real, reportable baseline" |

## Resumability

- **Feature extraction** is per-video and skips files that already exist. If
  the job dies after extracting 200/566 videos, just re-submit. It picks up
  where it left off.
- **`SKIP_FEATURES=1`** env var bypasses step 2 entirely once features are on
  disk. Strongly recommended for any re-run after the first.
- **`FEATURES_ONLY=1`** env var stops after step 2 (one-shot extraction
  without committing to a training run).
- **`RESUME_CKPT=<path>`** env var resumes training from a saved checkpoint:
  ```bash
  RESUME_CKPT=outputs/exps/tsu_full/gpu2_id0/checkpoint/epoch_29.pth \
      sbatch submit_job.sh
  ```
- **OpenTAD install** detects an existing checkout and skips the clone.
  Extension build commands are re-run on every submission (idempotent).

## Troubleshooting (issues actually hit during bring-up)

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: videos directory not found` | Earlier README used a relative path that depended on submission dir | Now defaults to absolute `/work/cs-503/sadgal`. Override with `DATASET_ROOT=...` if needed. |
| `error: Invalid requirement: 'opentad/models/...'` | OpenTAD's `requirements.txt` mixes valid pip lines with bare CUDA-extension paths that pip ≥24 rejects | `prepare_env.sh` installs requirements one-by-one, skipping the path lines |
| `ModuleNotFoundError: pytorchvideo` / `imgaug` | Both packages fail to build on Python 3.12 | Not imported anywhere in OpenTAD's source. The `prepare_env.sh` script skips them |
| `ModuleNotFoundError: No module named 'torch'` during a CUDA-extension build | Modern pip's build isolation hides conda's torch from the build subprocess | `run_pipeline.sh` passes `pip install --no-build-isolation .` to extension subdirs |
| `ModuleNotFoundError: pkg_resources` during cpp_extension import | setuptools ≥80 removed `pkg_resources.packaging`, which PyTorch 2.1 still imports | `prepare_env.sh` pins `setuptools<70`. `run_pipeline.sh` defensively reinstalls before extension builds |
| `OSError: CUDA_HOME environment variable is not set` building Align1D / boundary_pooling | Conda's PyTorch ships runtime libs, not `nvcc`. Cluster CUDA toolkit is not auto-loaded. | These two extensions are NOT on the ActionFormer code path. The patches in `apply_opentad_patches.py` route around their absence. NMS (`nms_1d_cpu`) is a pure C++ extension and builds fine. |
| `ImportError: No module named 'mmcv'` during opentad import | MMCV / mmaction backbones eagerly imported by `opentad/models/backbones/__init__.py` | `apply_opentad_patches.py` wraps each heavy backbone import in try/except. We never use the ViT/Swin/SlowFast backbones. |
| `ModuleNotFoundError: 'Align1D'` / `'boundary_max_pooling_cuda'` during opentad import | OpenTAD eagerly imports TadTR, ROIAlignExtractor, GTADExtractor, AFSDRefineHead, AFSDCoarseHead, etc. | `apply_opentad_patches.py` wraps all of these. None is on the ActionFormer code path. Look for `[opentad-patch]` lines at startup. They are informational, not errors. |
| `AssertionError: batch size 1 should be divided by world size 2` | OpenTAD's dataloader builder requires `batch_size % world_size == 0`. With 2 GPUs, val/test batch=1 fails. | Configs set `val.batch_size=2` and `test.batch_size=2`. Each rank still loads 1 sample at a time. `run_pipeline.sh` launches `tools/test.py` with `nproc_per_node=$N_GPUS` so the same constraint holds at standalone test time. |
| Exact-duplicate predictions in `predictions_canonical.json` | DDP test-time inference with 2 GPUs: when 185 videos is not divisible by world size, `DistributedSampler` pads by repeating a video, and OpenTAD's gather merges by concatenation | `postprocess_predictions.py` deduplicates by `(start, end, label_id)` and keeps the highest score |
| `numpy` version error | OpenTAD pins `numpy==1.23.5` but conda may install newer | `prepare_env.sh` installs `numpy==1.23.5` explicitly. If hit on a stale env, run `pip install -q numpy==1.23.5` inside `tsu`. |
| `decord` import error during feature extraction | `decord` not in env | `pip install decord` inside the `tsu` env (already in `prepare_env.sh`). |
| Validation mAP = 0 % at every training-time eval | Single-subject (P25) validation has very high variance | Not a bug. See *Known quirks* below. Test mAP is the headline number. |
| Job dies at wall-clock with training not done | Should not happen at this scale (around 18 min training on 2 GPUs) | Re-submit with `--time=24:00:00` and `RESUME_CKPT=$(ls -t outputs/exps/tsu_full/*/checkpoint/epoch_*.pth \| head -1)` |

## Known quirks worth understanding

- **Validation mAP is 0 % at every in-training eval** even though val *loss*
  is decreasing in parallel and test mAP is non-zero (12.52 %). The val set is
  a single subject (P25) with around 3,000 GT instances. The mAP variance over
  one subject is high enough that the metric reads as 0 every time. The
  training loop selects the *best* checkpoint by val *loss*, not val mAP, so
  this does not break best-checkpoint tracking. To get a less noisy val
  signal, hold out 2 subjects:
  ```bash
  python scripts/build_full_split.py --val-subjects P25,P19 ...
  ```
- **Test-time mAP is computed against the testing subset.** Earlier OpenTAD
  defaults compared test predictions against the validation GT, which is a
  known logging pitfall. `run_pipeline.sh` passes
  `--cfg-options evaluation.subset=testing` to `tools/test.py`, which fixes
  this.
- **The `result_detection.json` schema differs from
  `predictions_canonical.json`.** OpenTAD writes `{segment, label, score}` per
  detection. The canonical schema the hybrid pipeline consumes adds `label_id`
  (integer). The `postprocess_predictions.py` script performs the conversion,
  deduplicates exact duplicates introduced by DDP test inference, and also
  drops segments with `duration < 0.1 s` and `score < 0.05`.

## After the run

The headline numbers to report are:

- **event-mAP on the test set** from `outputs/log.json`'s final `Test INFO` block (12.52 % avg, 18.38 % @ IoU 0.3). This is the standard TAD metric for literature comparison.
- **Recall / Precision / F1 on the 68-video cross-pipeline subset** from `outputs/tad_complete_evaluation_results.csv` (**69.6 % / 27.9 % / 37.1 %**). These are the numbers directly comparable to the VLM-only and hybrid pipelines, since all three pipelines run the same `scripts/hybrid_complete_eval.py` on the same 68 videos.

For internal use, hand `outputs/predictions_canonical.json` to the hybrid
pipeline. The schema is `{"results": {video_id: [{segment, label, label_id, score}]}}`
with segments sorted by start time and exact duplicates already removed.
