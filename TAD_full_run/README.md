# TSU full-dataset training on SCITAS Izar

Production-ready bundle to run the full-TSU TAD training that the smoke notebook validated. Produces the same artifacts as the smoke run (training curves, predictions analysis, Gantt overlays, canonical predictions JSON), but on all 351 training / ~30 validation / 185 test videos for 40 epochs.

## Folder layout

```
cluster_run/
├── README.md                       # this file
├── submit_job.sh                   # SLURM batch script
├── run_pipeline.sh                 # the actual end-to-end pipeline (called by SLURM)
├── prepare_env.sh                  # one-time conda env setup
├── data_cs_split.json              # MUST be copied alongside before running
├── configs/
│   ├── tsu_features_clip_full.py   # OpenTAD dataset config (paths via env vars)
│   └── tsu_clip_full.py            # ActionFormer training config (40 epochs)
└── scripts/
    ├── build_full_split.py         # step 1: create tsu_cs_full.json with train/val/test
    ├── extract_clip_features.py    # step 2: CLIP ViT-B/32 features (resumable)
    ├── apply_opentad_patches.py    # step 3: patch eager imports for missing extensions
    ├── postprocess_predictions.py  # step 6: augment predictions to canonical schema
    ├── visualize_results.py        # step 7: produce all the figures
    └── verify_predictions.py       # step 8: schema check
```

## What you need to do

### 1. Copy this folder to the cluster

```bash
scp -r cluster_run/ <user>@izar.epfl.ch:~/tsu_full_run/
```

(Or `rsync -avz cluster_run/ izar:~/tsu_full_run/`.)

### 2. Create the conda env (one-time)

On Izar:

```bash
cd ~/tsu_full_run
bash prepare_env.sh
```

This creates a `tsu` conda env with PyTorch 2.x + CUDA, OpenTAD's runtime deps, `open_clip_torch`, and `decord`. ~5 minutes.

### 3. Submit the job

```bash
sbatch submit_job.sh
```

The pipeline expects the videos at `/work/cs-503/sadgal/Videos_mp4/`. If your dataset is elsewhere, override:

```bash
DATASET_ROOT=/some/other/path sbatch submit_job.sh
```

(`DATASET_ROOT` should point at the *parent* of `Videos_mp4/`, not at `Videos_mp4/` itself.)

The submit script writes logs to `tsu_full_<jobid>.out` and `tsu_full_<jobid>.err` in the submission directory.

### 4. Watch progress

```bash
squeue -u $USER                     # is it running?
tail -f tsu_full_*.out              # live training log
ls -la outputs/                      # artifacts as they appear
```

## Resource configuration

The submit script defaults to:

```
--gres=gpu:1            # 1 GPU (V100 or A100)
--time=18:00:00         # 18 h wall clock
--mem=32G
--cpus-per-task=8
```

Approximate breakdown on a single V100:

| Step | Wall clock |
|---|---|
| CLIP feature extraction (~566 videos) | 30–60 min |
| OpenTAD install + patches | ~10 min |
| Training (40 epochs, ~351 videos, batch=2) | 7–10 h |
| Inference | ~10 min |
| Post-processing + figures | ~1 min |

On 2 V100s with `--gres=gpu:2`, training time roughly halves. To switch, edit `submit_job.sh` (one line) and re-submit.

## Expected outputs

After a successful run, `outputs/` will contain:

- `tsu_cs_full.json` — annotation file with the train/val/test split
- `category_idx.txt` — 51-class map (alphabetical order)
- `features/clip_vitb32/*.npy` — per-video CLIP features (kept on `$WORK/.../features` for reuse across reruns)
- `exps/tsu_full/gpu1_id0/checkpoint/best.pth` — trained checkpoint
- `exps/tsu_full/gpu1_id0/result_detection.json` — raw OpenTAD output
- `predictions_canonical.json` — the schema Person 4 consumes
- `figures/training_curves.png` — train/val loss + val mAP across epochs
- `figures/predictions_analysis.png` — score / per-class / duration distributions
- `figures/gantt_overlay_<vid>.png` — one Gantt per test video (subsample by default)
- `log.json` — full text training log
- `verify.txt` — output of the schema verifier

## How this differs from the smoke notebook

| | Smoke (Colab) | Full (Izar) |
|---|---|---|
| Train videos | 4 hand-picked | 321 (all P03–P19, P25 held out) |
| Val videos | 1 | ~30 (P25, held-out subject) |
| Test videos | 2 | 185 (full CS test split) |
| Epochs | 5 | 40 |
| Gradient steps | ~5 | ~6500 |
| Wall clock | ~5 min | 7–10 h |
| Goal | "pipeline works" | "produce comparable mAP" |

## Resumability

- **Feature extraction** is per-video and skips files that already exist. If the job dies after extracting 200/566 videos, just re-submit; it picks up where it left off.
- **OpenTAD install** detects an existing checkout and skips the clone.
- **Configs** are regenerated on every run (cheap) so paths are always consistent.
- **Training** does NOT auto-resume from the last checkpoint by default. To resume an interrupted training run, set `RESUME_CKPT=<path>` in the environment before re-submitting and `run_pipeline.sh` will pass `--resume <path>` to OpenTAD's training script.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: Align1D` | Patches didn't apply | re-run `scripts/apply_opentad_patches.py` manually |
| `numpy` version error | OpenTAD pins numpy 1.23.5 but conda has newer | run `conda activate tsu && pip install -q numpy==1.23.5` |
| Job dies at 12 h with training not done | Underestimated wall clock | re-submit with `--time=24:00:00` and `RESUME_CKPT=$(ls -t outputs/exps/tsu_full/*/checkpoint/epoch_*.pth \| head -1)` |
| `decord` import error during feature extraction | decord not in env | `pip install decord` inside the `tsu` env |
| All-zero mAP after full training | Either extraction picked up wrong videos or annotations got mismatched | inspect `outputs/figures/predictions_analysis.png` — if predictions are <100 segments total, the model never trained; if they're 10000+ but still 0 mAP, evaluation subset may be misconfigured |

## After the run

The headline number to report is **Average-mAP on the test set** (from `outputs/exps/tsu_full/.../log.json`). Note that OpenTAD's `evaluation.subset` config makes test-time inference compare predictions against the validation GT; the `verify_predictions.py` step computes the correct test-set metric independently.
