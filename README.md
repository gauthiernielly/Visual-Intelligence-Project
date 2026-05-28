# Visual Intelligence Project — VLM Baseline

Zero-shot temporal activity detection on the **Toyota Smarthome Untrimmed (TSU)** dataset using a Vision-Language Model (Qwen3-VL-8B-Instruct). This repository contains the VLM baseline approach, one of three methods evaluated in this project alongside a Temporal Action Detection (TAD) model and a hybrid approach.

## Overview

The pipeline prompts a VLM with frames sampled from sliding windows over each video and asks it to output a JSON list of activity segments. No fine-tuning or temporal annotations are used during inference — the model relies entirely on its visual and language priors.

**Key design choices:**
- 60-second non-overlapping sliding windows, sampled at ~0.58 fps (≈35 frames/min)
- The model is given a closed-label vocabulary (51 TSU classes) and must output structured JSON
- Segments predicted across windows are merged per class before evaluation

## Results

Evaluated on **86 videos** (Cross-Subject test split, subjects P02/P10/P11/P16/P18/P20) — the same subset used by the TAD and hybrid approaches for direct comparability.

| Metric | Value |
|---|---|
| LCS Recall | 22.3% |
| LCS Precision | 59.5% |
| LCS F1 | 30.9% |
| mAP @ IoU 0.05 | 2.4% |

The LCS (Longest Common Subsequence) metrics are computed identically across all three project approaches and are directly comparable. The low mAP reflects the model's difficulty with precise temporal localisation, while the higher LCS precision indicates that predicted event labels are often correct — the main failure mode is over-segmentation and temporal boundary inaccuracy.

## Generation runs

| Run | Videos | Approx. wall time |
|---|---|---|
| Pilot (debug) | 50 | ~5h 25min |
| Full Cross-Subject test split | 185 | ~13h 20min |

Both runs used the same sliding-window configuration and were submitted as SLURM jobs on a single GPU node.

## Repository structure

```
VLM_Baseline/
├── config.py            # All hyperparameters and paths in one place
├── requirements.txt     # Python dependencies (excluding PyTorch)
├── setup_env.sh         # Creates the pure_vlm conda environment
├── submit_job.sh        # SLURM job script (generate / evaluate / all)
├── gen_indices.json     # Ordered list of video IDs to generate predictions for
├── eval_indices.json    # Subset of 86 video IDs used for evaluation
├── src/
│   ├── generate.py      # Sliding-window VLM inference → outputs/generated_segments.json
│   ├── inference.py     # Window sampling, prompt construction, JSON parsing, segment merging
│   ├── evaluate.py      # Computes all metrics and saves outputs/metrics.json + figures
│   ├── metrics.py       # Event-mAP, LCS, per-class recall, duration recall, hallucination
│   └── graphs.py        # Matplotlib/Plotly figures (recall by duration, per-class, timeline)
├── Q&A.ipynb            # Probing notebook: questions and answers to diagnose temporal reasoning failures
├── outputs/             # Generated predictions and metrics
└── logs/                # SLURM stdout/stderr
```

## Setup

### Requirements

- NVIDIA GPU with ≥16 GB VRAM (the 8B model runs in float16)
- 64 GB system RAM recommended for large video batches
- Conda (Miniconda or Anaconda)
- CUDA 12.8 (adjust `CUDA_TAG` in `setup_env.sh` for other versions)

### Environment

```bash
cd VLM_Baseline
bash setup_env.sh
```

This creates a conda environment named `pure_vlm` with Python 3.10, PyTorch 2.10 (CUDA 12.8), and all dependencies from `requirements.txt`. Run once before the first job submission.

### Data paths

Edit [VLM_Baseline/config.py](VLM_Baseline/config.py) to point to your data:

```python
VIDEO_DIR = "/path/to/Videos_mp4"   # directory of .mp4 files
GT_DIR    = "/path/to/Annotation"   # per-subject subdirectories with CSV annotations
HF_HOME   = "/path/to/hf_cache"     # Hugging Face model cache
```

The annotation directory is expected to follow the structure `<GT_DIR>/<subject_id>/<video_id>.csv`, e.g. `Annotation/P10/P10T03C04.csv`.

## Running the pipeline

```bash
cd VLM_Baseline

# Generate predictions for all videos in gen_indices.json
sbatch submit_job.sh generate

# Evaluate the generated predictions on the eval_indices subset
sbatch submit_job.sh evaluate --eval_indices eval_indices.json

# Run generation then evaluation in a single job
sbatch submit_job.sh all
```


### Generation options

| Flag | Default | Description |
|---|---|---|
| `--video_dir` | `config.VIDEO_DIR` | Directory containing `.mp4` files |
| `--output_dir` | `outputs/` | Where to write `generated_segments.json` |
| `--gen_indices` | `gen_indices.json` | JSON list of video IDs to process (processes all if omitted) |
| `--limit N` | — | Process only the first N videos (debugging) |
| `--no-resume` | — | Reprocess videos already present in `generated_segments.json` |
| `--window_sec` | `60` | Length of each sliding window in seconds |
| `--overlap_sec` | `0` | Overlap between consecutive windows |
| `--window_fps` | `0.583` | Sampling rate inside each window (~35 frames/min) |

### Evaluation options

| Flag | Default | Description |
|---|---|---|
| `--annotation_dir` | `config.GT_DIR` | Root directory of CSV annotations |
| `--output_dir` | `outputs/` | Where to read `generated_segments.json` and write results |
| `--eval_indices` | — | JSON list of video IDs to evaluate (all predictions if omitted) |
| `--iou_thresholds` | `0.05 0.1 0.3` | IoU thresholds for mAP computation |

## Outputs

After running evaluate.py the following files are written to `outputs/`:

| File | Description |
|---|---|
| `generated_segments.json` | Raw predictions keyed by video ID |
| `metrics.json` | All computed metrics (mAP, LCS, per-class, duration, hallucination) |
| `lcs_per_video.json` | Per-video LCS recall / precision / F1 |
| `recall_by_duration.png` | Bar chart of detection recall broken down by event duration |
| `per_class_recall.png` | Per-class recall chart |
| `timeline_viewer.html` | Interactive Plotly viewer comparing predicted vs GT timelines |

## Probing temporal reasoning (Q&A notebook)

[VLM_Baseline/Q&A.ipynb](VLM_Baseline/Q&A.ipynb) is a diagnostic notebook that generates targeted questions and answers to identify where the model's temporal reasoning breaks down — for example, whether it confuses event ordering, misses short events, or hallucinates plausible-sounding but absent activities.


## Common issues

**CUDA out of memory**
Reduce `--window_fps` to sample fewer frames per window, or increase the SLURM `--mem` allocation. The 8B model in float16 requires roughly 16 GB of VRAM; the remaining memory budget is consumed by the input frames.

**JSON parse failures during generation**
Occasional failures (logged as `[parse] no JSON found`) are expected — the model sometimes produces malformed output. Failed windows contribute zero segments. The overall failure rate can be monitored in the SLURM log.