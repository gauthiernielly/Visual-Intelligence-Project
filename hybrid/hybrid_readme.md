# Hybrid Pipeline

This folder contains the full implementation of the hybrid TAD + VLM pipeline for temporal action recognition on the TSU dataset.

## Overview

The hybrid pipeline combines two systems:
- **ActionFormer (TAD)** generates temporal proposals; candidate segments with predicted labels, timestamps, and confidence scores.
- **Qwen3-VL-8B (VLM)** verifies and refines each proposal by inspecting extracted video frames, confirming or overriding the TAD label.

The key motivation is that pure VLMs fail at temporal localization while pure TAD lacks semantic flexibility. By using TAD proposals as structural anchors, we restrict each VLM query to a short, well-localized clip rather than asking it to reason over a full 21-minute video.


## Folder Structure

```
hybrid/
├── pipeline/           # Core pipeline code
│   └── hybrid.py        # Main pipeline script (TAD → filter → VLM → output)
│
├── evaluation/         # Evaluation scripts
│   ├── plot_gantt.py
│   ├── plot_ratio_vs_f1.py 
│   ├── recall_graphs.py   # Full evaluation with per-class traceback + graphs
│   └── complete_metrics.py          # Unified evaluation with mAP + LCS + coverage metrics
│
├── graphs/             # Generated figures
│   ├── gantt.png                 # Action timeline: GT vs TAD vs Hybrid (sample video)
│   ├── recall_per_class.png      # Per-class LCS Recall bar chart
│   ├── recall_vs_length.png      # Recall vs video complexity scatter plot
│   └── ratio_vs_f1.png           # Pred/GT ratio vs F1 scatter (TAD vs Hybrid)
│
├── results/            # Output files
│   └── hybrid_pipeline_results.json   # Final predictions: {video_id: [{segment, label, score}]}
│
└── logs/               # SLURM job logs
```


## Pipeline Design

The pipeline runs in three stages:

### Stage 1 — Temporal Proposal Generation
ActionFormer processes each video using CLIP ViT-B/32 features (stride 16, one frame per 0.64s) and produces ranked candidate segments. Two filtering steps are applied:
- **Confidence threshold** (τ = 0.15): drop low-quality proposals
- **Per-class NMS** (IoU > 0.3): suppress overlapping segments of the same class, while preserving co-occurring activities

### Stage 2 — VLM Labeling
For each surviving proposal, 8 uniformly sampled frames are extracted and passed to Qwen3-VL-8B with a structured prompt grounding it to the specific time interval and 51-class TSU ontology. The TAD label is provided as a hint; the VLM confirms or overrides it. Responses are parsed with a fuzzy label resolver that falls back to the TAD prediction on failure.

### Stage 3 — Timeline Assembly
Labeled segments are sorted chronologically and a second round of per-class NMS is applied to remove duplicates introduced by VLM relabeling.

---

## Usage

### Run the pipeline
```bash
# Single machine
python pipeline/hybrid.py

# SLURM array job (parallel across video chunks)
sbatch --array=0-3 run_hybrid.sh
# Set NUM_CHUNKS=4 in the script to match array size
```

### Run evaluation
```bash
# LCS Recall / Precision / F1 with per-class breakdown and graphs
python evaluation/recall_graphs.py

# Full evaluation including mAP at IoU 0.1 / 0.3 / 0.5
python evaluation/compute_metrics.py \
    --pred  results/hybrid_pipeline_results.json \
    --gt    ../../../../work/cs-503/sadgal/Annotation \
    --out   hybrid_eval.csv

# Compare TAD baseline vs Hybrid side by side
python evaluation/complete_eval.py \
    --pred  ../TAD_full_run/outputs/predictions_canonical.json \
            results/hybrid_pipeline_results.json \
    --names TAD Hybrid \
    --gt    ../../../../work/cs-503/sadgal/Annotation \
    --out   results/comparison.csv
```

### Generate figures
```bash
# Gantt timeline (auto-selects most readable video)
python evaluation/plot_gantt.py \
    --tad    ../TAD_full_run/outputs/predictions_canonical.json \
    --hybrid results/hybrid_pipeline_results.json \
    --gt     ../../../../work/cs-503/sadgal/Annotation \
    --auto \
    --out    graphs/gantt.png

# Pred/GT ratio vs F1 scatter
python evaluation/plot_ratio_vs_f1.py \
    --tad    ../TAD_full_run/outputs/predictions_canonical.json \
    --hybrid results/hybrid_pipeline_results.json \
    --gt     ../../../../work/cs-503/sadgal/Annotation \
    --out    graphs/ratio_vs_f1.png
```

---

## Results

Evaluated on 86 videos across 6 subjects (P02, P10, P11, P14, P16, P18, P20), 7 camera angles, and 17 task scenarios.

| Metric              | TAD Baseline | Hybrid Pipeline |
|---------------------|-------------|-----------------|
| Avg LCS Recall      | 69.6%       | 50.1%           |
| Avg LCS Precision   | 27.9%       | 32.4%           |
| Avg LCS F1          | 37.1%       | 36.9%           |
| mAP @ IoU 0.5       | —           | 5.9%            |
| Pred / GT ratio     | ~4×         | ~2×             |

**Key finding:** TAD and Hybrid achieve near-identical F1 scores, but through opposite trade-offs. TAD maximises recall through heavy over-prediction; Hybrid improves precision by filtering and relabeling, at the cost of coverage. VLM relabeling does not degrade overall performance but redistributes it.

---

## Dependencies

```
torch
transformers
qwen-vl-utils
opencv-python
Pillow
pandas
numpy
matplotlib
seaborn
```

Install with:
```bash
pip install torch transformers qwen-vl-utils opencv-python Pillow pandas numpy matplotlib seaborn
```


