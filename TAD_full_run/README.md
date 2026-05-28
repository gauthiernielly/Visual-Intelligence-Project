# TAD baseline on Toyota Smarthome Untrimmed

This part of our project trains an ActionFormer temporal action detector on the TSU
Cross-Subject split and evaluates it on the 86-video subset shared with the
VLM-only and Hybrid pipelines. It contains every script, config, output and
figure used in our analysis of the TAD baseline.

## What the pipeline does

The pipeline reads `data_cs_split.json` (the master TSU annotation file) and
runs eight steps end to end.

1. Build the Cross-Subject train/val/test split, with 10 training subjects
   (P03 to P19, P25 held out for validation) and 7 test subjects (P02, P10,
   P11, P14, P16, P18, P20). Resulting counts are 315 / 36 / 185 videos.
2. Extract CLIP ViT-B/32 features for every video, one frame every 0.64 s
   and one 512-dim vector per frame. About two hours, done once.
3. Clone and patch OpenTAD. The eager imports of the optional model families
   (TadTR, AFSD, VSGN, ViT/Swin/SlowFast backbones) are wrapped in try/except
   so missing CUDA extensions or mmcv do not break the ActionFormer code path.
4. Substitute the runtime paths into the two TSU configs and copy them into
   the OpenTAD config tree.
5. Train ActionFormer for 40 epochs with AdamW (lr 1e-4, weight decay 0.05,
   3 epochs of linear warmup followed by cosine annealing, gradient clipping
   at norm 1.0, AMP and EMA). Best checkpoint is selected by validation loss.
6. Run inference on the 185 test videos.
7. Post-process the raw predictions into the canonical JSON the Hybrid
   pipeline consumes (`outputs/predictions_canonical.json`), then apply the
   matched prefilter on the 86-video subset (deduplicate, score >= 0.15,
   per-class temporal NMS at IoU 0.3) and score the result with the shared
   `complete_eval.py`.
8. Render every figure used in the report.

## Headline results on the 86-video evaluation subset

| Metric | TAD-only |
|---|---:|
| LCS Recall    | 72.5 % |
| LCS Precision | 26.8 % |
| LCS F1        | 36.6 % |
| mAP @ IoU 0.1 | 22.07 % |
| mAP @ IoU 0.3 | 17.82 % |
| mAP @ IoU 0.5 | 13.05 % |

TAD has high recall and low precision by design, the detector is dense and
rarely misses a salient action but it floods the timeline with redundant
proposals. The Hybrid pipeline (LCS Recall 49.0, Precision 30.7, F1 36.2)
trades part of TAD's recall for higher precision, the F1 score is essentially
tied. Per-video results are in `outputs/tad_complete_eval_86.csv`.

## Folder layout

```
TAD_full_run/
├── README.md                                this file
├── submit_job.sh                            SLURM batch script
├── run_pipeline.sh                          end-to-end pipeline (called by SLURM)
├── prepare_env.sh                           one-time conda env setup
├── data_cs_split.json                       master annotations (all 536 videos)
├── tsu_full_<jobid>.out / .err              SLURM logs from the validated run
├── configs/
│   ├── tsu_features_clip_full.py            OpenTAD dataset config (templated)
│   └── tsu_clip_full.py                     ActionFormer training config
├── scripts/
│   ├── build_full_split.py                  step 1
│   ├── extract_clip_features.py             step 2
│   ├── apply_opentad_patches.py             step 3
│   ├── postprocess_predictions.py           step 7a
│   ├── verify_predictions.py                step 7a, schema check
│   ├── prefilter_tad_for_hybrid_eval.py     step 7b, matched prefilter
│   ├── complete_eval.py                     step 7c, shared evaluator
│   └── visualize_results.py                 step 8, all figures
├── outputs/
│   ├── tsu_cs_full.json                     split with the `frame` field
│   ├── category_idx.txt                     51-class map, alphabetical
│   ├── predictions_canonical.json           raw TAD output, canonical schema
│   ├── hybrid_eval_86.txt                   the 86 video ids
│   ├── tad_pipeline_results_86.json         after matched prefilter
│   ├── tad_complete_eval_86.csv             per-video LCS Recall/Precision/F1
│   ├── log.json                             full training log
│   ├── verify.txt                           output of verify_predictions.py
│   ├── figures/                             every figure used in the report
│   ├── exps/tsu_full/gpu2_id0/              OpenTAD checkpoints + raw JSON
│   └── features/clip_vitb32/                per-video CLIP feature .npy files
└── work/                                    OpenTAD checkout
```

## How to run

```bash
# 1. Copy this folder to the cluster
scp -r TAD_full_run/ <user>@izar.epfl.ch:~/tsu_full_run/

# 2. One-time conda env setup
ssh <user>@izar.epfl.ch
cd ~/tsu_full_run
bash prepare_env.sh

# 3. Submit the job. The pipeline expects videos at
#    /work/cs-503/sadgal/Videos_mp4/ and the cross-pipeline GT CSV folder at
#    /work/cs-503/sadgal/Annotation/. Override with DATASET_ROOT if needed.
sbatch submit_job.sh
```

## Re-running specific stages

Useful env vars on `sbatch submit_job.sh`:

| Variable | Effect |
|---|---|
| `SKIP_FEATURES=1` | skip CLIP feature extraction, reuse cached `.npy` files |
| `FEATURES_ONLY=1` | stop after CLIP feature extraction |
| `RESUME_CKPT=<path>` | resume training from a previous checkpoint |
| `DATASET_ROOT=<path>` | parent of `Videos_mp4` and `Annotation` if your dataset lives elsewhere |

## Evaluating a different system on the same 86 videos

`scripts/complete_eval.py` is the shared evaluator used by the TAD, VLM and
Hybrid pipelines, so any predictions JSON in the canonical schema can be
scored under the same protocol.

```bash
python scripts/complete_eval.py \
  --pred  <your_predictions.json> \
  --names YourSystem \
  --gt    /work/cs-503/sadgal/Annotation \
  --fps   25 \
  --out   <your_eval.csv>
```

Two systems can be compared in a single call by passing two `--pred`
arguments together with `--names`.

## Regenerating the showcase figures locally

The head-to-head bar chart and the two 3-row Gantts (GT vs TAD vs Hybrid)
require the Hybrid pipeline's predictions JSON. Once you have it, you can regenerate
them with:

```bash
python scripts/visualize_results.py \
  --log                outputs/log.json \
  --predictions        outputs/predictions_canonical.json \
  --annotations        outputs/tsu_cs_full.json \
  --out-dir            outputs/figures \
  --video-list         outputs/hybrid_eval_86.txt \
  --gt-dir             /work/cs-503/sadgal/Annotation \
  --tad-pipeline-json  outputs/tad_pipeline_results_86.json \
  --tad-eval-csv       outputs/tad_complete_eval_86.csv \
  --hybrid-pipeline-json <hybrid_predictions.json> \
  --hybrid-eval-csv      <hybrid_eval.csv> \
  --showcase-vids        P20T16C03 P14T02C04
```
