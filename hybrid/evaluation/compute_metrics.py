"""
Metrics computed
  1. LCS Recall, Precision, F1   (sequence-level, density-mismatch-robust)
  2. Segment-level mAP            (at IoU thresholds 0.1, 0.3, 0.5)
  3. Per-class AP                 (at IoU 0.5)
  4. Coverage & over-prediction   (pred/gt ratio, unique class coverage)
  5. Per-subject breakdown        (LCS F1 grouped by subject ID prefix)

Usage
  python complete_eval.py \
      --pred  ../results/hybrid_pipeline_results.json \
      --gt   ../../../../../work/cs-503/sadgal/Annotation \
      --fps   25 \
      --out   ../eval/hybrid_evaluation_results.csv
"""

import os
import sys
import json
import argparse
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_IOU_THRESHOLDS = [0.1, 0.3, 0.5]
DEFAULT_FPS = 25.0

TSU_CLASSES = [
    "Walk", "Take_something_off_table", "Put_something_on_table", "Drink.From_cup",
    "Get_up", "Sit_down", "Read", "Watch_TV", "Enter", "Use_Drawer", "Leave",
    "Breakfast.Eat_at_table", "Cook.Stir", "Use_cupboard", "Write", "Use_laptop",
    "Use_telephone", "Clean_dishes.Dry_up", "Take_pills", "Drink.From_bottle",
    "Eat_snack", "Clean_dishes", "Drink.From_can", "Use_glasses", "Pour.From_bottle",
    "Cook.Use_oven", "Dump_in_trash", "Breakfast.Cut_bread", "Use_tablet", "Use_fridge",
    "Cook.Cut", "Wipe_table", "Lay_down", "Cook.Use_stove", "Cook",
    "Clean_dishes.Clean_with_water", "Pour.From_kettle", "Breakfast.Spread_jam_or_butter",
    "Insert_tea_bag", "Get_water", "Clean_dishes.Put_something_in_sink",
    "Make_coffee.Pour_water", "Make_coffee", "Drink.From_glass", "Pour.From_can",
    "Make_coffee.Pour_grains", "Breakfast", "Make_tea", "Make_tea.Boil_water",
    "Stir_coffee_tea", "Breakfast.Take_ham",
]


# Label normalisation
def normalise(label: str) -> str:
    return str(label).strip().lower().replace(".", "_").replace(" ", "_")

_NORM_TO_CANONICAL = {normalise(c): c for c in TSU_CLASSES}

def canonical(label: str) -> str:
    return _NORM_TO_CANONICAL.get(normalise(label), normalise(label))



def load_gt_from_csv_dir(
    annotations_dir: str,
    video_ids: List[str],
    fps: float = DEFAULT_FPS,
) -> Dict[str, List[dict]]:
    """
    Load GT from per-video CSVs.
    Expected columns: start_frame, end_frame (or end_frame derived from duration), event.
    Returns {video_id: [{start, end, label}]}
    """
    gt: Dict[str, List[dict]] = {}

    for video_id in video_ids:
        subject_id = video_id[:3]
        csv_path = os.path.join(annotations_dir, subject_id, f"{video_id}.csv")

        if not os.path.exists(csv_path):
            warnings.warn(f"GT CSV not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            if df.empty or "event" not in df.columns:
                continue

            if "start_time" in df.columns and "end_time" in df.columns:
                df = df.rename(columns={"start_time": "start", "end_time": "end"})
            elif "start_frame" in df.columns and "end_frame" in df.columns:
                _fps = df["fps"].iloc[0] if "fps" in df.columns else fps
                df["start"] = df["start_frame"] / _fps
                df["end"]   = df["end_frame"]   / _fps
            else:
                warnings.warn(f"Cannot derive timestamps for {video_id}, skipping.")
                continue

            df = df.sort_values("start")
            gt[video_id] = [
                {"start": row["start"], "end": row["end"], "label": canonical(row["event"])}
                for _, row in df.iterrows()
            ]
        except Exception as e:
            warnings.warn(f"Failed to load GT for {video_id}: {e}")

    return gt


def load_gt_from_json(gt_json_path: str) -> Dict[str, List[dict]]:
    """
    Load GT from ActivityNet-style JSON.
    Expected: {"annotations": {video_id: {"annotations": [{"segment": [s,e], "label": ...}]}}}
    """
    with open(gt_json_path) as f:
        data = json.load(f)

    gt: Dict[str, List[dict]] = {}
    annotations = data.get("annotations", data.get("results", {}))

    for video_id, content in annotations.items():
        segs = content if isinstance(content, list) else content.get("annotations", [])
        gt[video_id] = [
            {
                "start": s["segment"][0],
                "end":   s["segment"][1],
                "label": canonical(s["label"]),
            }
            for s in segs
        ]
    return gt


def load_predictions(pred_path: str) -> Dict[str, List[dict]]:
    with open(pred_path) as f:
        data = json.load(f)
    results = data.get("results", data)
    return {
        vid: [
            {
                "start": seg["segment"][0],
                "end":   seg["segment"][1],
                "label": canonical(seg["label"]),
                "score": seg.get("score", 1.0),
            }
            for seg in segs
        ]
        for vid, segs in results.items()
    }



def temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def compute_map(
    predictions: Dict[str, List[dict]],
    ground_truth: Dict[str, List[dict]],
    iou_thresholds: List[float] = DEFAULT_IOU_THRESHOLDS,
) -> Tuple[Dict[float, float], Dict[str, Dict[float, float]]]:
    """
    Compute mAP at each IoU threshold and per-class AP at each threshold.

    Returns
    map_scores      : {iou_threshold: mAP}
    per_class_ap    : {class_label: {iou_threshold: AP}}
    """
    all_classes = sorted({seg["label"] for segs in ground_truth.values() for seg in segs})

    # per_class_ap[cls][iou] = AP
    per_class_ap: Dict[str, Dict[float, float]] = {
        cls: {t: 0.0 for t in iou_thresholds} for cls in all_classes
    }

    for iou_thresh in iou_thresholds:
        for cls in all_classes:
            # Collect all predictions for this class across all videos, sorted by score desc
            cls_preds = []
            for vid, segs in predictions.items():
                for seg in segs:
                    if seg["label"] == cls:
                        cls_preds.append((seg["score"], vid, seg["start"], seg["end"]))
            cls_preds.sort(key=lambda x: -x[0])

            # Count total GT instances for this class
            total_gt = sum(
                1 for segs in ground_truth.values()
                for seg in segs if seg["label"] == cls
            )
            if total_gt == 0:
                continue

            # Track which GT segments have been matched (per video)
            matched: Dict[str, List[bool]] = {
                vid: [False] * len([s for s in segs if s["label"] == cls])
                for vid, segs in ground_truth.items()
            }

            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))

            for i, (score, vid, ps, pe) in enumerate(cls_preds):
                gt_segs = [s for s in ground_truth.get(vid, []) if s["label"] == cls]
                if not gt_segs:
                    fp[i] = 1
                    continue

                best_iou = 0.0
                best_j = -1
                for j, gt in enumerate(gt_segs):
                    iou = temporal_iou(ps, pe, gt["start"], gt["end"])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_iou >= iou_thresh and best_j >= 0 and not matched[vid][best_j]:
                    tp[i] = 1
                    matched[vid][best_j] = True
                else:
                    fp[i] = 1

            # Compute precision-recall curve
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall    = tp_cum / total_gt
            precision = tp_cum / (tp_cum + fp_cum + 1e-8)

            # AP via 11-point interpolation
            ap = 0.0
            for thr in np.linspace(0, 1, 11):
                prec_at_rec = precision[recall >= thr]
                ap += prec_at_rec.max() if len(prec_at_rec) > 0 else 0.0
            ap /= 11.0
            per_class_ap[cls][iou_thresh] = ap

    map_scores = {
        t: float(np.mean([per_class_ap[c][t] for c in all_classes if total_gt_per_class(ground_truth, c) > 0]))
        for t in iou_thresholds
    }
    return map_scores, per_class_ap


def total_gt_per_class(ground_truth: Dict[str, List[dict]], cls: str) -> int:
    return sum(1 for segs in ground_truth.values() for seg in segs if seg["label"] == cls)


def lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def collapse_consecutive(seq: List[str]) -> List[str]:
    out = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


def sequence_from_predictions(segs: List[dict]) -> List[str]:
    sorted_segs = sorted(segs, key=lambda x: x["start"])
    return collapse_consecutive([s["label"] for s in sorted_segs])


def sequence_from_gt(segs: List[dict]) -> List[str]:
    sorted_segs = sorted(segs, key=lambda x: x["start"])
    return collapse_consecutive([s["label"] for s in sorted_segs])


def compute_lcs_metrics(
    predictions: Dict[str, List[dict]],
    ground_truth: Dict[str, List[dict]],
) -> pd.DataFrame:
    rows = []
    for vid, gt_segs in ground_truth.items():
        pred_segs = predictions.get(vid, [])
        gt_seq   = sequence_from_gt(gt_segs)
        pred_seq = sequence_from_predictions(pred_segs)

        if not gt_seq:
            continue

        lcs = lcs_length(gt_seq, pred_seq)
        recall    = lcs / len(gt_seq) * 100 if gt_seq   else 0.0
        precision = lcs / len(pred_seq) * 100 if pred_seq else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append({
            "video_id":    vid,
            "subject":     vid[:3],
            "gt_count":    len(gt_seq),
            "pred_count":  len(pred_seq),
            "pred_gt_ratio": round(len(pred_seq) / len(gt_seq), 2) if gt_seq else None,
            "lcs_length":  lcs,
            "recall":      round(recall, 2),
            "precision":   round(precision, 2),
            "f1":          round(f1, 2),
        })

    return pd.DataFrame(rows)


def compute_coverage(
    predictions: Dict[str, List[dict]],
    ground_truth: Dict[str, List[dict]],
) -> dict:
    gt_classes   = {seg["label"] for segs in ground_truth.values() for seg in segs}
    pred_classes = {seg["label"] for segs in predictions.values()  for seg in segs}
    covered      = gt_classes & pred_classes

    return {
        "gt_unique_classes":    len(gt_classes),
        "pred_unique_classes":  len(pred_classes),
        "class_coverage_pct":   round(len(covered) / len(gt_classes) * 100, 1) if gt_classes else 0,
        "hallucinated_classes": sorted(pred_classes - gt_classes),
        "missed_classes":       sorted(gt_classes - pred_classes),
    }


def print_map_results(map_scores: Dict[float, float], iou_thresholds: List[float]) -> None:
    print("\n" + "=" * 55)
    print("SEGMENT-LEVEL mAP")
    print("-" * 55)
    for t in iou_thresholds:
        print(f"  mAP @ IoU {t:.1f}  :  {map_scores[t]*100:.2f}%")
    print("=" * 55)


def print_lcs_summary(df: pd.DataFrame) -> None:
    print("\n" + "-" * 50)
    print("LCS SEQUENCE METRICS")
    print(f"  Videos scored       : {len(df)}")
    print(f"  Avg LCS Recall      : {df['recall'].mean():.1f}%")
    print(f"  Avg LCS Precision   : {df['precision'].mean():.1f}%")
    print(f"  Avg LCS F1          : {df['f1'].mean():.1f}%")
    print(f"  Avg pred/GT ratio   : {df['pred_gt_ratio'].mean():.2f}x")
    print("-" * 55)
    print("Per-subject breakdown:")
    for subj, grp in df.groupby("subject"):
        print(f"  {subj}  n={len(grp):>2}  "
              f"recall={grp['recall'].mean():.1f}%  "
              f"precision={grp['precision'].mean():.1f}%  "
              f"f1={grp['f1'].mean():.1f}%")
    print("-" * 50)


def print_coverage(cov: dict) -> None:
    print("\n" + "-" * 50)
    print("CLASS COVERAGE")
    print(f"  GT unique classes    : {cov['gt_unique_classes']}")
    print(f"  Pred unique classes  : {cov['pred_unique_classes']}")
    print(f"  Class coverage       : {cov['class_coverage_pct']}%")
    if cov["missed_classes"]:
        print(f"  Missed classes ({len(cov['missed_classes'])})  : {', '.join(cov['missed_classes'][:8])}{'...' if len(cov['missed_classes']) > 8 else ''}")
    if cov["hallucinated_classes"]:
        print(f"  Hallucinated ({len(cov['hallucinated_classes'])})    : {', '.join(cov['hallucinated_classes'][:8])}{'...' if len(cov['hallucinated_classes']) > 8 else ''}")
    print("-" * 50)


def print_per_class_ap(per_class_ap: Dict[str, Dict[float, float]], iou: float = 0.5, top_n: int = 15) -> None:
    print(f"\n{'-'*50}")
    print(f"PER-CLASS AP @ IoU {iou}")
    print(f"{'-'*50}")
    rows = sorted(per_class_ap.items(), key=lambda x: -x[1].get(iou, 0))
    print(f"  {'Class':<40} AP")
    print(f"  {'-'*38} ------")
    for cls, aps in rows[:top_n]:
        print(f"  {cls:<40} {aps.get(iou,0)*100:.1f}%")
    if len(rows) > top_n:
        print(f"  ... ({len(rows) - top_n} more classes)")
    print("-" * 50)


def evaluate(
    pred_path: str,
    gt_source: str,
    fps: float = DEFAULT_FPS,
    iou_thresholds: List[float] = DEFAULT_IOU_THRESHOLDS,
    output_csv: Optional[str] = None,
    system_name: str = "System",
) -> dict:
    print(f"\n{'-'*50}")
    print(f"  Evaluating: {system_name}")
    print(f"  Predictions: {pred_path}")
    print(f"{'-'*55}")

    predictions = load_predictions(pred_path)
    video_ids   = list(predictions.keys())

    if os.path.isdir(gt_source):
        ground_truth = load_gt_from_csv_dir(gt_source, video_ids, fps)
    elif os.path.isfile(gt_source) and gt_source.endswith(".json"):
        ground_truth = load_gt_from_json(gt_source)
        ground_truth = {k: v for k, v in ground_truth.items() if k in predictions}
    else:
        raise ValueError(f"Cannot load GT from: {gt_source}")

    print(f"  Videos with GT: {len(ground_truth)} / {len(video_ids)}")

    # 1. LCS metrics
    lcs_df = compute_lcs_metrics(predictions, ground_truth)
    print_lcs_summary(lcs_df)

    # 2. mAP
    map_scores, per_class_ap = compute_map(predictions, ground_truth, iou_thresholds)
    print_map_results(map_scores, iou_thresholds)
    print_per_class_ap(per_class_ap, iou=0.5)

    # 3. Coverage
    cov = compute_coverage(predictions, ground_truth)
    print_coverage(cov)

    # 4. Save per-video CSV
    if output_csv:
        lcs_df.to_csv(output_csv, index=False)
        print(f"\n  Per-video results saved to: {output_csv}")

    return {
        "system":      system_name,
        "lcs_recall":  lcs_df["recall"].mean(),
        "lcs_prec":    lcs_df["precision"].mean(),
        "lcs_f1":      lcs_df["f1"].mean(),
        "map_0.1":     map_scores.get(0.1, 0) * 100,
        "map_0.3":     map_scores.get(0.3, 0) * 100,
        "map_0.5":     map_scores.get(0.5, 0) * 100,
        "class_coverage": cov["class_coverage_pct"],
        "pred_gt_ratio":  lcs_df["pred_gt_ratio"].mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Complete TAD/VLM evaluation")
    parser.add_argument("--pred",  nargs="+", required=True,  help="Path(s) to prediction JSON(s)")
    parser.add_argument("--gt",    required=True,              help="GT CSV directory or ActivityNet JSON")
    parser.add_argument("--fps",   type=float, default=DEFAULT_FPS, help="Fallback FPS for frame→time conversion")
    parser.add_argument("--iou",   nargs="+", type=float, default=DEFAULT_IOU_THRESHOLDS, help="IoU thresholds for mAP")
    parser.add_argument("--names", nargs="+", default=None,   help="System names (one per --pred)")
    parser.add_argument("--out",   default=None,               help="Output CSV path for per-video results")
    args = parser.parse_args()

    names = args.names or [os.path.basename(p).replace(".json", "") for p in args.pred]
    if len(names) != len(args.pred):
        parser.error("--names must have the same number of entries as --pred")

    summary_rows = []
    for pred_path, name in zip(args.pred, names):
        out_csv = args.out.replace(".csv", f"_{name}.csv") if args.out and len(args.pred) > 1 else args.out
        row = evaluate(pred_path, args.gt, args.fps, args.iou, out_csv, name)
        summary_rows.append(row)

    if len(summary_rows) > 1:
        print("\n" + "-" * 70)
        print("COMPARISON SUMMARY")
        print("#" * 70)
        df = pd.DataFrame(summary_rows).set_index("system")
        print(df.to_string(float_format=lambda x: f"{x:.1f}"))
        print("-" * 70)

        if args.out:
            comparison_path = args.out.replace(".csv", "_comparison.csv")
            df.to_csv(comparison_path)
            print(f"Comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()
