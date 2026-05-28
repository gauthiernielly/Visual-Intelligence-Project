"""
Evaluate temporal segmentation predictions against GT annotations.

Metrics:
  - Event-mAP at IoU in {0.05, 0.1, 0.3}
  - Global LCS recall, precision and F1-score
  - Per-class recall at IoU >= 0.05
  - Recall by event duration at IoU >= 0.1
  - Hallucination rate and per-class analysis at IoU >= 0.05

Usage:
    python src/evaluate.py
    python src/evaluate.py --annotation_dir /path/to/annotations --output_dir results/
    python src/evaluate.py --eval_indices path/to/eval_indices.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config as cfg
from metrics import compute_event_map, compute_lcs_metrics, compute_per_class_recall, \
                    compute_per_duration_recall, compute_hallucination_analysis, \
                    compute_substitutions
from graphs import plot_recall_by_duration, plot_per_class_recall, plot_timeline_viewer


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate temporal segmentation predictions")
    p.add_argument("--annotation_dir", default=cfg.GT_DIR)
    p.add_argument("--output_dir",     default=cfg.OUTPUT_DIR)
    p.add_argument("--iou_thresholds", nargs="+", type=float, default=cfg.MAP_IOU_THRESHOLDS)
    p.add_argument("--cls_recall_iou",     type=float, default=cfg.CLASS_RECALL_IOU)
    p.add_argument("--dur_recall_iou",     type=float, default=cfg.DURATION_RECALL_IOU)
    p.add_argument("--hallu_iou",     type=float, default=cfg.HALLU_IOU)
    p.add_argument("--eval_indices",   default=None,
                   help="Path to a JSON file listing video IDs to process. "
                        "If omitted, all predictions are evaluated.")

    args, _ = p.parse_known_args()
    return args


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_predictions(out_dir: str) -> dict[str, list[dict]]:
    """
    Reads generated_segments.json produced by generate.py.
    Returns {video_id: [{"event": ..., "start_frame": ..., "end_frame": ...}, ...]}.
    Entries that contain an "error" key are dropped.
    """
    pred_path = Path(out_dir) / "generated_segments.json"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}\n"
            "Please run generate.py first to produce generated_segments.json."
        )
    with open(pred_path) as f:
        raw = json.load(f)
    return {
        vid: entry["segments"]
        for vid, entry in raw.items()
        if "error" not in entry
    }


def filter_by_eval_indices(preds: dict[str, list[dict]], indices_path: str) -> dict[str, list[dict]]:
    """
    Keeps only the video IDs listed in eval_indices.json.
    The file must contain a JSON array of video ID strings.
    """
    path = Path(indices_path)
    if not path.exists():
        raise FileNotFoundError(f"eval_indices file not found: {path}")
    with open(path) as f:
        video_ids: list[str] = json.load(f)
    unknown = [v for v in video_ids if v not in preds]
    if unknown:
        print(f"  [warn] {len(unknown)} video ID(s) in eval_indices not found in predictions: {unknown}")
    return {vid: preds[vid] for vid in video_ids if vid in preds}


def load_gt(video_id: str, annotation_dir: str) -> list[dict] | None:
    """
    Reads <annotation_dir>/<subject_id>/<video_id>.csv.
    Subject ID is the leading prefix of video_id (e.g. P10T03C04 → P10).
    Returns None if the file does not exist.
    """
    subject_id = video_id[:3]
    csv_path = Path(annotation_dir) / subject_id / f"{video_id}.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, usecols=["event", "start_frame", "end_frame"])
        df["event"] = df["event"].replace(cfg.CLASS_ALIASES)
        return df.to_dict("records")
    except Exception as e:
        print(f"  [Warning] Failed to parse CSV for {video_id}: {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"Loading predictions from {args.output_dir}/generated_segments.json ...")
    all_preds = load_predictions(args.output_dir)
    if args.eval_indices is not None:
        print(f"Filtering to subset defined in {args.eval_indices} ...")
        all_preds = filter_by_eval_indices(all_preds, args.eval_indices)

    print(f"Loading GT annotations from {args.annotation_dir} ...")
    all_gts: dict[str, list[dict]] = {}
    n_missing = 0
    for vid in list(all_preds):
        gt = load_gt(vid, args.annotation_dir)
        if gt is None:
            print(f"  [warn] no annotation file for {vid}, skipping")
            del all_preds[vid]
            n_missing += 1
        else:
            all_gts[vid] = gt

    n_videos = len(all_preds)
    n_pred = sum(len(s) for s in all_preds.values())
    n_gt = sum(len(s) for s in all_gts.values())
    suffix = f"  ({n_missing} skipped — no GT file)" if n_missing else ""
    print(f"  {n_videos} videos  |  {n_pred} predictions  |  {n_gt} GT segments{suffix}")

    print("\nComputing Event-mAP ...")
    event_map, per_class_ap = compute_event_map(all_preds, all_gts, args.iou_thresholds)

    print("Computing Global Average LCS metrics ...")
    lcs_metrics = compute_lcs_metrics(all_preds, all_gts)
    lcs_recall    = lcs_metrics["recall"]
    lcs_precision = lcs_metrics["precision"]
    lcs_f1        = lcs_metrics["f1"]
    lcs_per_video = lcs_metrics["per_video"]

    print("Computing per-class recall ...")
    per_class_recall = compute_per_class_recall(all_preds, all_gts, args.cls_recall_iou)

    print("Computing detection rate by event duration ...")
    detection_by_duration = compute_per_duration_recall(all_preds, all_gts, args.dur_recall_iou)

    print("Computing hallucination analysis ...")
    hallucination = compute_hallucination_analysis(all_preds, all_gts, args.hallu_iou)

    print("Computing substitution errors ...")
    substitutions = compute_substitutions(all_preds, all_gts, args.hallu_iou)
    

    # Print summary
    recall_vals = [v for v in per_class_recall.values() if v is not None]
    mean_recall = round(float(np.mean(recall_vals)), 4) if recall_vals else None

    print("\n── Results ───────────────────────────────────────────")
    for k, v in event_map.items():
        print(f"  {k:<20} {v:.4f}")
    if mean_recall is not None:
        print(f"  {'Mean recall':<20} {mean_recall * 100:.2f} %"
              f"  (over {len(recall_vals)} classes with ≥1 GT instance)")
    print(f"  {'Hallucination rate':<20} {hallucination['rate'] * 100:.2f} %")
    print(f"  {'Global LCS Recall':<20} {lcs_recall * 100:.2f} %")
    print(f"  {'Global LCS Precision':<20} {lcs_precision * 100:.2f} %")
    print(f"  {'Global LCS F1-Score':<20} {lcs_f1 * 100:.2f} %")

    print("\n  Detection by duration:")
    for row in detection_by_duration:
        recall_str = f"{row['recall'] * 100:.1f} %" if row["recall"] is not None else "  N/A  "
        print(f"    {row['bin']:<8}  GT: {row['total']:>4}  detected: {row['detected']:>4}  recall: {recall_str}")

    print("\n  Top hallucinated classes:")
    for entry in hallucination["by_class"][:10]:
        print(f"    {entry['class']:<25} {entry['count']:>4}  ({entry['fraction'] * 100:.1f} %)")

    print("\n  Top substitution errors (predicted → gt):")
    for entry in substitutions:
        print(f"    {entry['predicted_class']:<25} → {entry['gt_class']:<35} ×{entry['count']}")

    print("\n  Per-subject LCS (recall / precision / F1):")
    subject_entries: dict[str, list] = {}
    for entry in lcs_per_video:
        subject_entries.setdefault(entry["video_id"][:3], []).append(entry)
    for subj in sorted(subject_entries):
        entries = subject_entries[subj]
        sr = float(np.mean([e["recall"]    for e in entries]))
        sp = float(np.mean([e["precision"] for e in entries]))
        sf = float(np.mean([e["f1"]        for e in entries]))
        print(f"    {subj}   R: {sr * 100:.1f} %   P: {sp * 100:.1f} %   F1: {sf * 100:.1f} %   ({len(entries)} videos)")
    print("──────────────────────────────────────────────────────")

    # Save
    metrics = {
        "event_map":              event_map,
        "per_class_ap":           per_class_ap,
        "mean_recall":            mean_recall,
        "per_class_recall":       per_class_recall,
        "hallucination_rate":     hallucination["rate"],
        "hallucination_by_class": hallucination["by_class"],
        "lcs_recall":             lcs_recall,
        "lcs_precision":          lcs_precision,
        "lcs_f1":                 lcs_f1,
        "detection_by_duration":  detection_by_duration,
        "substitutions":          substitutions,
        "meta": {
            "n_videos":             n_videos,
            "n_predictions":        n_pred,
            "n_gt_segments":        n_gt,
            "iou_thresholds":       args.iou_thresholds,
            "class_recall_iou":     args.cls_recall_iou,
            "duration_recall_iou":  args.dur_recall_iou,
            "hallucination_iou":    args.hallu_iou
        },
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    lcs_path = out_dir / "lcs_per_video.json"
    with open(lcs_path, "w") as f:
        json.dump(lcs_per_video, f, indent=2)
    print(f"Per-video LCS saved to {lcs_path}")

    print("\nGenerating figures ...")
    plot_recall_by_duration(detection_by_duration, args.output_dir)
    plot_per_class_recall(per_class_recall, all_gts, args.output_dir)
    plot_timeline_viewer(all_gts, all_preds, args.output_dir)


if __name__ == "__main__":
    main()
