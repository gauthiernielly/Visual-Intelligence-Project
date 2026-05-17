import json
import numpy as np
from collections import defaultdict

# --- Configuration ---
GT_PATH          = "data_cs_split.json"
HYBRID_PATH      = "hybrid_pipeline_results.json"
TAD_PATH         = "TAD_full_run/outputs/predictions_canonical.json"
IOU_THRESHOLDS   = [0.3, 0.5, 0.7]

# --- GT Loader ---

def load_gt(gt_path):
    """
    Loads GT from ActivityNet-style JSON.
    Returns dict: video_id -> list of {"segment": [s, e], "label": str}
    Only keeps videos from the test subset.
    """
    with open(gt_path) as f:
        raw = json.load(f)

    gt = {}
    for video_id, data in raw["database"].items():
        if data["subset"] != "testing":
            continue
        gt[video_id] = [
            {"segment": ann["segment"], "label": ann["label"]}
            for ann in data["annotations"]
        ]
    return gt


def load_predictions(pred_path):
    """
    Loads predictions from {"results": {video_id: [{"segment", "label", "score"}]}} format.
    Works for both TAD and hybrid outputs.
    """
    with open(pred_path) as f:
        raw = json.load(f)
    return raw["results"]


# --- Metrics ---

def temporal_iou(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    if inter == 0:
        return 0.0
    return inter / ((e1 - s1) + (e2 - s2) - inter)


def compute_ap_11pt(preds, n_gt):
    """
    11-point interpolated AP.
    preds: list of (score, is_tp) sorted descending by score.
    """
    if n_gt == 0:
        return 0.0
    tp = fp = 0
    precisions, recalls = [], []
    for _, is_tp in sorted(preds, key=lambda x: -x[0]):
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)
    return sum(
        max((p for p, r in zip(precisions, recalls) if r >= t), default=0.0)
        for t in np.linspace(0, 1, 11)
    ) / 11


def evaluate(predictions, gt, iou_threshold):
    """
    Computes per-class AP and mAP at a given IoU threshold.
    Also computes segment-level P/R/F1 (class-agnostic).
    """
    class_preds    = defaultdict(list)   # label -> [(score, is_tp)]
    class_gt_count = defaultdict(int)    # label -> n_gt

    total_pred = total_gt = total_tp = 0

    for video_id, gt_segs in gt.items():
        preds = predictions.get(video_id, [])

        for g in gt_segs:
            class_gt_count[g["label"]] += 1
        total_gt += len(gt_segs)
        total_pred += len(preds)

        # --- mAP matching (per-class, greedy by score) ---
        gt_matched = [False] * len(gt_segs)
        for pred in sorted(preds, key=lambda x: -x["score"]):
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt_segs):
                if g["label"] != pred["label"]:
                    continue
                iou = temporal_iou(*pred["segment"], *g["segment"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            is_tp = best_iou >= iou_threshold and best_j >= 0 and not gt_matched[best_j]
            if is_tp:
                gt_matched[best_j] = True
            class_preds[pred["label"]].append((pred["score"], is_tp))

        # --- Segment-level P/R/F1 (class-agnostic) ---
        seg_matched = [False] * len(gt_segs)
        for pred in preds:
            for j, g in enumerate(gt_segs):
                if not seg_matched[j] and temporal_iou(*pred["segment"], *g["segment"]) >= iou_threshold:
                    seg_matched[j] = True
                    total_tp += 1
                    break

    # Per-class AP
    per_class_ap = {
        cls: compute_ap_11pt(class_preds[cls], class_gt_count[cls])
        for cls in class_gt_count
    }
    mean_ap = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0

    precision = total_tp / total_pred if total_pred else 0.0
    recall    = total_tp / total_gt   if total_gt   else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "mAP":          mean_ap,
        "precision":    precision,
        "recall":       recall,
        "f1":           f1,
        "per_class_ap": per_class_ap,
    }


# --- Reporting ---

def print_report(name, results_by_iou):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  {'IoU':<8} {'mAP':>8} {'P':>8} {'R':>8} {'F1':>8}")
    print(f"  {'-'*44}")
    for iou, r in results_by_iou.items():
        print(f"  {iou:<8.1f} {r['mAP']:>8.3f} {r['precision']:>8.3f} {r['recall']:>8.3f} {r['f1']:>8.3f}")

    # Per-class AP at IoU=0.5
    r50 = results_by_iou[0.5]
    print(f"\n  Per-class AP @ IoU=0.5:")
    print(f"  {'-'*36}")
    for cls, ap in sorted(r50["per_class_ap"].items(), key=lambda x: -x[1]):
        print(f"  {cls:<40} {ap:.3f}")


def print_comparison(tad_results, hybrid_results):
    print(f"\n{'='*60}")
    print(f"  TAD vs Hybrid — Per-class AP @ IoU=0.5")
    print(f"{'='*60}")
    print(f"  {'Class':<40} {'TAD':>8} {'Hybrid':>8} {'Delta':>8}")
    print(f"  {'-'*60}")

    tad_ap    = tad_results[0.5]["per_class_ap"]
    hybrid_ap = hybrid_results[0.5]["per_class_ap"]
    all_cls   = sorted(set(tad_ap) | set(hybrid_ap))

    for cls in all_cls:
        t = tad_ap.get(cls, 0.0)
        h = hybrid_ap.get(cls, 0.0)
        delta = h - t
        marker = " ▲" if delta > 0.01 else (" ▼" if delta < -0.01 else "")
        print(f"  {cls:<40} {t:>8.3f} {h:>8.3f} {delta:>+8.3f}{marker}")

    t_mean = tad_results[0.5]["mAP"]
    h_mean = hybrid_results[0.5]["mAP"]
    print(f"  {'-'*60}")
    print(f"  {'mAP':<40} {t_mean:>8.3f} {h_mean:>8.3f} {h_mean - t_mean:>+8.3f}")


# --- Main ---

def main():
    print("Loading GT (test subset)...")
    gt = load_gt(GT_PATH)
    print(f"  {len(gt)} test videos, "
          f"{sum(len(v) for v in gt.values())} GT segments.")

    print("Loading predictions...")
    tad_preds    = load_predictions(TAD_PATH)
    hybrid_preds = load_predictions(HYBRID_PATH)

    tad_results    = {}
    hybrid_results = {}

    for iou in IOU_THRESHOLDS:
        tad_results[iou]    = evaluate(tad_preds,    gt, iou)
        hybrid_results[iou] = evaluate(hybrid_preds, gt, iou)

    print_report("TAD Baseline",    tad_results)
    print_report("Hybrid Pipeline", hybrid_results)
    print_comparison(tad_results, hybrid_results)


if __name__ == "__main__":
    main()