from collections import defaultdict
from pathlib import Path
import sys
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import config as cfg


# ── Helper functions ─────────────────────────────────────────────────────────

def temporal_iou(a: dict, b: dict) -> float:
    inter = max(0, min(a["end_frame"], b["end_frame"]) - max(a["start_frame"], b["start_frame"]))
    union = (a["end_frame"] - a["start_frame"]) + (b["end_frame"] - b["start_frame"]) - inter
    return inter / union if union > 0 else 0.0


def _ap_11point(tp_flags: list[int], n_gt: int) -> float:
    """11-point interpolated average precision."""
    if n_gt == 0 or not tp_flags:
        return 0.0
    tp   = np.cumsum(tp_flags, dtype=float)
    fp   = np.cumsum(1 - np.array(tp_flags, dtype=float))
    rec  = tp / n_gt
    prec = tp / (tp + fp)
    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p_at_r = prec[rec >= r]
        ap += float(p_at_r.max()) if len(p_at_r) > 0 else 0.0
    return ap / 11


# ── Metric 1 — Event-mAP ─────────────────────────────────────────────────────

def compute_event_map(
    all_preds: dict[str, list[dict]],
    all_gts:   dict[str, list[dict]],
    iou_thresholds: list[float],
) -> tuple[dict[str, float], dict[str, dict[str, Optional[float]]]]:
    """
    Returns (map_scores, per_class_ap).
      map_scores:    {"mAP@<thresh>": float, ...}
      per_class_ap:  {"<thresh>": {"<class>": float | None, ...}, ...}
    Per-class AP uses 11-point interpolation. Classes with no GT instances
    are excluded from the mAP average and stored as None in per_class_ap.
    """
    map_scores:   dict[str, float]                           = {}
    per_class_ap: dict[str, dict[str, Optional[float]]]      = {}

    for iou_thresh in iou_thresholds:
        preds_by_class: dict[str, list[tuple]] = defaultdict(list)
        gts_by_class:   dict[str, list[tuple]] = defaultdict(list)
        for vid, segs in all_preds.items():
            for s in segs:
                preds_by_class[s["event"]].append((vid, s))
        for vid, segs in all_gts.items():
            for s in segs:
                gts_by_class[s["event"]].append((vid, s))

        aps: list[float] = []
        cls_aps: dict[str, Optional[float]] = {}
        for cls in cfg.TSU_CLASSES:
            cls_gts = gts_by_class.get(cls, [])
            if not cls_gts:
                cls_aps[cls] = None  # no GT instances — excluded from average
                continue
            cls_preds = preds_by_class.get(cls, [])
            if not cls_preds:
                aps.append(0.0)
                cls_aps[cls] = 0.0
                continue

            matched_gt: dict[str, set[int]] = defaultdict(set)
            tp_flags = []
            for vid, pred in cls_preds:
                vid_gts = [(j, g) for j, (v, g) in enumerate(cls_gts) if v == vid]
                best_iou, best_j = 0.0, -1
                for j, gt in vid_gts:
                    if j in matched_gt[vid]:
                        continue
                    iou = temporal_iou(pred, gt)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_thresh:
                    matched_gt[vid].add(best_j)
                    tp_flags.append(1)
                else:
                    tp_flags.append(0)

            ap = _ap_11point(tp_flags, len(cls_gts))
            aps.append(ap)
            cls_aps[cls] = round(ap, 4)

        map_scores[f"mAP@{iou_thresh}"]       = round(float(np.mean(aps)) if aps else 0.0, 4)
        per_class_ap[str(iou_thresh)] = cls_aps

    return map_scores, per_class_ap


# ── Metric 2 — Longest Common Subsequence ────────────────────────────────────

def lcs_metrics_single(gt_seq: list[str], pred_seq: list[str]) -> tuple[float, float, float, int]:
    """
    LCS recall, precision, F1 and LCS length for a single video.
    Recall/precision/F1 are in [0, 1].
    """
    if not gt_seq:
        return 0.0, 0.0, 0.0, 0

    m, n = len(gt_seq), len(pred_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_seq[i - 1] == pred_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    recall    = lcs_length / m
    precision = lcs_length / n if n > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, f1, lcs_length


def compute_lcs_metrics(
    all_preds: dict[str, list[dict]],
    all_gts:   dict[str, list[dict]],
) -> dict:
    """
    Returns a dict with:
      "recall", "precision", "f1"  — macro averages across videos (values in [0, 1])
      "per_video"                  — list of per-video dicts with video_id, gt_length,
                                     pred_length, lcs_length, recall, precision, f1
    Videos with empty GT are skipped.
    """
    recalls, precisions, f1s = [], [], []
    per_video = []
    for vid, gt_segs in all_gts.items():
        gt_seq   = [s["event"] for s in sorted(gt_segs,                   key=lambda s: s["start_frame"])]
        pred_seq = [s["event"] for s in sorted(all_preds.get(vid, []),    key=lambda s: s["start_frame"])]
        if not gt_seq:
            continue
        r, p, f, lcs_len = lcs_metrics_single(gt_seq, pred_seq)
        recalls.append(r)
        precisions.append(p)
        f1s.append(f)
        per_video.append({
            "video_id":    vid,
            "gt_length":   len(gt_seq),
            "pred_length": len(pred_seq),
            "lcs_length":  lcs_len,
            "recall":      round(r, 4),
            "precision":   round(p, 4),
            "f1":          round(f, 4),
        })
    if not recalls:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "per_video": []}
    return {
        "recall":    round(float(np.mean(recalls)),    4),
        "precision": round(float(np.mean(precisions)), 4),
        "f1":        round(float(np.mean(f1s)),        4),
        "per_video": per_video,
    }


# ── Metric 3 — Per-class recall ──────────────────────────────────────────────

def compute_per_class_recall(
    all_preds: dict[str, list[dict]],
    all_gts:   dict[str, list[dict]],
    iou_thresh: float,
) -> dict[str, float | None]:
    preds_by_cls_vid: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for vid, segs in all_preds.items():
        for s in segs:
            preds_by_cls_vid[s["event"]][vid].append(s)

    recall_by_class: dict[str, float | None] = {}
    for cls in cfg.TSU_CLASSES:
        matched = total = 0
        for vid, segs in all_gts.items():
            cls_gts = [s for s in segs if s["event"] == cls]
            cls_preds = preds_by_cls_vid[cls].get(vid, [])
            for gt in cls_gts:
                total += 1
                if any(temporal_iou(p, gt) >= iou_thresh for p in cls_preds):
                    matched += 1
        recall_by_class[cls] = round(matched / total, 4) if total > 0 else None
    return recall_by_class


# ── Metric 4 — Recall by event duration ──────────────────────────────

def compute_per_duration_recall(
    all_preds: dict[str, list[dict]],
    all_gts:   dict[str, list[dict]],
    iou_thresh: float,
    fps: float = cfg.DATASET_FPS,
) -> list[dict]:
    """
    For each duration bin, counts GT instances and how many were detected
    (same-class prediction with IoU >= iou_thresh).
    """
    bins: dict[str, dict[str, int]] = {
        label: {"total": 0, "detected": 0} for label, _, _ in cfg.DURATION_BINS
    }
    for vid, gt_segs in all_gts.items():
        preds = all_preds.get(vid, [])
        for gt in gt_segs:
            duration_s = (gt["end_frame"] - gt["start_frame"]) / fps
            label = next(lbl for lbl, lo, hi in cfg.DURATION_BINS if lo <= duration_s < hi)
            bins[label]["total"] += 1
            same_cls = [p for p in preds if p["event"] == gt["event"]]
            if any(temporal_iou(p, gt) >= iou_thresh for p in same_cls):
                bins[label]["detected"] += 1

    result = []
    for label, _, _ in cfg.DURATION_BINS:
        total    = bins[label]["total"]
        detected = bins[label]["detected"]
        result.append({
            "bin":      label,
            "total":    total,
            "detected": detected,
            "recall":   round(detected / total, 4) if total > 0 else None,
        })
    return result


# ── Metric 5 — Hallucination analysis ───────────────────────────────────────

def compute_hallucination_analysis(
    all_preds: dict[str, list[dict]],
    all_gts:   dict[str, list[dict]],
    iou_thresh: float,
) -> dict:
    """
    Returns hallucination rate + per-class breakdown ranked by count.
    A prediction is a hallucination when no GT segment (any class) overlaps
    it at IoU >= iou_thresh.
    """
    total = 0
    halluc_by_class: dict[str, int] = defaultdict(int)
    for vid, preds in all_preds.items():
        gts = all_gts.get(vid, [])
        for pred in preds:
            total += 1
            if not any(temporal_iou(pred, gt) >= iou_thresh for gt in gts):
                halluc_by_class[pred["event"]] += 1

    n_halluc = sum(halluc_by_class.values())
    rate = round(n_halluc / total, 4) if total > 0 else 0.0
    by_class = sorted(
        [
            {
                "class": cls,
                "count": cnt,
                "fraction": round(cnt / n_halluc, 4) if n_halluc > 0 else 0.0,
            }
            for cls, cnt in halluc_by_class.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )
    return {"rate": rate, "by_class": by_class}


# ── Metric 6 — Substitution errors ───────────────────────────────────────────

def compute_substitutions(
    all_preds: dict[str, list[dict]],
    all_gts:   dict[str, list[dict]],
    iou_thresh: float,
    top_k: int = 10,
) -> list[dict]:
    """
    For each missed GT event (no same-class prediction at IoU >= iou_thresh),
    records every different-class prediction that overlaps it.
    Returns the top_k (predicted_class, gt_class) pairs ranked by count.
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for vid, gt_segs in all_gts.items():
        preds = all_preds.get(vid, [])
        for gt in gt_segs:
            same_cls = [p for p in preds if p["event"] == gt["event"]]
            if any(temporal_iou(p, gt) >= iou_thresh for p in same_cls):
                continue  # correctly detected
            for pred in preds:
                if pred["event"] != gt["event"] and temporal_iou(pred, gt) >= iou_thresh:
                    counts[(pred["event"], gt["event"])] += 1

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"predicted_class": pred_cls, "gt_class": gt_cls, "count": cnt}
        for (pred_cls, gt_cls), cnt in ranked
    ]