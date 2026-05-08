"""
visualize_results.py
====================
Produce the same visual artifacts the smoke notebook produced:
  - training_curves.png   : 3-panel (train loss, val loss, val mAP) across epochs
  - predictions_analysis.png : score histogram + per-class counts + duration histo
  - gantt_overlay_<vid>.png  : GT (blue) vs predictions (red) timeline per video

OpenTAD's "log.json" is plain text (a logging.FileHandler output) despite the
name, so we parse it with regex.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIN_RE   = re.compile(r"\[Train\]: \[(\d+)\]\[\d+/\d+\]\s+Loss=([\d.]+)\s+cls_loss=([\d.]+)\s+reg_loss=([\d.]+)")
VAL_RE     = re.compile(r"\[Val\]: \[(\d+)\]\s+Loss=([\d.]+)\s+cls_loss=([\d.]+)\s+reg_loss=([\d.]+)")
AVG_MAP_RE = re.compile(r"Average-mAP: ([\d.]+) \(%\)")


def plot_training_curves(log_path, out_path, label="full"):
    txt = Path(log_path).read_text()
    train = [(int(e), float(L), float(c), float(r)) for e, L, c, r in TRAIN_RE.findall(txt)]
    val   = [(int(e), float(L), float(c), float(r)) for e, L, c, r in VAL_RE.findall(txt)]
    avg_map = [float(x) for x in AVG_MAP_RE.findall(txt)]

    print(f"  parsed {len(train)} train rows, {len(val)} val rows, {len(avg_map)} mAP rows")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    if train:
        # multiple log lines per epoch -> aggregate by epoch (mean)
        by_e = {}
        for e, L, c, r in train:
            by_e.setdefault(e, []).append((L, c, r))
        es = sorted(by_e)
        Ls = [np.mean([t[0] for t in by_e[e]]) for e in es]
        cs = [np.mean([t[1] for t in by_e[e]]) for e in es]
        rs = [np.mean([t[2] for t in by_e[e]]) for e in es]
        axes[0].plot(es, Ls, "o-", color="C0", label="total")
        axes[0].plot(es, cs, "s-", color="C1", label="cls")
        axes[0].plot(es, rs, "^-", color="C2", label="reg")
        axes[0].set_title(f"Training loss ({label})")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

    if val:
        es, Ls, cs, rs = zip(*val)
        axes[1].plot(es, Ls, "o-", color="C0", label="total")
        axes[1].plot(es, cs, "s-", color="C1", label="cls")
        axes[1].plot(es, rs, "^-", color="C2", label="reg")
        axes[1].set_title("Validation loss")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
        axes[1].legend(); axes[1].grid(alpha=0.3)

    if avg_map:
        epochs = list(range(1, len(avg_map) + 1))
        axes[2].plot(epochs, avg_map, "D-", color="C3")
        axes[2].set_title("Validation Average-mAP (%)")
        axes[2].set_xlabel("epoch"); axes[2].set_ylabel("mAP %")
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_predictions_analysis(canon_path, ann_path, out_path):
    with open(canon_path) as f:
        preds = json.load(f).get("results", {})
    with open(ann_path) as f:
        gt_db = json.load(f)["database"]

    all_scores, all_labels, all_durs = [], [], []
    for vid, segs in preds.items():
        for s in segs:
            all_scores.append(s["score"])
            all_labels.append(s["label"])
            all_durs.append(s["segment"][1] - s["segment"][0])

    gt_durs = []
    for vid in preds.keys():
        if vid in gt_db:
            for ann in gt_db[vid].get("annotations", []):
                gt_durs.append(ann["segment"][1] - ann["segment"][0])

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    if all_scores:
        axes[0].hist(all_scores, bins=40, color="steelblue", alpha=0.85)
        axes[0].set_title(f"Score distribution (n={len(all_scores)})")
        axes[0].set_xlabel("confidence score"); axes[0].set_ylabel("# predictions")
        axes[0].grid(alpha=0.3)

    if all_labels:
        cnt = Counter(all_labels)
        # top-30 classes for readability on the full run
        top = sorted(cnt, key=lambda c: -cnt[c])[:30]
        y = np.arange(len(top))
        axes[1].barh(y, [cnt[c] for c in top], color="C2", alpha=0.85)
        axes[1].set_yticks(y); axes[1].set_yticklabels(top, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_title(f"Top-{len(top)} predicted classes (out of {len(cnt)})")
        axes[1].set_xlabel("# predictions"); axes[1].grid(alpha=0.3, axis="x")

    if all_durs or gt_durs:
        hi = max(max(all_durs, default=1.0), max(gt_durs, default=1.0)) * 1.1
        bins = np.logspace(np.log10(0.1), np.log10(hi), 40)
        if gt_durs:
            axes[2].hist(gt_durs, bins=bins, alpha=0.55, color="C0", label=f"GT (n={len(gt_durs)})")
        if all_durs:
            axes[2].hist(all_durs, bins=bins, alpha=0.55, color="C3", label=f"pred (n={len(all_durs)})")
        axes[2].set_xscale("log")
        axes[2].set_title("Segment durations: GT vs predicted")
        axes[2].set_xlabel("duration (s, log)"); axes[2].set_ylabel("# segments")
        axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_gantt(vid, gt_db, preds, out_path):
    info = gt_db[vid]
    duration = info["duration"]

    gt_by_class = {}
    for s in info.get("annotations", []):
        gt_by_class.setdefault(s["label"], []).append(s["segment"])
    pred_by_class = {}
    for s in preds.get(vid, []):
        pred_by_class.setdefault(s["label"], []).append(s["segment"])

    classes = sorted(set(gt_by_class) | set(pred_by_class))
    if not classes:
        return False
    fig, ax = plt.subplots(figsize=(14, 0.4 * max(8, len(classes))))
    for i, cls in enumerate(classes):
        for start, end in gt_by_class.get(cls, []):
            ax.barh(i + 0.18, end - start, left=start, height=0.35,
                    color="steelblue", alpha=0.75)
        for start, end in pred_by_class.get(cls, []):
            ax.barh(i - 0.18, max(end - start, duration * 0.002),
                    left=start, height=0.35, color="crimson", alpha=0.85)
        ax.text(-duration * 0.005, i, cls, ha="right", va="center", fontsize=8)
    ax.set_xlim(0, duration); ax.set_ylim(-0.7, len(classes) + 0.3)
    ax.set_yticks([]); ax.set_xlabel("time (s)")
    n_gt   = sum(len(v) for v in gt_by_class.values())
    n_pred = sum(len(v) for v in pred_by_class.values())
    ax.set_title(f"{vid}  GT (blue, top) vs preds (red, bottom)  "
                 f"GT={n_gt}  pred={n_pred}  duration={duration:.0f}s")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="OpenTAD's log.json (text format)")
    p.add_argument("--predictions", required=True, help="canonical predictions JSON")
    p.add_argument("--annotations", required=True, help="full split annotation JSON")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-gantt", type=int, default=8,
                   help="render up to this many per-video Gantts (test set, "
                        "sampled by GT segment count)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Training curves...")
    plot_training_curves(args.log, out_dir / "training_curves.png")

    print("Predictions analysis...")
    plot_predictions_analysis(args.predictions, args.annotations,
                              out_dir / "predictions_analysis.png")

    print(f"Per-video Gantts (up to {args.max_gantt})...")
    with open(args.predictions) as f:
        preds = json.load(f).get("results", {})
    with open(args.annotations) as f:
        gt_db = json.load(f)["database"]
    test_vids = [v for v, e in gt_db.items() if e.get("subset") == "testing"]
    # Pick the videos with the most GT segments for the Gantts (most informative)
    test_vids.sort(key=lambda v: -len(gt_db[v].get("annotations", [])))
    chosen = test_vids[: args.max_gantt]
    n_done = 0
    for vid in chosen:
        if plot_gantt(vid, gt_db, preds, out_dir / f"gantt_overlay_{vid}.png"):
            n_done += 1
    print(f"  rendered {n_done} Gantts in {out_dir}")


if __name__ == "__main__":
    main()
