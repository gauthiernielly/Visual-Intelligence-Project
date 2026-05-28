"""
Produce every figure used in the TAD analysis on the 86-video evaluation subset.

Outputs (under --out-dir):
  training_curves.png         2-panel: training loss and validation loss
  predictions_analysis.png    Score, per-class and duration distributions
  per_subject.png             LCS Recall/Precision/F1 per test subject
  per_class_ap.png            Per class Average Precision at IoU 0.5
  recall_vs_complexity.png    LCS Recall vs number of GT actions per video
  showcase_bars.png           Head-to-head bar chart for the two showcase videos
  gantt_3row_<vid>.png        Per-class timeline GT vs TAD vs Hybrid for each --showcase-vid

The TAD predictions are read from --tad-pipeline-json (the output of
prefilter_tad_for_hybrid_eval.py). The Hybrid predictions are read from
--hybrid-pipeline-json. The raw OpenTAD log is parsed for the loss curves.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_RE = re.compile(
    r"\[Train\]: \[(\d+)\]\[\d+/\d+\]\s+Loss=([\d.]+)\s+cls_loss=([\d.]+)\s+reg_loss=([\d.]+)"
)
VAL_RE = re.compile(
    r"\[Val\]: \[(\d+)\]\s+Loss=([\d.]+)\s+cls_loss=([\d.]+)\s+reg_loss=([\d.]+)"
)


def norm(label):
    return str(label).strip().lower().replace(".", "_").replace(" ", "_")


def temporal_iou(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


# Training curves

def plot_training_curves(log_path, out_path):
    txt = Path(log_path).read_text()
    train = [(int(e), float(L), float(c), float(r)) for e, L, c, r in TRAIN_RE.findall(txt)]
    val = [(int(e), float(L), float(c), float(r)) for e, L, c, r in VAL_RE.findall(txt)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    if train:
        by_e = defaultdict(list)
        for e, L, c, r in train:
            by_e[e].append((L, c, r))
        es = sorted(by_e)
        Ls = [np.mean([t[0] for t in by_e[e]]) for e in es]
        cs = [np.mean([t[1] for t in by_e[e]]) for e in es]
        rs = [np.mean([t[2] for t in by_e[e]]) for e in es]
        axes[0].plot(es, Ls, "o-", color="C0", label="total")
        axes[0].plot(es, cs, "s-", color="C1", label="cls")
        axes[0].plot(es, rs, "^-", color="C2", label="reg")
        axes[0].set_title("Training loss (full)")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    if val:
        es, Ls, cs, rs = zip(*val)
        axes[1].plot(es, Ls, "o-", color="C0", label="total")
        axes[1].plot(es, cs, "s-", color="C1", label="cls")
        axes[1].plot(es, rs, "^-", color="C2", label="reg")
        axes[1].set_title("Validation loss")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("loss")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  -> {out_path}")


# Predictions analysis (86-video raw output)

def plot_predictions_analysis(canon_path, video_list_path, gt_dir, out_path, fps=25.0):
    with open(canon_path) as f:
        raw = json.load(f).get("results", {})
    target_ids = [l.strip() for l in open(video_list_path) if l.strip()]

    pred_segs = []
    for vid in target_ids:
        pred_segs.extend(raw.get(vid, []))
    n_pred = len(pred_segs)

    gt_durs = []
    n_gt = 0
    for vid in target_ids:
        subj = vid[:3]
        csv_path = Path(gt_dir) / subj / f"{vid}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                gt_durs.append((float(r["end_frame"]) - float(r["start_frame"])) / fps)
                n_gt += 1

    pred_dur = np.array([s["segment"][1] - s["segment"][0] for s in pred_segs])
    scores = np.array([s["score"] for s in pred_segs])
    pred_labels = Counter(s["label"] for s in pred_segs)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].hist(scores, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("confidence score")
    axes[0].set_ylabel("# predictions")
    axes[0].set_title(f"Score distribution (n={n_pred:,})")
    axes[0].grid(linestyle=":", alpha=0.4)

    top30 = pred_labels.most_common(30)
    y = np.arange(len(top30))
    axes[1].barh(y, [c for _, c in top30], color="seagreen", edgecolor="white")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([l for l, _ in top30], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("# predictions")
    axes[1].set_title("Top-30 predicted classes (out of 51)")
    axes[1].grid(axis="x", linestyle=":", alpha=0.4)

    bins = np.logspace(-1, 3, 50)
    axes[2].hist(gt_durs, bins=bins, color="steelblue", alpha=0.6,
                 label=f"GT (n={n_gt:,})", edgecolor="white")
    axes[2].hist(pred_dur, bins=bins, color="crimson", alpha=0.55,
                 label=f"pred (n={n_pred:,})", edgecolor="white")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("duration (s, log)")
    axes[2].set_ylabel("# segments")
    axes[2].set_title("Segment durations: GT vs predicted")
    axes[2].legend()
    axes[2].grid(linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  -> {out_path}")


# Per-subject (LCS R / P / F1 from the per-video CSV produced by complete_eval.py)

def plot_per_subject(eval_csv, out_path):
    df = pd.read_csv(eval_csv)
    g = (df.groupby("subject")
           .agg(n=("video_id", "count"),
                recall=("recall", "mean"),
                precision=("precision", "mean"),
                f1=("f1", "mean"))
           .reset_index()
           .sort_values("f1", ascending=False))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(g))
    w = 0.27
    b1 = ax.bar(x - w, g["recall"],    w, label="Recall",    color="#264653", edgecolor="white")
    b2 = ax.bar(x,     g["precision"], w, label="Precision", color="#2a9d8f", edgecolor="white")
    b3 = ax.bar(x + w, g["f1"],        w, label="F1",        color="#e76f51", edgecolor="white")
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(g["subject"], g["n"])])
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 100)
    ax.set_title("TAD baseline, LCS Recall / Precision / F1 per test subject (86 videos)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  -> {out_path}")


# Per-class AP @ IoU 0.5 on the 86-video subset

def compute_per_class_ap(tad_pipeline_json, video_list_path, gt_dir, iou_thresh=0.5, fps=25.0):
    target_ids = [l.strip() for l in open(video_list_path) if l.strip()]
    preds = json.load(open(tad_pipeline_json))["results"]
    gt = {}
    for vid in target_ids:
        csv_path = Path(gt_dir) / vid[:3] / f"{vid}.csv"
        if not csv_path.exists():
            continue
        rows = []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.append({
                    "start": float(r["start_frame"]) / fps,
                    "end":   float(r["end_frame"])   / fps,
                    "label": norm(r["event"]),
                })
        gt[vid] = rows

    all_cls = sorted({s["label"] for segs in gt.values() for s in segs})
    ap_at_t = {}
    gt_count = {}
    for cls in all_cls:
        total_gt = sum(1 for segs in gt.values() for s in segs if s["label"] == cls)
        gt_count[cls] = total_gt
        if total_gt == 0:
            continue
        cls_preds = []
        for vid, segs in preds.items():
            for s in segs:
                if norm(s["label"]) == cls:
                    cls_preds.append((float(s["score"]), vid, s["segment"][0], s["segment"][1]))
        cls_preds.sort(key=lambda x: -x[0])
        matched = {vid: [False] * sum(1 for s in segs if s["label"] == cls)
                   for vid, segs in gt.items()}
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        for i, (_, vid, ps, pe) in enumerate(cls_preds):
            gts = [s for s in gt.get(vid, []) if s["label"] == cls]
            if not gts:
                fp[i] = 1
                continue
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                iou = temporal_iou(ps, pe, g["start"], g["end"])
                if iou > best:
                    best, bj = iou, j
            if best >= iou_thresh and bj >= 0 and not matched[vid][bj]:
                tp[i] = 1
                matched[vid][bj] = True
            else:
                fp[i] = 1
        tpc = np.cumsum(tp)
        fpc = np.cumsum(fp)
        rec = tpc / total_gt
        prec = tpc / (tpc + fpc + 1e-8)
        ap = sum((prec[rec >= t].max() if (rec >= t).any() else 0.0)
                 for t in np.linspace(0, 1, 11)) / 11.0
        ap_at_t[cls] = ap * 100
    return ap_at_t, gt_count


def plot_per_class_ap(ap_dict, gt_count, out_path, min_gt=5):
    items = [(c, a) for c, a in ap_dict.items() if gt_count.get(c, 0) >= min_gt]
    items.sort(key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(13, 9))
    labels = [c.replace("_", " ") for c, _ in items]
    vals = [v for _, v in items]
    colors = [
        "#2a9d8f" if v >= 40 else "#e9c46a" if v >= 20 else "#e76f51" if v > 0 else "#bbbbbb"
        for v in vals
    ]
    ax.barh(range(len(items)), vals, color=colors, edgecolor="#222", linewidth=0.4)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Average Precision @ IoU 0.5 (%)")
    ax.set_title("TAD baseline, per-class AP @ IoU 0.5 (86-video subset)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(0, max(max(vals), 1) * 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  -> {out_path}")


# Recall vs complexity

def plot_recall_vs_complexity(eval_csv, out_path):
    df = pd.read_csv(eval_csv)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.scatter(df["gt_count"], df["recall"], s=42, alpha=0.65,
               color="#264653", edgecolor="white", linewidth=0.6)
    x = df["gt_count"].values.astype(float)
    y = df["recall"].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, slope * xs + intercept, "-", color="#e63946", lw=2,
            label=f"linear fit (slope={slope:.3f})")
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.97, 0.05, f"Pearson r = {r:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="#888", boxstyle="round,pad=0.4"))
    ax.set_xlabel("Video complexity (GT actions per video)")
    ax.set_ylabel("TAD LCS Recall (%)")
    ax.set_title("TAD baseline, recall vs video complexity (86 videos)")
    ax.legend(loc="upper right")
    ax.grid(linestyle=":", alpha=0.4)
    ax.set_xlim(0, x.max() * 1.05)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  -> {out_path}")


# Head-to-head showcase bars

def plot_showcase_bars(tad_csv, hybrid_csv, vids, out_path):
    t = pd.read_csv(tad_csv).set_index("video_id")
    h = pd.read_csv(hybrid_csv).set_index("video_id")
    fig, axes = plt.subplots(1, len(vids), figsize=(6 * len(vids), 4.8), sharey=True)
    if len(vids) == 1:
        axes = [axes]
    metrics = ["recall", "precision", "f1"]
    labels = ["Recall", "Precision", "F1"]
    x = np.arange(3)
    w = 0.36
    for ax, vid in zip(axes, vids):
        tv = [t.loc[vid, m] for m in metrics]
        hv = [h.loc[vid, m] for m in metrics]
        b1 = ax.bar(x - w / 2, tv, w, label="TAD",    color="#264653", edgecolor="white")
        b2 = ax.bar(x + w / 2, hv, w, label="Hybrid", color="#e76f51", edgecolor="white")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                        f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        gt_n = int(t.loc[vid, "gt_count"])
        ax.set_title(f"{vid}   (GT = {gt_n} actions)")
        if ax is axes[0]:
            ax.set_ylabel("Percent (%)")
    axes[0].legend(loc="upper right")
    plt.suptitle("Head-to-head per-video comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


# Per-class 3-row Gantt (GT vs TAD vs Hybrid)

def plot_gantt_3row(vid, gt_db, tad_results, hyb_results, out_path, subtitle=""):
    if vid not in gt_db:
        return False
    info = gt_db[vid]
    duration = info["duration"]
    by_gt = {}
    for s in info.get("annotations", []):
        by_gt.setdefault(s["label"], []).append(s["segment"])
    by_tad = {}
    for s in tad_results.get(vid, []):
        by_tad.setdefault(s["label"], []).append(s["segment"])
    by_hyb = {}
    for s in hyb_results.get(vid, []):
        by_hyb.setdefault(s["label"], []).append(s["segment"])

    classes = sorted(set(by_gt) | set(by_tad) | set(by_hyb))
    if not classes:
        return False

    fig, ax = plt.subplots(figsize=(14, 0.4 * max(8, len(classes))))
    BAR_H = 0.24
    MIN_W = duration * 0.002
    for i, cls in enumerate(classes):
        for s, e in by_gt.get(cls, []):
            ax.barh(i + 0.25, max(e - s, MIN_W), left=s, height=BAR_H,
                    color="steelblue", alpha=0.85)
        for s, e in by_tad.get(cls, []):
            ax.barh(i,        max(e - s, MIN_W), left=s, height=BAR_H,
                    color="crimson",   alpha=0.85)
        for s, e in by_hyb.get(cls, []):
            ax.barh(i - 0.25, max(e - s, MIN_W), left=s, height=BAR_H,
                    color="seagreen",  alpha=0.85)
        ax.text(-duration * 0.005, i, cls, ha="right", va="center", fontsize=8)

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.7, len(classes) + 0.3)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    n_gt  = sum(len(v) for v in by_gt.values())
    n_tad = sum(len(v) for v in by_tad.values())
    n_hyb = sum(len(v) for v in by_hyb.values())
    title = (f"{vid}  GT (blue, top) vs TAD (red, mid) vs Hybrid (green, bottom)  "
             f"GT={n_gt}  TAD={n_tad}  Hybrid={n_hyb}  duration={duration:.0f}s")
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=10)
    handles = [
        mpatches.Patch(color="steelblue", label="Ground Truth"),
        mpatches.Patch(color="crimson",   label="TAD prediction"),
        mpatches.Patch(color="seagreen",  label="Hybrid prediction"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  -> {out_path}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="OpenTAD's log.json (text format)")
    p.add_argument("--predictions", required=True,
                   help="predictions_canonical.json, raw TAD output before prefilter")
    p.add_argument("--annotations", required=True,
                   help="full split annotation JSON (with `database` key)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--video-list", default=None,
                   help="86-video subset id list, default outputs/hybrid_eval_86.txt")
    p.add_argument("--gt-dir", default=None,
                   help="directory of per-video GT CSVs (subject subfolders)")
    p.add_argument("--tad-pipeline-json", default=None,
                   help="prefiltered TAD predictions, default outputs/tad_pipeline_results_86.json")
    p.add_argument("--hybrid-pipeline-json", default=None,
                   help="hybrid post-VLM predictions JSON")
    p.add_argument("--tad-eval-csv", default=None,
                   help="per-video TAD eval CSV from complete_eval.py")
    p.add_argument("--hybrid-eval-csv", default=None,
                   help="per-video Hybrid eval CSV from complete_eval.py")
    p.add_argument("--showcase-vids", nargs="+",
                   default=["P20T16C03", "P14T02C04"],
                   help="video ids for the showcase Gantts and bar chart")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Training curves...")
    plot_training_curves(args.log, out_dir / "training_curves.png")

    if args.video_list and args.gt_dir:
        print("Predictions analysis (86-video subset)...")
        plot_predictions_analysis(args.predictions, args.video_list,
                                  args.gt_dir, out_dir / "predictions_analysis.png")

    if args.tad_eval_csv:
        print("Per-subject breakdown...")
        plot_per_subject(args.tad_eval_csv, out_dir / "per_subject.png")
        print("Recall vs complexity...")
        plot_recall_vs_complexity(args.tad_eval_csv, out_dir / "recall_vs_complexity.png")

    if args.tad_pipeline_json and args.video_list and args.gt_dir:
        print("Per-class AP @ IoU 0.5...")
        ap, gtc = compute_per_class_ap(args.tad_pipeline_json, args.video_list, args.gt_dir)
        plot_per_class_ap(ap, gtc, out_dir / "per_class_ap.png")

    if args.tad_eval_csv and args.hybrid_eval_csv:
        print("Showcase head-to-head bars...")
        plot_showcase_bars(args.tad_eval_csv, args.hybrid_eval_csv,
                           args.showcase_vids, out_dir / "showcase_bars.png")

    if args.tad_pipeline_json and args.hybrid_pipeline_json:
        print(f"3-row Gantts for {args.showcase_vids}...")
        with open(args.annotations) as f:
            gt_db = json.load(f)["database"]
        tad_res = json.load(open(args.tad_pipeline_json))["results"]
        hyb_res = json.load(open(args.hybrid_pipeline_json))["results"]
        for vid in args.showcase_vids:
            plot_gantt_3row(vid, gt_db, tad_res, hyb_res,
                            out_dir / f"gantt_3row_{vid}.png")


if __name__ == "__main__":
    main()
