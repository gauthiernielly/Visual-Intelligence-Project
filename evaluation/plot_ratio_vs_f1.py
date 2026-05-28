"""
Scatter plot of pred/GT ratio vs LCS F1 for TAD and Hybrid systems.
Reveals whether over-prediction helps or hurts, and compares the two systems.

Usage
  python plot_ratio_vs_f1.py \
      --tad    ../TAD_full_run/outputs/predictions_canonical.json \
      --hybrid ../results/hybrid_pipeline_results.json \
      --gt     ../../../../../work/cs-503/sadgal/Annotation \
      --fps    25 \
      --out    ../graphs/ratio_vs_f1.png
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict

matplotlib.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

DEFAULT_FPS  = 25.0
MIN_CONF_TAD = 0.15
NMS_IOU      = 0.3


def normalise(label: str) -> str:
    return str(label).strip().lower().replace(".", "_").replace(" ", "_")


def lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def collapse(seq: list) -> list:
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def nms(segments: list, iou_thresh: float = NMS_IOU) -> list:
    by_class = defaultdict(list)
    for s in segments:
        by_class[s["label"]].append(s)
    kept = []
    for cls_segs in by_class.values():
        cls_segs = sorted(cls_segs, key=lambda x: -x.get("score", 1.0))
        survivors = []
        for cand in cls_segs:
            cs, ce = cand["segment"]
            ok = True
            for s in survivors:
                ss, se = s["segment"]
                inter = max(0.0, min(ce, se) - max(cs, ss))
                union = (ce - cs) + (se - ss) - inter
                if union > 0 and inter / union > iou_thresh:
                    ok = False
                    break
            if ok:
                survivors.append(cand)
        kept.extend(survivors)
    return kept


def load_gt_sequences(annotations_dir: str, video_ids: list, fps: float) -> dict:
    gt = {}
    for vid in video_ids:
        subject = vid[:3]
        path = os.path.join(annotations_dir, subject, f"{vid}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path).sort_values("start_frame")
            _fps = df["fps"].iloc[0] if "fps" in df.columns else fps
            seq = collapse([normalise(str(r["event"])) for _, r in df.iterrows()])
            if seq:
                gt[vid] = seq
        except Exception as e:
            warnings.warn(f"Could not load GT for {vid}: {e}")
    return gt


def pred_sequence(segs: list) -> list:
    segs = sorted(segs, key=lambda x: x["segment"][0])
    return collapse([normalise(s["label"]) for s in segs])


def compute_metrics_df(pred_data: dict, gt_seqs: dict, name: str, filter_tad: bool = False) -> pd.DataFrame:
    rows = []
    for vid, gt_seq in gt_seqs.items():
        raw = pred_data.get(vid, [])
        if filter_tad:
            raw = [s for s in raw if s.get("score", 0) >= MIN_CONF_TAD]
            raw = nms(raw)

        pred_seq = pred_sequence(raw)

        if not gt_seq or not pred_seq:
            continue

        lcs = lcs_length(gt_seq, pred_seq)
        recall    = lcs / len(gt_seq)    * 100
        precision = lcs / len(pred_seq)  * 100
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        ratio = len(pred_seq) / len(gt_seq)

        rows.append({
            "video_id": vid,
            "system":   name,
            "gt_len":   len(gt_seq),
            "pred_len": len(pred_seq),
            "ratio":    ratio,
            "recall":   recall,
            "precision": precision,
            "f1":       f1,
        })
    return pd.DataFrame(rows)



def plot_ratio_vs_f1(df_tad: pd.DataFrame, df_hybrid: pd.DataFrame, out_path: str) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    systems = [
        (df_tad,    "TAD",    "#5B4FCF", axes[0]),
        (df_hybrid, "Hybrid", "#0F6E56", axes[1]),
    ]

    for df, name, color, ax in systems:
        ratios = df["ratio"].values
        f1s    = df["f1"].values

        sc = ax.scatter(
            ratios, f1s,
            c=color, alpha=0.72, s=60,
            edgecolors="white", linewidths=0.5,
            zorder=3,
        )

        if len(ratios) > 2:
            z = np.polyfit(ratios, f1s, 1)
            p = np.poly1d(z)
            x_line = np.linspace(ratios.min(), ratios.max(), 100)
            ax.plot(x_line, p(x_line), color=color, linewidth=1.5,
                    linestyle="--", alpha=0.6, zorder=2)

        ax.axvline(x=1, color="#999", linewidth=1, linestyle=":", zorder=1)
        ax.text(1.02, ax.get_ylim()[0] + 2 if ax.get_ylim()[0] > 0 else 2,
                "pred = GT", fontsize=7.5, color="#888", va="bottom")

        ax.axhline(y=df["f1"].mean(), color=color, linewidth=1,
                   linestyle="-.", alpha=0.4, zorder=1)
        ax.text(
            ratios.max() * 0.97, df["f1"].mean() + 1,
            f"mean F1={df['f1'].mean():.1f}%",
            fontsize=8, color=color, ha="right", alpha=0.8,
        )

        ax.set_xlabel("Pred / GT sequence length ratio", fontsize=10)
        ax.set_title(
            f"{name}  (n={len(df)})\n"
            f"Recall {df['recall'].mean():.1f}%  ·  "
            f"Precision {df['precision'].mean():.1f}%  ·  "
            f"F1 {df['f1'].mean():.1f}%",
            fontsize=10, color=color, loc="left",
        )
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, color="#bbb")
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4, color="#bbb")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("LCS F1 Score (%)", fontsize=10)

    fig.suptitle(
        "Prediction density vs. LCS F1 — does over-predicting help?",
        fontsize=12, fontweight="bold", y=1.01, color="#1A1A2E",
    )

    fig.text(
        0.5, -0.04,
        "Points to the right of the dotted line have more predictions than GT actions. "
        "A downward trend means over-prediction hurts F1.",
        ha="center", fontsize=9, color="#555", style="italic",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Scatter saved → {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tad",    required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--gt",     required=True)
    parser.add_argument("--fps",    type=float, default=DEFAULT_FPS)
    parser.add_argument("--out",    default="ratio_vs_f1.png")
    args = parser.parse_args()

    with open(args.tad) as f:
        tad_data = json.load(f)["results"]
    with open(args.hybrid) as f:
        hybrid_data = json.load(f)["results"]

    all_video_ids = sorted(set(tad_data.keys()) | set(hybrid_data.keys()))
    print(f"Loading GT for {len(all_video_ids)} videos...")
    gt_seqs = load_gt_sequences(args.gt, all_video_ids, args.fps)
    print(f"GT loaded for {len(gt_seqs)} videos.")

    df_tad    = compute_metrics_df(tad_data,    gt_seqs, "TAD",    filter_tad=True)
    df_hybrid = compute_metrics_df(hybrid_data, gt_seqs, "Hybrid", filter_tad=False)

    print(f"\nTAD:    {len(df_tad)} videos scored")
    print(f"Hybrid: {len(df_hybrid)} videos scored")

    plot_ratio_vs_f1(df_tad, df_hybrid, args.out)


if __name__ == "__main__":
    main()
