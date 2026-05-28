"""
Gantt-style timeline comparing GT, TAD, and Hybrid predictions for a video.

Usage
-----
  python plot_gantt.py \
      --tad   ../TAD_full_run/outputs/predictions_canonical.json \
      --hybrid ../results/hybrid_pipeline_results.json \
      --gt    ../../../../../work/cs-503/sadgal/Annotation \
      --video videoID \
      --fps   25 \
      --out   ../graphs/gantt_{videoID}.png

  # Auto-pick the most readable video (medium complexity):
  python plot_gantt.py \
      --tad   predictions_canonical.json \
      --hybrid hybrid_pipeline_results.json \
      --gt    ../../../../../work/cs-503/sadgal/Annotation \
      --auto \
      --out   ../graphs/gantt.png
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
})

DEFAULT_FPS  = 25.0
MIN_CONF_TAD = 0.15
NMS_IOU      = 0.3


ROW_COLORS = {
    "GT":     "#2D2D2D",
    "TAD":    "#5B4FCF",
    "Hybrid": "#0F6E56",
}
ROW_ALPHA = {"GT": 0.85, "TAD": 0.65, "Hybrid": 0.65}


def load_gt(annotations_dir: str, video_id: str, fps: float) -> list:
    subject = video_id[:3]
    path = os.path.join(annotations_dir, subject, f"{video_id}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"GT CSV not found: {path}")
    df = pd.read_csv(path).sort_values("start_frame")
    if "end_frame" not in df.columns:
        raise ValueError("GT CSV must have start_frame and end_frame columns.")
    _fps = df["fps"].iloc[0] if "fps" in df.columns else fps
    segs = []
    for _, row in df.iterrows():
        segs.append({
            "start": row["start_frame"] / _fps,
            "end":   row["end_frame"]   / _fps,
            "label": str(row["event"]).strip(),
        })
    return segs


def load_tad(path: str, video_id: str) -> list:
    with open(path) as f:
        data = json.load(f)
    raw = data["results"].get(video_id, [])
    # we apply confidence filter + NMS (to mirror the pipeline hybrid.py)
    segs = [s for s in raw if s.get("score", 0) >= MIN_CONF_TAD]
    segs = _nms(segs)
    return [{"start": s["segment"][0], "end": s["segment"][1],
             "label": s["label"], "score": s.get("score", 1.0)} for s in segs]


def load_hybrid(path: str, video_id: str) -> list:
    with open(path) as f:
        data = json.load(f)
    segs = data["results"].get(video_id, [])
    return [{"start": s["segment"][0], "end": s["segment"][1],
             "label": s["label"], "score": s.get("score", 1.0)} for s in segs]


def _nms(segments: list, iou_thresh: float = NMS_IOU) -> list:
    from collections import defaultdict
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



def build_label_colors(all_labels: list) -> dict:
    cmap = plt.get_cmap("tab20")
    unique = sorted(set(all_labels))
    return {lbl: cmap(i % 20) for i, lbl in enumerate(unique)}


def auto_select_video(tad_path: str, hybrid_path: str, annotations_dir: str, fps: float) -> str:
    """Pick the video with the most readable Gantt (hybrid 50-120 segs, GT accessible)."""
    with open(hybrid_path) as f:
        h = json.load(f)
    candidates = []
    for vid, segs in h["results"].items():
        subject = vid[:3]
        csv_path = os.path.join(annotations_dir, subject, f"{vid}.csv")
        if os.path.exists(csv_path) and 50 <= len(segs) <= 120:
            candidates.append((vid, len(segs)))
    if not candidates:
        # Relax constraint
        for vid, segs in h["results"].items():
            subject = vid[:3]
            csv_path = os.path.join(annotations_dir, subject, f"{vid}.csv")
            if os.path.exists(csv_path):
                candidates.append((vid, len(segs)))
    if not candidates:
        raise RuntimeError("No valid video found. Check --gt path.")
    # Pick the one closest to 80 segments
    candidates.sort(key=lambda x: abs(x[1] - 80))
    chosen = candidates[0][0]
    print(f"[auto] Selected video: {chosen} ({candidates[0][1]} hybrid segments)")
    return chosen


def plot_gantt(
    gt_segs:     list,
    tad_segs:    list,
    hybrid_segs: list,
    video_id:    str,
    out_path:    str,
) -> None:

    all_labels = (
        [s["label"] for s in gt_segs] +
        [s["label"] for s in tad_segs] +
        [s["label"] for s in hybrid_segs]
    )
    label_colors = build_label_colors(all_labels)

    rows = [
        ("GT",     gt_segs),
        ("TAD",    tad_segs),
        ("Hybrid", hybrid_segs),
    ]

    all_ends = (
        [s["end"] for s in gt_segs] +
        [s["end"] for s in tad_segs] +
        [s["end"] for s in hybrid_segs]
    )
    duration = max(all_ends) if all_ends else 600
    duration_min = duration / 60

    fig_width = min(22, max(14, duration_min * 0.9))
    fig, ax = plt.subplots(figsize=(fig_width, 3.6))

    bar_height = 0.55
    y_positions = {name: i for i, (name, _) in enumerate(rows)}

    for row_name, segs in rows:
        y = y_positions[row_name]
        color_base = ROW_COLORS[row_name]
        alpha = ROW_ALPHA[row_name]

        for seg in segs:
            start_m = seg["start"] / 60
            width_m = (seg["end"] - seg["start"]) / 60
            if width_m <= 0:
                continue
            face = label_colors[seg["label"]]
            ax.barh(
                y, width_m, left=start_m, height=bar_height,
                color=face, alpha=alpha, linewidth=0,
            )
            ax.barh(
                y, width_m, left=start_m, height=bar_height,
                fill=False, edgecolor=color_base, linewidth=0.25, alpha=0.5,
            )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(
        [name for name, _ in rows],
        fontsize=11, fontweight="bold",
    )
    for tick, (name, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(ROW_COLORS[name])

    ax.set_xlim(0, duration_min)
    ax.set_xlabel("Time (minutes)", fontsize=10)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4, color="#999")
    ax.set_axisbelow(True)

    stats_text = (
        f"GT: {len(gt_segs)} actions   "
        f"TAD: {len(tad_segs)} predictions   "
        f"Hybrid: {len(hybrid_segs)} predictions   "
        f"Duration: {duration_min:.1f} min"
    )
    ax.set_title(
        f"Action Timeline — {video_id}\n"
        f"{stats_text}",
        fontsize=10.5, loc="left", pad=10,
        color="#333",
    )

    from collections import Counter
    top_labels = [lbl for lbl, _ in Counter(all_labels).most_common(12)]
    legend_patches = [
        mpatches.Patch(color=label_colors[lbl], label=lbl.replace("_", " ").replace(".", " · "))
        for lbl in top_labels
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        title="Action classes (top 12 by frequency)",
        title_fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Gantt saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tad",    required=True, help="TAD predictions JSON")
    parser.add_argument("--hybrid", required=True, help="Hybrid predictions JSON")
    parser.add_argument("--gt",     required=True, help="Annotations directory")
    parser.add_argument("--video",  default=None,  help="Specific video ID")
    parser.add_argument("--auto",   action="store_true", help="Auto-select best video")
    parser.add_argument("--fps",    type=float, default=DEFAULT_FPS)
    parser.add_argument("--out",    default="gantt.png")
    args = parser.parse_args()

    if args.auto or args.video is None:
        video_id = auto_select_video(args.tad, args.hybrid, args.gt, args.fps)
    else:
        video_id = args.video

    print(f"Loading GT for {video_id}...")
    gt_segs     = load_gt(args.gt, video_id, args.fps)
    tad_segs    = load_tad(args.tad, video_id)
    hybrid_segs = load_hybrid(args.hybrid, video_id)

    print(f"  GT={len(gt_segs)}  TAD={len(tad_segs)}  Hybrid={len(hybrid_segs)}")
    plot_gantt(gt_segs, tad_segs, hybrid_segs, video_id, args.out)


if __name__ == "__main__":
    main()
