import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np



def plot_recall_by_duration(
    detection_by_duration: list[dict],
    output_dir: str,
) -> None:
    """
    Saves a vertical bar chart of detection recall per duration bin to
    <output_dir>/figures/recall_by_duration.png.
    """
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels   = [row["bin"] for row in detection_by_duration]
    totals   = [row["total"] for row in detection_by_duration]
    recalls  = [row["recall"] if row["recall"] is not None else 0.0 for row in detection_by_duration]

    n_bins = len(labels)
    x = np.arange(n_bins)

    # Gradient from light to dark blue (short events → light, long → dark)
    cmap   = plt.cm.Blues
    colors = [cmap(0.25 + 0.65 * i / max(n_bins - 1, 1)) for i in range(n_bins)]

    # Weighted mean recall across all bins
    total_detected = sum(row["detected"] for row in detection_by_duration)
    total_gt       = sum(row["total"]    for row in detection_by_duration)
    mean_recall    = total_detected / total_gt if total_gt > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(x, recalls, color=colors, width=0.6, zorder=3)

    # Per-bar annotations
    for bar, recall, total in zip(bars, recalls, totals):
        h = bar.get_height()
        # Recall value just above bar
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.05,
            f"{recall:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
        # GT count below the recall label
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"n={total}",
            ha="center", va="bottom", fontsize=8, color="#666666",
        )

    # Weighted-mean dashed line
    ax.axhline(mean_recall, color="red", linestyle="--", linewidth=0.6, zorder=4)
    ax.text(
        n_bins - 0.5 + 0.1,
        mean_recall + 0.02,
        "mean",
        ha="left", va="bottom", fontsize=9, color="red",
    )

    # Axes
    ax.set_xlim(-0.5, n_bins - 0.5 + 0.8)   # extra right margin for "mean" label
    ax.set_ylim(0, max(recalls) * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Event duration", fontsize=11, labelpad=8)
    ax.set_ylabel("Recall @ IoU ≥ 0.1", fontsize=11, labelpad=8)
    ax.set_title("Detection Recall by Event Duration", fontsize=13, pad=12)

    # Clean minimal style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="lightgray", linestyle="-", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = fig_dir / "recall_by_duration.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    # ── Interactive Plotly version ─────────────────────────────────────────────
    import plotly.graph_objects as go

    colors_hex = [
        mcolors.to_hex(cmap(0.25 + 0.65 * i / max(n_bins - 1, 1)))
        for i in range(n_bins)
    ]
    cd = [[row["total"], row["detected"]] for row in detection_by_duration]

    pfig = go.Figure()
    pfig.add_trace(go.Bar(
        x=labels,
        y=recalls,
        marker=dict(color=colors_hex),
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Recall: %{y:.4f}<br>"
            "GT instances: %{customdata[0]}<br>"
            "Detected: %{customdata[1]}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))
    pfig.add_hline(
        y=mean_recall,
        line_dash="dash",
        line_color="red",
        line_width=1,
        annotation_text="mean",
        annotation_position="top right",
        annotation_font_color="red",
    )
    pfig.update_layout(
        title="Detection Recall by Event Duration",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Event duration"),
        yaxis=dict(title="Recall @ IoU ≥ 0.1",
                   range=[0, max(recalls) * 1.25],
                   showgrid=True, gridcolor="#e0e0e0"),
        margin=dict(t=60, b=50, l=60, r=30),
    )
    html_path = fig_dir / "recall_by_duration.html"
    pfig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"Figure saved to {html_path}")


def plot_per_class_recall(
    per_class_recall: dict[str, "float | None"],
    all_gts: dict[str, list[dict]],
    output_dir: str,
) -> None:
    """
    Horizontal bar chart of per-class recall, sorted descending.
    Excludes classes with no GT instances (recall == None).
    Saved to <output_dir>/figures/per_class_recall.html.
    """
    import plotly.graph_objects as go

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # GT count per class (to populate hover n_detected / n_gt)
    n_gt_by_cls: dict[str, int] = defaultdict(int)
    for segs in all_gts.values():
        for s in segs:
            n_gt_by_cls[s["event"]] += 1

    # Filter: keep only classes with GT instances AND recall > 0
    all_with_gt = [(cls, r, n_gt_by_cls[cls])
                   for cls, r in per_class_recall.items() if r is not None]
    n_zero = sum(1 for _, r, _ in all_with_gt if r == 0.0)
    mean_recall = sum(r for _, r, _ in all_with_gt) / len(all_with_gt) if all_with_gt else 0.0
    rows = [(cls, r, n) for cls, r, n in all_with_gt if r > 0.0]
    rows.sort(key=lambda r: r[1])   # ascending → top of horizontal chart = highest

    if not rows:
        return

    classes  = [r[0] for r in rows]
    recalls  = [r[1] for r in rows]
    n_gts    = [r[2] for r in rows]
    n_det    = [int(round(r * n)) for r, n in zip(recalls, n_gts)]

    zero_note = f" — {n_zero} class{'es' if n_zero != 1 else ''} with recall = 0 not shown"
    title = f"Per-Class Recall (VLM Baseline){zero_note}"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        orientation="h",
        y=classes,
        x=recalls,
        marker=dict(
            color=recalls,
            colorscale=[[0.0, "#ffcccc"], [1.0, "#990000"]],
            cmin=0,
            cmax=1,
        ),
        customdata=[[nd, ng] for nd, ng in zip(n_det, n_gts)],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Recall: %{x:.1%}<br>"
            "Detected: %{customdata[0]}<br>"
            "GT instances: %{customdata[1]}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_vline(
        x=mean_recall,
        line_dash="dash",
        line_color="#555",
        line_width=1.2,
        annotation_text="mean",
        annotation_position="top",
        annotation_font_size=11,
    )
    fig.update_layout(
        title=title,
        height=max(400, len(classes) * 20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title="Recall @ IoU ≥ 0.05",
            range=[0, max(recalls) * 1.25],
            showgrid=True,
            gridcolor="#e0e0e0",
        ),
        yaxis=dict(
            tickfont=dict(size=9),
            showgrid=False,
        ),
        margin=dict(l=200, r=30, t=60, b=50),
    )

    output_path = fig_dir / "per_class_recall.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Figure saved to {output_path}")


# ── Clip definitions ─────────────────────────────────────────────────────────

_CLIP_WINDOWS: dict[str, tuple[int, int]] = {
    "P10T11C03": (22875, 24375),
    "P11T03C04": (18250, 19750),
    "P14T05C05": (9000, 10500),
    "P16T14C06": (12125, 13625),
    "P18T11C02": (19125, 20625),
}

_VIDEO_LABELS: dict[str, str] = {
    "P10T11C03": "Kitchen",
    "P11T03C04": "Couch",
    "P14T05C05": "TV",
    "P16T14C06": "Cooking",
    "P18T11C02": "Table",
}

_FPS = 25


def _fmt_mmss(sec: float) -> str:
    sec = max(0.0, sec)
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def _clip_segs(raw: list[dict], clip_start: int, clip_end: int) -> list[dict]:
    out = []
    for s in raw:
        if s["end_frame"] <= clip_start or s["start_frame"] >= clip_end:
            continue
        sf = max(s["start_frame"], clip_start)
        ef = min(s["end_frame"], clip_end)
        t0 = (sf - clip_start) / _FPS
        t1 = (ef - clip_start) / _FPS
        out.append({"event": s["event"], "start_sec": t0, "end_sec": t1,
                    "duration_sec": t1 - t0})
    return out


def _assign_lanes(segs: list[dict], track_label: str) -> int:
    """
    Greedy interval scheduling: assigns 'lane_label' to each segment in-place.
    Returns the total number of lanes used.
    """
    if not segs:
        return 0
    lane_ends: list[float] = []
    for seg in sorted(segs, key=lambda s: s["start_sec"]):
        placed = False
        for i, end in enumerate(lane_ends):
            if seg["start_sec"] >= end:
                seg["_lane"] = i
                lane_ends[i] = seg["end_sec"]
                placed = True
                break
        if not placed:
            seg["_lane"] = len(lane_ends)
            lane_ends.append(seg["end_sec"])
    n = max(s["_lane"] for s in segs) + 1
    for s in segs:
        s["lane_label"] = f"{track_label} — lane {s['_lane'] + 1}"
    return n


def plot_timeline_viewer(
    all_gts: dict[str, list[dict]],
    all_preds: dict[str, list[dict]],
    output_dir: str,
) -> None:
    """
    Generates an interactive HTML timeline comparing GT and VLM predictions
    for five pre-defined 60-second clip windows.
    Saved to <output_dir>/figures/timeline_viewer.html.
    """
    import plotly.graph_objects as go
    import plotly.express as px

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Color palette (consistent across videos and tracks) ───────────────────
    all_classes = sorted({
        s["event"]
        for segs_list in [*all_gts.values(), *all_preds.values()]
        for s in segs_list
    })
    palette = px.colors.qualitative.Alphabet
    class_color = {cls: palette[i % len(palette)] for i, cls in enumerate(all_classes)}

    # ── Per-video lane assignment ─────────────────────────────────────────────
    video_info: dict[str, dict] = {}
    for vid, (cs, ce) in _CLIP_WINDOWS.items():
        lbl  = _VIDEO_LABELS[vid]
        gt   = _clip_segs(all_gts.get(vid, []),   cs, ce)
        pred = _clip_segs(all_preds.get(vid, []),  cs, ce)
        n_gt   = _assign_lanes(gt,   f"{lbl} — GT")
        n_pred = _assign_lanes(pred, f"{lbl} — VLM")
        # categoryarray goes bottom-to-top → VLM lanes at bottom, GT lanes at top
        y_cats = (
            [f"{lbl} — VLM — lane {i + 1}" for i in range(n_pred)] +
            [f"{lbl} — GT — lane {i + 1}"  for i in range(n_gt)]
        )
        video_info[vid] = {
            "gt": gt, "pred": pred,
            "y_cats": y_cats,
            "n_gt": n_gt, "n_pred": n_pred,
        }

    # ── Build traces (one go.Bar per video × class) ───────────────────────────
    # NOTE: no Scatter legend-only traces — they force the y-axis to linear mode,
    # preventing categorical lane labels from rendering. showlegend=True on each
    # Bar trace + legendgroup deduplication gives one legend entry per class.
    traces: list = []
    vid_trace_idx: dict[str, list[int]] = {v: [] for v in _CLIP_WINDOWS}
    first_vid = next(iter(_CLIP_WINDOWS))

    for vid, info in video_info.items():
        is_first = vid == first_vid
        segs_by_cls: dict[str, list[dict]] = defaultdict(list)
        for s in info["gt"] + info["pred"]:
            segs_by_cls[s["event"]].append(s)

        for cls, segs in segs_by_cls.items():
            cd = [
                [s["event"], _fmt_mmss(s["start_sec"]),
                 _fmt_mmss(s["end_sec"]), round(s["duration_sec"], 1)]
                for s in segs
            ]
            traces.append(go.Bar(
                orientation="h",
                y=[s["lane_label"]   for s in segs],
                x=[s["duration_sec"] for s in segs],
                base=[s["start_sec"] for s in segs],
                marker=dict(color=class_color[cls],
                            line=dict(width=0.5, color="white")),
                name=cls,
                legendgroup=cls,
                showlegend=True,
                visible=is_first,
                customdata=cd,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Start: %{customdata[1]}<br>"
                    "End:   %{customdata[2]}<br>"
                    "Duration: %{customdata[3]}s"
                    "<extra></extra>"
                ),
            ))
            vid_trace_idx[vid].append(len(traces) - 1)

    # ── Helpers: separator line + y-axis tick labels ──────────────────────────
    def _sep_shape(n_pred: int, n_gt: int) -> list[dict]:
        if n_pred > 0 and n_gt > 0:
            return [dict(
                type="line", xref="paper", x0=0, x1=1,
                yref="y", y0=n_pred - 0.5, y1=n_pred - 0.5,
                line=dict(color="#666", width=1.5, dash="dot"),
            )]
        return []
    
    def _tick_labels(lbl: str, n_pred: int, n_gt: int) -> tuple[list, list]:
        """Returns (tickvals, ticktext) for the two group labels."""
        tickvals, ticktext = [], []
        if n_pred > 0:
            mid = (n_pred - 1) // 2 + 1
            tickvals.append(f"{lbl} — VLM — lane {mid}")
            ticktext.append("Qwen3-VL output")
        if n_gt > 0:
            mid = (n_gt - 1) // 2 + 1
            tickvals.append(f"{lbl} — GT — lane {mid}")
            ticktext.append("Ground truth")
        return tickvals, ticktext

    # ── Dropdown buttons ──────────────────────────────────────────────────────
    buttons = []
    for vid, info in video_info.items():
        vis = [False] * len(traces)
        for idx in vid_trace_idx[vid]:
            vis[idx] = True
        lbl    = _VIDEO_LABELS[vid]
        y_cats = info["y_cats"]
        tv, tt = _tick_labels(lbl, info["n_pred"], info["n_gt"])
        buttons.append(dict(
            label=lbl,
            method="update",
            args=[
                {"visible": vis},
                {
                    "xaxis.range":          [0, 60],
                    "yaxis.type":           "category",
                    "yaxis.categoryarray":  y_cats,
                    "yaxis.range":          [-0.5, len(y_cats) - 0.5],
                    "yaxis.tickvals":       tv,
                    "yaxis.ticktext":       tt,
                    "shapes":               _sep_shape(info["n_pred"], info["n_gt"]),
                    "title.text":           f"Timeline Viewer — {lbl}",
                },
            ],
        ))

    # ── Initial layout (first video) ──────────────────────────────────────────
    fst      = video_info[first_vid]
    fst_lbl  = _VIDEO_LABELS[first_vid]
    max_lanes = max(len(info["y_cats"]) for info in video_info.values())
    fst_tv, fst_tt = _tick_labels(fst_lbl, fst["n_pred"], fst["n_gt"])

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Timeline Viewer — {fst_lbl}",
        height=max(500, 44 * max_lanes + 200),
        barmode="overlay",
        bargap=0.25,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title="Time (seconds)",
            range=[0, 60],
            showgrid=True,
            gridcolor="#e0e0e0",
            dtick=5,
        ),
        yaxis=dict(
            type="category",           # explicit — prevents Plotly from delaying
            categoryarray=fst["y_cats"],  # categorical-axis detection until interaction
            categoryorder="array",
            range=[-0.5, len(fst["y_cats"]) - 0.5],
            tickvals=fst_tv,
            ticktext=fst_tt,
        ),
        shapes=_sep_shape(fst["n_pred"], fst["n_gt"]),
        legend=dict(title="Event class", tracegroupgap=0),
        updatemenus=[dict(
            type="dropdown",
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0, xanchor="left",
            y=1.13, yanchor="top",
        )],
        margin=dict(t=130, l=160, r=30, b=60),
    )

    output_path = fig_dir / "timeline_viewer.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Timeline viewer saved to {output_path}")