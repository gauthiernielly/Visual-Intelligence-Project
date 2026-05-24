"""
tad_lcs_eval.py
===============
LCS-recall evaluation for the TAD baseline, matched to the Hybrid metrics
hybrid_eval.py uses, so the TAD-only, VLM-only, and Hybrid numbers are
directly comparable.

Differences vs the hybrid pipeline's eval script:
  - Reads ground truth from data_cs_split.json (or tsu_cs_full.json) instead
    of the per-subject CSV folder. Removes the dependency on the cluster path
    "../../../../work/cs-503/sadgal/Annotation".
  - Adds --min-score and --topk filters because TAD outputs are dense
    (~1000 segments per test video). Without a filter, LCS recall is
    artificially inflated because any short GT sequence is trivially a
    subsequence of a very long prediction sequence.

LCS-recall formula is identical to Hybrid's:
    sort segments by start time, collapse consecutive duplicate labels,
    then  recall = LCS(gt_labels, pred_labels) / len(gt_labels) * 100

Usage:
    python tad_lcs_eval.py \\
        --predictions   predictions_canonical.json \\
        --annotations   data_cs_split.json \\
        --subset        testing \\
        --min-score     0.1 \\
        --out-csv       tad_lcs_results.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def clean_label(label: str) -> str:
    """Same normalisation as the hybrid pipeline's eval script."""
    return str(label).strip().lower().replace(".", "_").replace(" ", "_")


def gt_sequence_from_json(gt_db: dict, video_id: str) -> list:
    """Return chronological label sequence from data_cs_split.json-style GT."""
    if video_id not in gt_db:
        return []
    anns = gt_db[video_id].get("annotations", [])
    anns = sorted(anns, key=lambda a: a["segment"][0])
    return [clean_label(a["label"]) for a in anns]


def pred_sequence(segments: list, min_score: float, topk: int | None) -> list:
    """Filter, sort, collapse-consecutive-duplicates."""
    segs = [s for s in segments if float(s["score"]) >= min_score]
    if topk is not None and len(segs) > topk:
        segs = sorted(segs, key=lambda s: -s["score"])[:topk]
    segs = sorted(segs, key=lambda s: s["segment"][0])
    raw = [clean_label(s["label"]) for s in segs]
    out = []
    for x in raw:
        if not out or out[-1] != x:
            out.append(x)
    return out


def load_video_list(spec: str) -> set:
    """
    Resolve `spec` into a set of video IDs. Accepted forms:
      - A comma-separated string: "P02T08C04,P10T03C05"
      - A path to a plain text file (one ID per line)
      - A path to a CSV with a column named 'video_id'
      - A path to a JSON file in the {"results": {video_id: ...}} schema
        (handy: pass the hybrid pipeline's own output JSON directly)
    """
    if "," in spec and not Path(spec).exists():
        return {x.strip() for x in spec.split(",") if x.strip()}
    path = Path(spec)
    if not path.exists():
        raise SystemExit(f"--video-list: file not found and not a CSV string: {spec}")
    text = path.read_text().strip()
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        ids = (obj.get("results") or obj).keys()
        return set(ids)
    if path.suffix.lower() == ".csv":
        ids = set()
        lines = text.splitlines()
        header = lines[0].split(",")
        try:
            col = header.index("video_id")
        except ValueError:
            raise SystemExit(
                f"--video-list: CSV {spec} has no 'video_id' column. "
                f"Header was: {header}"
            )
        for line in lines[1:]:
            cells = line.split(",")
            if col < len(cells):
                ids.add(cells[col].strip())
        return {x for x in ids if x}
    # plain text, one per line
    return {l.strip() for l in text.splitlines() if l.strip()}


def lcs_recall(gt: list, pred: list) -> float:
    """Exact same LCS DP as the hybrid pipeline's eval, returning recall percentage."""
    if not gt:
        return 0.0
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return (dp[m][n] / len(gt)) * 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="path to predictions_canonical.json (the TAD output)")
    p.add_argument("--annotations", required=True,
                   help="path to data_cs_split.json or tsu_cs_full.json (the GT)")
    p.add_argument("--subset", default="testing",
                   help="evaluate only videos with this subset label "
                        "(testing / validation / training). Default: testing. "
                        "Ignored if --video-list is provided.")
    p.add_argument("--video-list", default=None,
                   help="restrict evaluation to these video IDs. Accepts "
                        "either a path to a file (one ID per line, OR a CSV "
                        "with a 'video_id' column, OR a JSON file of the "
                        "{\"results\": {video_id: ...}} schema) OR a "
                        "comma-separated list directly. Use this to match the "
                        "exact set the hybrid pipeline evaluates on.")
    p.add_argument("--min-score", type=float, default=0.1,
                   help="drop predictions with score below this threshold "
                        "before computing LCS. Default 0.1.")
    p.add_argument("--topk", type=int, default=None,
                   help="optionally keep only the top-K highest-score "
                        "predictions per video before sorting by time.")
    p.add_argument("--out-csv", default=None,
                   help="optional path to dump per-video results as CSV.")
    args = p.parse_args()

    print(f"Loading predictions from {args.predictions}")
    with open(args.predictions) as f:
        pred_data = json.load(f)
    pred_results = pred_data.get("results", pred_data)

    print(f"Loading ground truth from {args.annotations}")
    with open(args.annotations) as f:
        gt_data = json.load(f)
    gt_db = gt_data.get("database", gt_data)

    if args.video_list:
        target_vids = sorted(load_video_list(args.video_list))
        print(f"Restricted to {len(target_vids)} videos from --video-list")
        # Sanity: warn if any requested video isn't in the GT DB
        missing_in_gt = [v for v in target_vids if v not in gt_db]
        if missing_in_gt:
            print(f"  [warn] {len(missing_in_gt)} requested videos not found in GT: "
                  f"{missing_in_gt[:5]}{'...' if len(missing_in_gt) > 5 else ''}")
    else:
        target_vids = [v for v, e in gt_db.items() if e.get("subset") == args.subset]
        print(f"Found {len(target_vids)} videos in subset '{args.subset}'")
    print(f"Predictions JSON has {len(pred_results)} videos")
    print(f"Filtering predictions by score >= {args.min_score}"
          + (f" + top-{args.topk} per video" if args.topk else ""))
    print()

    rows = []
    for vid in sorted(target_vids):
        gt_seq = gt_sequence_from_json(gt_db, vid)
        pred_segs = pred_results.get(vid, [])
        pred_seq = pred_sequence(pred_segs, args.min_score, args.topk)

        if not gt_seq:
            status, recall = "empty GT", None
        elif vid not in pred_results:
            status, recall = "no prediction", 0.0
        else:
            recall = lcs_recall(gt_seq, pred_seq)
            status = "OK"
            print(f"  {vid:<14} GT:{len(gt_seq):>4}  pred(filtered):{len(pred_seq):>4}  "
                  f"recall: {recall:>6.2f} %")

        rows.append(dict(
            video_id=vid,
            gt_actions=len(gt_seq),
            pred_actions=len(pred_seq),
            recall_percent=round(recall, 2) if recall is not None else None,
            status=status,
        ))

    valid = [r["recall_percent"] for r in rows if r["recall_percent"] is not None]
    avg = sum(valid) / len(valid) if valid else 0.0

    print()
    print("=" * 60)
    print(f"TAD LCS-recall evaluation, subset = '{args.subset}'")
    print(f"  videos in subset      : {len(rows)}")
    print(f"  with valid GT + pred  : {len(valid)}")
    print(f"  empty/missing         : {len(rows) - len(valid)}")
    print(f"  filter: score >= {args.min_score}"
          + (f", top-{args.topk}" if args.topk else ""))
    print(f"  GLOBAL AVERAGE LCS RECALL: {avg:.2f} %")
    print("=" * 60)

    if args.out_csv:
        fieldnames = ["video_id", "gt_actions", "pred_actions",
                      "recall_percent", "status"]
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Per-video results saved to {args.out_csv}")


if __name__ == "__main__":
    main()
