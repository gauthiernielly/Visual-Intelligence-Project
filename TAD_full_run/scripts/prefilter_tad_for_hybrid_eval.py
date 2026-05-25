"""
prefilter_tad_for_hybrid_eval.py
================================
Prepare the TAD predictions JSON so it can be evaluated by the shared
`hybrid_complete_eval.py` script (the same one used for Hybrid and VLM-only),
producing a directly comparable TAD-only number.

What this script does, step by step:
  1. Loads the canonical TAD predictions JSON (predictions_canonical.json).
  2. Restricts to the agreed evaluation video subset (default: hybrid_eval_68.txt).
  3. Applies the EXACT prefilter the hybrid pipeline runs before calling the
     VLM, in `hybrid.py`:
        - deduplicate_segments  (drop (start, end, label) duplicates)
        - score >= MIN_CONFIDENCE  (default 0.15)
        - nms_per_class with IoU = 0.3
     This produces the same set of segments the hybrid pipeline would send to
     the VLM, except no VLM relabel happens. Useful because it isolates the
     TAD localisation+labelling quality at the same density as Hybrid.
  4. Saves the result in the format hybrid_complete_eval.py reads
     (`{"results": {video_id: [{segment, label, score, ...}]}}`).

Output filename is "tad_pipeline_results.json" by default. Once written,
point hybrid_complete_eval.py at it:
    JSON_PATH = "tad_pipeline_results.json"
and run it on the cluster (or anywhere the GT CSV folder is reachable).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def deduplicate_segments(segments: list) -> list:
    """Identical to hybrid.py: drop (start, end, label) duplicates."""
    seen = set()
    unique = []
    for seg in segments:
        key = (round(seg["segment"][0], 3),
               round(seg["segment"][1], 3),
               seg["label"])
        if key not in seen:
            seen.add(key)
            unique.append(seg)
    return unique


def nms_per_class(segments: list, iou_threshold: float = 0.3) -> list:
    """Identical per-class temporal NMS used by hybrid.py."""
    by_class = defaultdict(list)
    for seg in segments:
        by_class[seg["label"]].append(seg)
    kept = []
    for cls, cls_segs in by_class.items():
        cls_segs = sorted(cls_segs, key=lambda x: -x["score"])
        survivors = []
        for candidate in cls_segs:
            cs, ce = candidate["segment"]
            suppress = False
            for s in survivors:
                ss, se = s["segment"]
                inter = max(0.0, min(ce, se) - max(cs, ss))
                union = (ce - cs) + (se - ss) - inter
                if union > 0 and inter / union > iou_threshold:
                    suppress = True
                    break
            if not suppress:
                survivors.append(candidate)
        kept.extend(survivors)
    return sorted(kept, key=lambda x: x["segment"][0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="path to predictions_canonical.json (raw TAD output)")
    p.add_argument("--video-list", required=True,
                   help="text file with one video_id per line (the 68 IDs)")
    p.add_argument("--output", default="tad_pipeline_results.json",
                   help="path for the prefiltered JSON in hybrid-eval format")
    p.add_argument("--min-score", type=float, default=0.15,
                   help="score threshold (matches hybrid.py MIN_CONFIDENCE=0.15)")
    p.add_argument("--nms-iou", type=float, default=0.3,
                   help="per-class NMS IoU threshold (matches hybrid.py default 0.3)")
    args = p.parse_args()

    print(f"Loading TAD predictions from {args.predictions}")
    with open(args.predictions) as f:
        raw = json.load(f)
    pred_results = raw.get("results", raw)
    print(f"  TAD JSON contains {len(pred_results)} videos total")

    video_ids = [l.strip() for l in open(args.video_list) if l.strip()]
    print(f"Restricting to {len(video_ids)} videos from {args.video_list}")

    out_results = {}
    total_in = total_out = 0
    n_dropped_dup = n_dropped_score = n_dropped_nms = 0
    missing = []

    for vid in video_ids:
        if vid not in pred_results:
            missing.append(vid)
            out_results[vid] = []
            continue

        segs = pred_results[vid]
        total_in += len(segs)

        before = len(segs)
        segs = deduplicate_segments(segs)
        n_dropped_dup += before - len(segs)

        before = len(segs)
        segs = [s for s in segs if float(s["score"]) >= args.min_score]
        n_dropped_score += before - len(segs)

        before = len(segs)
        segs = nms_per_class(segs, iou_threshold=args.nms_iou)
        n_dropped_nms += before - len(segs)

        out_results[vid] = segs
        total_out += len(segs)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": out_results}, f, indent=1)

    print()
    print(f"Wrote {args.output}")
    print(f"  videos kept                : {len(out_results)}")
    print(f"  videos missing from TAD JSON: {len(missing)}"
          + (f" -> {missing[:5]}..." if missing else ""))
    print(f"  segments in (before filter): {total_in}")
    print(f"  segments out (after filter): {total_out}")
    print(f"  dropped by dedup            : {n_dropped_dup}")
    print(f"  dropped by score < {args.min_score}    : {n_dropped_score}")
    print(f"  dropped by NMS IoU={args.nms_iou}      : {n_dropped_nms}")
    if total_out and len(out_results):
        print(f"  average segments / video    : {total_out / len(out_results):.0f}")
    print()
    print("Next step: run hybrid_complete_eval.py with JSON_PATH pointing at this file.")
    print(f"For example, edit hybrid_complete_eval.py:")
    print(f'    JSON_PATH = "{args.output}"')
    print("and run it from a location where the GT CSV folder is reachable.")


if __name__ == "__main__":
    main()
