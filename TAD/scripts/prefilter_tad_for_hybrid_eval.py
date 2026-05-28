"""
Apply the same prefilter the Hybrid pipeline runs before invoking the VLM, so
the TAD baseline can be evaluated by the shared `complete_eval.py` script at
the same prediction density as Hybrid. Restricts to a video-id list and writes
a JSON in the format `complete_eval.py` consumes.

The three filtering steps are identical to those in `hybrid.py`:
  - deduplicate_segments
  - score >= min_score (default 0.15)
  - per-class temporal NMS at IoU > nms_iou (default 0.3)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def deduplicate_segments(segments):
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


def nms_per_class(segments, iou_threshold=0.3):
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
                   help="text file with one video_id per line")
    p.add_argument("--output", default="tad_pipeline_results_86.json",
                   help="output JSON path, in the format complete_eval.py consumes")
    p.add_argument("--min-score", type=float, default=0.15,
                   help="confidence threshold, matches hybrid.py default")
    p.add_argument("--nms-iou", type=float, default=0.3,
                   help="per-class NMS IoU threshold, matches hybrid.py default")
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
    print(f"  dropped by dedup           : {n_dropped_dup}")
    print(f"  dropped by score < {args.min_score}   : {n_dropped_score}")
    print(f"  dropped by NMS IoU={args.nms_iou}     : {n_dropped_nms}")
    if total_out and len(out_results):
        print(f"  average segments / video   : {total_out / len(out_results):.0f}")


if __name__ == "__main__":
    main()
