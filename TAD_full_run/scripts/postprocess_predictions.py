"""
Read OpenTAD's raw `result_detection.json`, add the integer `label_id` field,
drop degenerate or low-confidence segments, dedupe segments produced by the
DDP test sampler, and write the canonical predictions JSON consumed by the
hybrid stage.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="OpenTAD's result_detection.json")
    p.add_argument("--classes", required=True, help="category_idx.txt, one class per line")
    p.add_argument("--output", required=True)
    p.add_argument("--min-duration", type=float, default=0.1,
                   help="drop segments shorter than this many seconds")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="drop segments with score below this threshold")
    args = p.parse_args()

    with open(args.raw) as f:
        raw = json.load(f)
    with open(args.classes) as f:
        class_names = [l.strip() for l in f if l.strip()]
    name_to_id = {n: i for i, n in enumerate(class_names)}

    results = raw.get("results", raw)
    canon = {}
    n_kept = n_drop_dur = n_drop_score = n_drop_label = n_drop_dup = 0
    for vid, segs in results.items():
        # Dedup by (rounded start, rounded end, label_id), keep highest score.
        # The DDP test sampler pads test_set to a multiple of world_size by
        # repeating videos, so duplicate predictions can show up here.
        seen = {}
        for s in segs:
            label = s["label"]
            if label not in name_to_id:
                n_drop_label += 1
                continue
            start, end = s["segment"]
            if end - start < args.min_duration:
                n_drop_dur += 1
                continue
            score = float(s["score"])
            if score < args.min_score:
                n_drop_score += 1
                continue
            label_id = int(name_to_id[label])
            key = (round(float(start), 4), round(float(end), 4), label_id)
            if key in seen:
                n_drop_dup += 1
                if score > seen[key]["score"]:
                    seen[key]["score"] = score
                continue
            seen[key] = {
                "segment":  [float(start), float(end)],
                "label":    label,
                "label_id": label_id,
                "score":    score,
            }
            n_kept += 1
        canon[vid] = sorted(seen.values(), key=lambda x: x["segment"][0])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": canon}, f, indent=1)

    print(f"Wrote {args.output}")
    print(f"  videos        : {len(canon)}")
    print(f"  segments kept : {n_kept}")
    print(f"  dropped (zero/short duration < {args.min_duration}s): {n_drop_dur}")
    print(f"  dropped (low score < {args.min_score}): {n_drop_score}")
    print(f"  dropped (exact duplicates from DDP test): {n_drop_dup}")
    if n_drop_label:
        print(f"  dropped (unknown label): {n_drop_label}")


if __name__ == "__main__":
    main()
