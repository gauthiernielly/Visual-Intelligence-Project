"""
Sanity-check the canonical predictions JSON before it is handed downstream.

Checks the schema (required keys, sorted by start time, valid label_id, score
in [0, 1]), checks segment boundaries against the ground-truth duration, and
prints a small summary.
"""

import argparse
import json
from collections import Counter


def get_results(pred_obj):
    if isinstance(pred_obj, dict) and "results" in pred_obj:
        return pred_obj["results"]
    return pred_obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground-truth", required=True,
                   help="OpenTAD-format GT JSON, used to read video durations")
    p.add_argument("--num-classes", type=int, default=51)
    args = p.parse_args()

    print(f"Loading predictions: {args.predictions}")
    with open(args.predictions) as f:
        pred_obj = json.load(f)
    results = get_results(pred_obj)
    print(f"  videos with predictions: {len(results)}")

    print(f"Loading ground truth: {args.ground_truth}")
    with open(args.ground_truth) as f:
        gt = json.load(f)
    gt_db = gt.get("database", gt)

    n_problems = 0
    total_segments = 0
    score_min, score_max = 1.0, 0.0
    durations = []
    classes_seen = Counter()

    for vid, segments in results.items():
        if vid not in gt_db:
            print(f"  [warn] {vid} not in ground truth")

        gt_dur = gt_db.get(vid, {}).get("duration", None)

        if not isinstance(segments, list):
            print(f"  [error] {vid}: predictions is not a list")
            n_problems += 1
            continue

        prev_start = -1.0
        for i, seg in enumerate(segments):
            for key in ("label", "label_id", "score", "segment"):
                if key not in seg:
                    print(f"  [error] {vid}#{i}: missing key '{key}'")
                    n_problems += 1
                    continue

            cls_id = seg["label_id"]
            score = float(seg["score"])
            start, end = seg["segment"]

            if not (0 <= cls_id < args.num_classes):
                print(f"  [error] {vid}#{i}: label_id={cls_id} out of [0,{args.num_classes})")
                n_problems += 1
            if not (0.0 <= score <= 1.0):
                print(f"  [error] {vid}#{i}: score={score} out of [0,1]")
                n_problems += 1
            if not (start < end):
                print(f"  [error] {vid}#{i}: invalid segment [{start}, {end}]")
                n_problems += 1
            if gt_dur is not None and (start < 0 or end > gt_dur + 0.5):
                print(f"  [warn] {vid}#{i}: segment [{start}, {end}] "
                      f"outside duration {gt_dur}")
            if start < prev_start - 1e-3:
                print(f"  [warn] {vid}#{i}: not sorted by start time")
            prev_start = start

            score_min = min(score_min, score)
            score_max = max(score_max, score)
            durations.append(end - start)
            classes_seen[cls_id] += 1
            total_segments += 1

    print("\n--- Summary ---")
    print(f"  total segments         : {total_segments}")
    print(f"  problems               : {n_problems}")
    print(f"  score range            : [{score_min:.3f}, {score_max:.3f}]")
    if durations:
        durations.sort()
        print(f"  duration min/median/max: "
              f"{durations[0]:.1f} / "
              f"{durations[len(durations)//2]:.1f} / "
              f"{durations[-1]:.1f} sec")
    print(f"  unique classes         : {len(classes_seen)}")
    print(f"  most common classes    : {classes_seen.most_common(5)}")

    if n_problems == 0:
        print("\nOK, predictions JSON looks good.")
    else:
        print(f"\n{n_problems} problems found. Fix and re-export.")


if __name__ == "__main__":
    main()
