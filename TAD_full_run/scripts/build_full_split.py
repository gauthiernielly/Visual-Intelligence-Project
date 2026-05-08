"""
build_full_split.py
===================
Build the full-TSU annotation file for the cluster run from data_cs_split.json.

Operations:
- Hold out N entire subjects from the training set as the validation set
  (default: P25). The remaining 10 train subjects stay as training.
- Test set is unchanged (the original CS test split).
- Add a per-video `frame` field (= round(duration * fps)) required by
  OpenTAD's ThumosPaddingDataset.
- Write a `category_idx.txt` (one class per line, alphabetical) for OpenTAD.

Outputs are deterministic given the same inputs.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="path to data_cs_split.json")
    p.add_argument("--output", required=True,
                   help="output annotation JSON for OpenTAD")
    p.add_argument("--cat-out", required=True,
                   help="output category_idx.txt")
    p.add_argument("--val-subjects", default="P25",
                   help="comma-separated subject IDs to move from training to validation "
                        "(must be a subset of the original training subjects)")
    p.add_argument("--fps", type=float, default=25.0)
    args = p.parse_args()

    with open(args.input) as f:
        master = json.load(f)

    db = master["database"]
    classes = master["classes"]
    val_subjects = {s.strip() for s in args.val_subjects.split(",") if s.strip()}

    train_subjects_orig = sorted({v[:3] for v, e in db.items()
                                  if e.get("subset") == "training"})
    test_subjects = sorted({v[:3] for v, e in db.items()
                            if e.get("subset") == "testing"})
    bad = val_subjects - set(train_subjects_orig)
    if bad:
        raise SystemExit(
            f"--val-subjects {sorted(bad)} are not in the original training subjects "
            f"({train_subjects_orig}); cannot move them to validation."
        )

    out_db = {}
    n_per_subset = Counter()
    for vid, entry in db.items():
        subj = vid[:3]
        new_entry = dict(entry)
        if entry.get("subset") == "training" and subj in val_subjects:
            new_entry["subset"] = "validation"
        new_entry["frame"] = int(round(float(entry["duration"]) * args.fps))
        out_db[vid] = new_entry
        n_per_subset[new_entry["subset"]] += 1

    out = {
        "version": master.get("version", "TSU-CS-51"),
        "classes": classes,
        "database": out_db,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)

    Path(args.cat_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.cat_out, "w") as f:
        for c in classes:
            f.write(c + "\n")

    print(f"Wrote {args.output}")
    print(f"  videos: {len(out_db)}  ({dict(n_per_subset)})")
    print(f"  train subjects: "
          f"{sorted(set(train_subjects_orig) - val_subjects)} "
          f"(moved {sorted(val_subjects)} to validation)")
    print(f"  test subjects:  {test_subjects}")
    print(f"  classes file:   {args.cat_out}  ({len(classes)} classes)")


if __name__ == "__main__":
    main()
