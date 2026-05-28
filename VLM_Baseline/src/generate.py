"""
Run sliding-window VLM temporal segmentation on all videos in VIDEO_DIR.

Outputs a single results.json in OUTPUT_DIR keyed by video_id. The file is
written after every video so interrupted runs can be safely resumed with
--resume (the default).

Usage:
    python src/generate.py
    python src/generate.py --video_dir /path/to/videos --output_dir results/
    python src/generate.py --gen_indices /path/to/gen_indices.json
    python src/generate.py --limit 5          # process only the first 5 videos
    python src/generate.py --no-resume        # reprocess even if entry exists
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import config as cfg
from inference import sliding_window_inference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM temporal segmentation over a video dataset")

    p.add_argument("--video_dir",      default=cfg.VIDEO_DIR)
    p.add_argument("--output_dir",     default=cfg.OUTPUT_DIR)
    p.add_argument("--model_id",       default=cfg.MODEL_ID)
    p.add_argument("--dataset_fps",    type=int,   default=cfg.DATASET_FPS)
    p.add_argument("--window_sec",     type=float, default=cfg.WINDOW_SEC)
    p.add_argument("--overlap_sec",    type=float, default=cfg.OVERLAP_SEC)
    p.add_argument("--window_fps",     type=float, default=cfg.WINDOW_FPS)
    p.add_argument("--max_new_tokens", type=int,   default=cfg.MAX_NEW_TOKENS)
    p.add_argument("--gen_indices",    default="gen_indices.json",
                   help="Path to a JSON file listing video IDs to process (in order). "
                        "If omitted, all *.mp4 files in video_dir are used.")
    p.add_argument("--limit",          type=int,   default=None,
                   help="Cap the number of videos to process (for debugging)")
    p.add_argument("--resume",         action=argparse.BooleanOptionalAction, default=True,
                   help="Skip videos whose output JSON already exists (default: on)")

    args, _ = p.parse_known_args()
    return args


def load_model(model_id: str):
    print(f"Loading model {model_id} ...")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Model loaded in {time.time() - t0:.1f}s  |  device map: {model.hf_device_map}")
    return model, processor


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "generated_segments.json"

    # Load existing results for resume support
    results: dict = {}
    if output_file.exists():
        with open(output_file) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {output_file}")

    video_dir = Path(args.video_dir)
    if args.gen_indices:
        with open(args.gen_indices) as f:
            video_ids = json.load(f)
        video_paths = []
        for vid_id in video_ids:
            p = video_dir / f"{vid_id}.mp4"
            if p.exists():
                video_paths.append(p)
            else:
                print(f"[warn] {vid_id}.mp4 not found in {args.video_dir}, skipping")
        print(f"Loaded {len(video_paths)} videos from {args.gen_indices}")
    else:
        video_paths = sorted(video_dir.glob("*.mp4"))
        print(f"Found {len(video_paths)} videos in {args.video_dir}")
    if args.limit:
        video_paths = video_paths[: args.limit]

    model, processor = load_model(args.model_id)

    n_failed = 0

    for video_path in tqdm(video_paths, desc="Videos", unit="video"):
        video_id = video_path.stem

        if args.resume and video_id in results:
            tqdm.write(f"[skip] {video_id} (already in results)")
            continue

        tqdm.write(f"\n[start] {video_id}")
        t_start = time.time()

        try:
            segments = sliding_window_inference(
                str(video_path),
                model,
                processor,
                window_sec=args.window_sec,
                overlap_sec=args.overlap_sec,
                window_fps=args.window_fps,
                dataset_fps=args.dataset_fps,
                max_new_tokens=args.max_new_tokens,
            )
            elapsed = time.time() - t_start

            results[video_id] = {
                "video_id": video_id,
                "video_path": str(video_path),
                "segments": segments,
                "elapsed_sec": round(elapsed, 1),
                "config": {
                    "model_id":       args.model_id,
                    "window_sec":     args.window_sec,
                    "overlap_sec":    args.overlap_sec,
                    "window_fps":     args.window_fps,
                    "dataset_fps":    args.dataset_fps,
                    "max_new_tokens": args.max_new_tokens,
                },
            }
            tqdm.write(f"[done]  {video_id} → {len(segments)} segments  ({elapsed:.0f}s)")

        except Exception as e:
            elapsed = time.time() - t_start
            tqdm.write(f"[fail]  {video_id}: {e}  ({elapsed:.0f}s)")
            results[video_id] = {
                "video_id": video_id,
                "error": str(e),
                "elapsed_sec": round(elapsed, 1),
            }
            n_failed += 1

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    n_ok = len(video_paths) - n_failed
    print(f"\nFinished: {n_ok}/{len(video_paths)} succeeded, {n_failed} failed.")
    print(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
