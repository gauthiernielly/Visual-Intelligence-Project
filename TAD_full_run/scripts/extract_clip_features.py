"""
extract_clip_features.py
========================
CLIP ViT-B/32 feature extraction for every video referenced by the annotation
JSON. Resumable: skips videos whose .npy already exists.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import open_clip

try:
    from decord import VideoReader, cpu
    DECORD_OK = True
except ImportError:
    DECORD_OK = False
    import cv2


def list_videos_from_ann(video_dir, ann_file):
    """Only extract features for videos present in the annotation file."""
    with open(ann_file) as f:
        db = json.load(f)["database"]
    out = []
    missing = []
    for vid in db.keys():
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            p = Path(video_dir) / f"{vid}{ext}"
            if p.exists():
                out.append(p); break
        else:
            missing.append(vid)
    if missing:
        print(f"  WARNING: {len(missing)} videos in JSON not found in {video_dir}: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    return sorted(out)


def video_dur_fps_count(path):
    if DECORD_OK:
        vr = VideoReader(str(path), ctx=cpu(0))
        n = len(vr); fps = float(vr.get_avg_fps())
        return n / max(fps, 1e-6), fps, n
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n / max(fps, 1e-6), fps, n


def sample_indices(n_native, native_fps, target_fps, feat_stride):
    """One snippet every feat_stride/target_fps seconds."""
    period = feat_stride / target_fps
    duration = n_native / native_fps
    n_snip = int(duration / period)
    if n_snip <= 0:
        return []
    out = []
    for k in range(n_snip):
        t = k * period
        idx = int(round(t * native_fps))
        out.append(min(max(idx, 0), n_native - 1))
    return out


def read_frames(path, indices):
    if DECORD_OK:
        vr = VideoReader(str(path), ctx=cpu(0))
        batch = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(arr) for arr in batch]
    cap = cv2.VideoCapture(str(path))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            frames.append(frames[-1] if frames else Image.new("RGB", (640, 480)))
            continue
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ann-file", required=True,
                   help="annotation JSON; only videos listed here are extracted")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--feat-stride", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ({args.pretrained}) on {args.device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(args.device).eval()
    fdim = model.visual.output_dim
    print(f"  feature dim: {fdim}")

    videos = list_videos_from_ann(args.video_dir, args.ann_file)
    if args.limit:
        videos = videos[: args.limit]
    print(f"Found {len(videos)} videos to extract")

    n_done = n_skipped = n_failed = 0
    metadata = []

    for vpath in tqdm(videos, desc="videos"):
        vid = vpath.stem
        out_path = out_dir / f"{vid}.npy"
        if out_path.exists() and not args.overwrite:
            n_skipped += 1; continue

        try:
            duration, native_fps, n_frames = video_dur_fps_count(vpath)
        except Exception as e:
            print(f"  [error] {vid}: cannot open ({e})"); n_failed += 1; continue

        idx = sample_indices(n_frames, native_fps, args.fps, args.feat_stride)
        if not idx:
            print(f"  [warn] {vid}: too short, skipping"); n_failed += 1; continue

        try:
            frames = read_frames(vpath, idx)
        except Exception as e:
            print(f"  [error] {vid}: read failed ({e})"); n_failed += 1; continue

        feats = []
        with torch.no_grad():
            for i in range(0, len(frames), args.batch_size):
                batch = frames[i:i + args.batch_size]
                t = torch.stack([preprocess(im) for im in batch]).to(args.device)
                with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                    f = model.encode_image(t)
                feats.append(f.float().cpu().numpy())
        feats = np.concatenate(feats, axis=0).astype(np.float32)
        assert feats.shape == (len(idx), fdim), f"bad shape {feats.shape} for {vid}"
        np.save(out_path, feats)
        metadata.append(dict(video_id=vid, num_snippets=len(idx),
                             duration=duration, native_fps=native_fps))
        n_done += 1

    with open(out_dir / "EXTRACTION_INFO.json", "w") as f:
        json.dump(dict(
            feature_backbone=f"{args.model}/{args.pretrained}",
            feature_dim=int(fdim),
            target_fps=args.fps,
            feat_stride=args.feat_stride,
            videos_done=len(metadata),
            videos=metadata,
        ), f, indent=2)

    print(f"\n  extracted: {n_done}")
    print(f"  skipped (cached): {n_skipped}")
    print(f"  failed: {n_failed}")


if __name__ == "__main__":
    main()
