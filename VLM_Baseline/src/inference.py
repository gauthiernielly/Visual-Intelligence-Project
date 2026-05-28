import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from qwen_vl_utils import process_vision_info

import config as cfg


def build_input(frames: list, frame_indices: np.ndarray, dataset_fps: int = 25) -> list[dict]:
    frame_ts_lines = []
    for i, f in enumerate(frame_indices):
        ts = f / dataset_fps
        mm, ss = int(ts // 60), ts % 60
        frame_ts_lines.append(f"  frame {i+1:02d} → index {f:6d} → {mm:02d}:{ss:05.2f}")
    frame_map = "\n".join(frame_ts_lines)

    first_frame = int(frame_indices[0])
    last_frame = int(frame_indices[-1])
    classes_str = "\n".join(f"  - {c}" for c in cfg.TSU_CLASSES)

    prompt = f"""
      You are analyzing a segment of a home activity video recorded at {dataset_fps} fps.

      You have been given {len(frame_indices)} frames sampled from this segment.
      The frames you see, in order, correspond to the following video positions:
      {frame_map}

      Your task: identify all activity segments in chronological order.

      IMPORTANT — OVERLAPPING ACTIVITIES:
      Multiple activities can occur simultaneously. For example, a person can be
      walking while talking on the phone, or drinking while watching TV.
      Each activity must be listed as a separate segment with its own time range,
      even if it overlaps with another segment. Do not merge co-occurring activities
      into one label.

      RULES:
      1. Use ONLY labels from this list:
      {classes_str}
      2. Express start_frame and end_frame as GLOBAL frame indices (as shown in the map above), \
      not as frame numbers within this segment.
      3. Output ONLY a JSON object with a "segments" array. Each element must have:
        - "event":       one label from the list above
        - "start_frame": global frame index where the activity starts
        - "end_frame":   global frame index where the activity ends
      4. Only include activities you can actually observe. Do not guess beyond the visible frames.

      Example:
      {{
        "segments": [
          {{"event": "Walk",         "start_frame": {first_frame},       "end_frame": {first_frame + 270}}},
          {{"event": "Use_telephone", "start_frame": {first_frame + 180},  "end_frame": {last_frame}}},
          {{"event": "Sit_down",     "start_frame": {first_frame + 270}, "end_frame": {first_frame + 270}}}
        ]
      }}

      Return the JSON object only.
    """
    return [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text",  "text": prompt},
        ],
    }]


def run_and_parse(
    processor,
    messages: list,
    model,
    frame_indices: np.ndarray,
    dataset_fps: int,
    total_video_frames: int,
    max_new_tokens: int = 512,
) -> list[dict]:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True
    )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
        video_metadatas[0]["fps"] = dataset_fps
        video_metadatas[0]["frames_indices"] = [int(i) for i in frame_indices]
        video_metadatas[0]["total_num_frames"] = float(total_video_frames)
    else:
        video_metadatas = None

    if video_kwargs:
        video_kwargs = {
            k: v[0] if isinstance(v, list) and len(v) == 1 else v
            for k, v in video_kwargs.items()
        }

    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    print(f"    Input tokens: {inputs['input_ids'].shape[-1]:,}")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"    Generated tokens: {generated.shape[-1]}")

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        print("    [parse] no JSON found in response")
        return []
    try:
        data = json.loads(match.group())
        segments = data.get("segments", [])
        valid = []
        for s in segments:
            s["start_frame"] = max(0, min(int(s["start_frame"]), total_video_frames - 1))
            s["end_frame"] = max(0, min(int(s["end_frame"]), total_video_frames - 1))
            if s["end_frame"] > s["start_frame"] and s.get("event") in cfg.TSU_CLASSES:
                valid.append(s)
        return valid
    except (json.JSONDecodeError, KeyError) as e:
        print(f"    [parse] failed: {e}")
        return []


def merge_segments(all_segments: list[dict], overlap_frames: int) -> list[dict]:
    if not all_segments:
        return []

    by_class = defaultdict(list)
    for s in sorted(all_segments, key=lambda s: s["start_frame"]):
        by_class[s["event"]].append(s)

    merged_by_class = []
    for _, class_segs in by_class.items():
        class_segs = sorted(class_segs, key=lambda s: s["start_frame"])
        merged = [dict(class_segs[0])]
        for seg in class_segs[1:]:
            last = merged[-1]
            overlap_len = (
                min(last["end_frame"], seg["end_frame"])
                - max(last["start_frame"], seg["start_frame"])
            )
            gap = seg["start_frame"] - last["end_frame"]
            if overlap_len > 0 or gap <= overlap_frames:
                merged[-1]["end_frame"] = max(last["end_frame"], seg["end_frame"])
            else:
                merged.append(dict(seg))
        merged_by_class.extend(merged)

    return sorted(merged_by_class, key=lambda s: s["start_frame"])


def sliding_window_inference(
    video_path: str,
    model,
    processor,
    window_sec: float = cfg.WINDOW_SEC,
    overlap_sec: float = cfg.OVERLAP_SEC,
    window_fps: float = cfg.WINDOW_FPS,
    dataset_fps: int = cfg.DATASET_FPS,
    max_new_tokens: int = cfg.MAX_NEW_TOKENS,
) -> list[dict]:
    vr = VideoReader(video_path)
    total_frames = len(vr)
    duration_sec = total_frames / dataset_fps

    overlap_frames = int(overlap_sec * dataset_fps)
    stride_sec = window_sec - overlap_sec
    window_starts = np.arange(0, duration_sec - overlap_sec, stride_sec)

    print(f"  {total_frames} frames, {duration_sec / 60:.1f} min → {len(window_starts)} windows")

    all_segments = []
    for i, start_sec in enumerate(window_starts):
        end_sec = min(start_sec + window_sec, duration_sec)

        start_frame = int(start_sec * dataset_fps)
        end_frame = min(int(end_sec * dataset_fps), total_frames - 1)

        window_duration = end_sec - start_sec
        nframes = max(2, int(window_duration * window_fps))
        indices = np.linspace(start_frame, end_frame, nframes, dtype=int)

        print(f"  Window {i + 1}/{len(window_starts)}: "
              f"frames {start_frame}–{end_frame} ({window_duration:.0f}s, {nframes} frames)")

        frames = vr.get_batch(indices).asnumpy()
        pil_frames = [Image.fromarray(f) for f in frames]

        messages = build_input(pil_frames, indices, dataset_fps)
        segments = run_and_parse(
            processor, messages, model, indices, dataset_fps, total_frames, max_new_tokens
        )

        print(f"    → {len(segments)} segments parsed")
        all_segments.extend(segments)

    return merge_segments(all_segments, overlap_frames)
