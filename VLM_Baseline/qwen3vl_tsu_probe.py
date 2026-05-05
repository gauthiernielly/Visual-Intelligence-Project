"""
qwen3vl_tsu_probe.py
Quick diagnostic: run Qwen3-VL-8B on a single TSU video.
Measures timing and tests basic temporal reasoning.

Usage:
    python qwen3vl_tsu_probe.py --video /path/to/video.mp4 --nframes 64
"""

import argparse
import time
import json
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Temporal reasoning questions to ask after the timeline prompt.
# Adapt these to match what you actually see in your test video.
# TEMPORAL_QUESTIONS = [
#     "List every distinct activity you observed, in the order they occurred. "
#     "For each one, give an approximate start and end time as MM:SS.",

#     "Did the person perform any activity more than once? If so, which one, "
#     "and at approximately what times?",

#     "Roughly how long did the longest single activity last?",

#     "What was the person doing between the 5-minute and 10-minute marks?",
# ]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_model(model_id: str):
    t0 = time.time()
    print(f"Loading model: {model_id} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,   # V100 doesn't support bfloat16
        attn_implementation="flash_attention_2",
        device_map="auto",           # spreads across available GPUs automatically
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Device map: {model.hf_device_map}")
    return model, processor


def run_inference(model, processor, messages, label=""):
    t0 = time.time()

    # 1. Build the text prompt from the chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 2. Process vision inputs (this is where frame extraction happens)
    t_vision = time.time()
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    print(f"  [{label}] Vision preprocessing: {time.time() - t_vision:.1f}s")

    # 3. Tokenize everything together
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    n_tokens = inputs["input_ids"].shape[-1]
    print(f"  [{label}] Total input tokens: {n_tokens:,}")

    # 4. Generate
    t_gen = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # greedy for reproducibility
        )
    gen_time = time.time() - t_gen

    # 5. Decode only the newly generated tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    total_time = time.time() - t0
    new_tokens = generated.shape[-1]
    print(f"  [{label}] Generation: {gen_time:.1f}s  "
          f"({new_tokens} tokens, {new_tokens/gen_time:.1f} tok/s)")
    print(f"  [{label}] Total wall time: {total_time:.1f}s")

    return response, {
        "label": label,
        "input_tokens": n_tokens,
        "output_tokens": new_tokens,
        "gen_time_s": round(gen_time, 2),
        "total_time_s": round(total_time, 2),
        "tok_per_s": round(new_tokens / gen_time, 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to a TSU video file")
    parser.add_argument("--nframes", type=int, default=32,
                        help="Number of frames to sample (default 64; try 32, 64, 128)")
    parser.add_argument("--out", default="results.json",
                        help="Where to save timing + answers")
    args = parser.parse_args()

    model, processor = load_model(MODEL_ID)

    results = {"video": args.video, "nframes": args.nframes, "runs": []}

    # ── Pass 1: Ask for a full activity timeline (the core task) ─────────────
    print("\n─── Pass 1: Timeline reconstruction ───")
    timeline_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.video,
                    "nframes": args.nframes,   # hard cap on frame count
                    # "fps": 0.5,             # alternative: use fps instead
                },
                {
                    "type": "text",
                    "text": (
                        "You are analyzing a home activity video. "
                        "Watch the entire video and produce a structured timeline of all activities.\n\n"
                        "Return ONLY a JSON array. Each element must have:\n"
                        '  "start": approximate start time as "MM:SS",\n'
                        '  "end": approximate end time as "MM:SS",\n'
                        '  "activity": a short description of what the person is doing.\n\n'
                        "Example format:\n"
                        '[{"start": "00:00", "end": "01:30", "activity": "walking to kitchen"},\n'
                        ' {"start": "01:30", "end": "04:00", "activity": "preparing coffee"}]\n\n'
                        "Return the JSON array only, no other text."
                    ),
                },
            ],
        }
    ]

    timeline_response, timeline_timing = run_inference(
        model, processor, timeline_messages, label="timeline"
    )
    print(f"\nTimeline response:\n{timeline_response}\n")

    # Try to parse the JSON; record parse success
    try:
        timeline_json = json.loads(timeline_response)
        timeline_timing["parse_ok"] = True
        timeline_timing["n_segments"] = len(timeline_json)
    except json.JSONDecodeError:
        timeline_timing["parse_ok"] = False
        timeline_timing["raw_response"] = timeline_response

    results["runs"].append(timeline_timing)

    # ── Save everything ───────────────────────────────────────────────────────
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()