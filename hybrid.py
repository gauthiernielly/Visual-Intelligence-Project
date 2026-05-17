import json
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration # Adjust import if using a specific Qwen3 class
from qwen_vl_utils import process_vision_info
import time

# --- Configuration ---
VIDEO_ROOT = "../../../../work/cs-503/sadgal/Videos_mp4"
TAD_RESULTS_PATH = "TAD_full_run/outputs/predictions_canonical.json"
OUTPUT_PATH = "hybrid_pipeline_results.json"
MIN_SEGMENT_DURATION = 0.5  # seconds
NFRAMES_PER_CLIP = 8

TSU_CLASSES = [
    "Walk", "Take_something_off_table", "Put_something_on_table", "Drink.From_cup", "Get_up",
    "Sit_down", "Read", "Watch_TV", "Enter", "Use_Drawer", "Leave", "Breakfast.Eat_at_table",
    "Cook.Stir", "Use_cupboard", "Write", "Use_laptop", "Use_telephone", "Clean_dishes.Dry_up",
    "Take_pills", "Drink.From_bottle", "Eat_snack", "Clean_dishes", "Drink.From_can", "Use_glasses",
    "Pour.From_bottle", "Cook.Use_oven", "Dump_in_trash", "Breakfast.Cut_bread", "Use_tablet",
    "Use_fridge", "Cook.Cut", "Wipe_table", "Lay_down", "Cook.Use_stove", "Cook",
    "Clean_dishes.Clean_with_water", "Pour.From_kettle", "Breakfast.Spread_jam_or_butter", "Insert_tea_bag",
    "Get_water", "Clean_dishes.Put_something_in_sink", "Make_coffee.Pour_water", "Make_coffee",
    "Drink.From_glass", "Pour.From_can", "Make_coffee.Pour_grains", "Breakfast", "Make_tea",
    "Make_tea.Boil_water", "Stir_coffee_tea", "Breakfast.Take_ham",
]

CLASS_LIST_FOR_PROMPT = "\n".join(f"  - {c}" for c in TSU_CLASSES)

_CANONICAL_LOWER = {c.lower().replace(".", "_").replace(" ", "_"): c for c in TSU_CLASSES}

def build_messages(video_path, start, end, tad_label, tad_score):
    """
    Constructs the structured prompt for Qwen3-VL, natively trimming via the video dict.
    """
    prompt_text = (
        f"This clip runs from {start:.1f}s to {end:.1f}s. "
        f"An action detector predicted '{tad_label}' with confidence {tad_score:.2f}. "
        "Use this as a hint but override it if the frames clearly show something different.\n\n"
        "You MUST pick exactly one label from this list:\n"
        f"{CLASS_LIST_FOR_PROMPT}\n\n"
        "Return ONLY a valid JSON object, no markdown:\n"
        '{"label": "<exact class from list above>", "confidence": <your confidence as a float>}'
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "video_start": start,
                    "video_end": end,
                    "nframes": NFRAMES_PER_CLIP,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    return messages
'''
import cv2

def extract_frames(video_path, start, end, n_frames=NFRAMES_PER_CLIP):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = min(n_frames, max(1, int((end - start) * fps)))
    
    timestamps = (
        [start + (end - start) / 2] if n == 1
        else [start + i * (end - start) / (n - 1) for i in range(n)]
    )
    
    frames = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB PIL Image
            from PIL import Image
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    
    cap.release()
    return frames


def build_messages(frames, start, end, tad_label, tad_score):
    prompt_text = (
        f"This clip runs from {start:.1f}s to {end:.1f}s. "
        f"An action detector predicted '{tad_label}' with confidence {tad_score:.2f}. "
        "Use this as a hint but override it if the frames clearly show something different.\n\n"
        "You MUST pick exactly one label from this list:\n"
        f"{CLASS_LIST_FOR_PROMPT}\n\n"
        "Return ONLY a valid JSON object, no markdown:\n"
        '{"label": "<exact class from list above>", "confidence": 0.95}'
    )
    
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt_text})
    
    return [{"role": "user", "content": content}]
'''

def fuzzy_resolve(raw: str) -> str | None:
    """
    Try to map a raw VLM string to one of the 51 TSU class names.
    1. Exact match after normalising case, dots and spaces to underscores.
    2. Substring match in either direction.
    Returns None if nothing matches.
    """
    norm = raw.strip().lower().replace(".", "_").replace(" ", "_")

    # 1. Exact
    if norm in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[norm]

    # 2. Substring — prefer longer key to avoid spurious short matches
    best = None
    for key, canonical in _CANONICAL_LOWER.items():
        if norm in key or key in norm:
            if best is None or len(key) > len(_CANONICAL_LOWER.get(best, "")):
                best = canonical

    return best


def parse_vlm_response(response_text, tad_label, tad_score):
    try:
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        vlm_data = json.loads(clean_text)

        raw_label = vlm_data.get("label", "")
        confidence = vlm_data.get("confidence", tad_score)

        # Fuzzy resolve the label
        resolved = fuzzy_resolve(raw_label)
        if resolved:
            vlm_data["label"] = resolved
        else:
            # VLM returned something unresolvable — scan full response text as last resort
            resolved_from_text = fuzzy_resolve(response_text)
            if resolved_from_text:
                print(f"  [fuzzy] '{raw_label}' unresolvable from JSON, found '{resolved_from_text}' in response text.")
                vlm_data["label"] = resolved_from_text
            else:
                print(f"  [fuzzy] '{raw_label}' unresolvable — falling back to TAD label '{tad_label}'.")
                vlm_data["label"] = tad_label
                vlm_data["confidence"] = tad_score

        return vlm_data

    except json.JSONDecodeError:
        print(f"  [parse] Failed to parse JSON: {response_text!r}")
        return {"label": tad_label, "confidence": tad_score}


def deduplicate_segments(segments):
    seen = set()
    unique = []
    for seg in segments:
        key = (round(seg["segment"][0], 3), round(seg["segment"][1], 3), seg["label"])
        if key not in seen:
            seen.add(key)
            unique.append(seg)
    return unique

MIN_CONFIDENCE = 0.15  # drop anything below this

def nms_per_class(segments, iou_threshold=0.3):
    from collections import defaultdict
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
    # 1. Load Model and Processor (Adjust device/dtype as needed for your cluster)
    print("Loading VLM...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    # 2. Load TAD predictions
    with open(TAD_RESULTS_PATH, 'r') as f:
        tad_data = json.load(f)

    hybrid_results = {"results": {}}

    # 3. Main Hybrid Loop
    for video_id, segments in tad_data["results"].items():
        segments = deduplicate_segments(segments)
        segments = [s for s in segments if s["score"] >= MIN_CONFIDENCE]
        segments = nms_per_class(segments)
        print(f"  {len(segments)} segments after dedup + conf filter + NMS")

        t_vid = time.time()
        print(f"Processing video: {video_id}")
        video_path = f"{VIDEO_ROOT}/{video_id}.mp4"
        final_video_timeline = []

        for seg in segments:
            start, end = seg["segment"]
            tad_label = seg["label"]
            tad_score = seg["score"]

            # Guard against ultra-short segments that crash the video reader
            if end - start < MIN_SEGMENT_DURATION:
                # Keep the original TAD segment untouched, skip VLM
                final_video_timeline.append(seg)
                continue

            # Build messages and prepare inputs
            frames = extract_frames(video_path, start, end)
            if not frames:
                print(f"  [skip] No frames extracted for [{start:.1f}s - {end:.1f}s] — keeping TAD label.")
                final_video_timeline.append(seg)
                continue
            
            messages = build_messages(frames, start, end, tad_label, tad_score)
            #messages = build_messages(video_path, start, end, tad_label, tad_score)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Run Inference
            t0 = time.time()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Parse and merge logic
            vlm_decision = parse_vlm_response(output_text, tad_label, tad_score)
            
            # Assembly Logic
            final_seg = {
                "segment": [start, end],
                "label": vlm_decision.get("label", tad_label),
                "score": vlm_decision.get("confidence", tad_score),
            }
            
            final_video_timeline.append(final_seg)
            
            print(f"  [{start:.1f}s - {end:.1f}s] TAD: {tad_label} ({tad_score:.2f}) -> VLM: {final_seg['label']} ({final_seg['score']:.2f})")
        
        hybrid_results["results"][video_id] = final_video_timeline
        print(f"  Done in {time.time() - t_vid:.1f}s ({len(final_video_timeline)} segments)")

    # 4. Save Final Assembly
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(hybrid_results, f, indent=4)
    print(f"Saved hybrid results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()