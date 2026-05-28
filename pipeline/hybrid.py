import os
import json
import torch
from typing import Optional
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import time

VIDEO_ROOT = "../../../../../work/cs-503/sadgal/Videos_mp4"
TAD_RESULTS_PATH = "../TAD_full_run/outputs/predictions_canonical.json"
OUTPUT_PATH = "../results/hybrid_pipeline_results.json"
MIN_SEGMENT_DURATION = 0.5  # seconds
NFRAMES_PER_CLIP = 8
MIN_CONFIDENCE = 0.15  # drop anything below this

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


def build_messages(video_path: str, start: float, end: float, tad_label: str, tad_score: float) -> list:
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


def fuzzy_resolve(raw: str) -> Optional[str]:
    """Safely map a raw VLM string to one of the 51 TSU class names."""
    norm = raw.strip().lower().replace(".", "_").replace(" ", "_")
    if norm in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[norm]

    best_canonical = None
    best_key_len = 0
    for key, canonical in _CANONICAL_LOWER.items():
        if norm == key:
            return canonical
        if f"_{norm}_" in f"_{key}_" or f"_{key}_" in f"_{norm}_":
            if len(key) > best_key_len:
                best_canonical = canonical
                best_key_len = len(key)
    return best_canonical


def parse_vlm_response(response_text: str, tad_label: str, tad_score: float) -> dict:
    try:
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        vlm_data = json.loads(clean_text)

        raw_label = vlm_data.get("label", "")

        # Fuzzy resolve the label from the JSON field
        resolved = fuzzy_resolve(raw_label)
        if resolved:
            vlm_data["label"] = resolved
        else:
            # VLM returned something unresolvable, we scan full response text as last resort
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


def deduplicate_segments(segments: list) -> list:
    seen = set()
    unique = []
    for seg in segments:
        key = (round(seg["segment"][0], 3), round(seg["segment"][1], 3), seg["label"])
        if key not in seen:
            seen.add(key)
            unique.append(seg)
    return unique


def nms_per_class(segments: list, iou_threshold: float = 0.3) -> list:
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
    # Model loading
    print("Loading VLM...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    # TAD predictions loading
    with open(TAD_RESULTS_PATH, 'r') as f:
        tad_data = json.load(f)

    # loading existing progress if available
    hybrid_results = {"results": {}}
    if os.path.exists(OUTPUT_PATH):
        print(f"Found existing progress in {OUTPUT_PATH}. Loading checkpoint...")
        with open(OUTPUT_PATH, 'r') as f:
            try:
                hybrid_results = json.load(f)
                print(f"Loaded {len(hybrid_results['results'])} already processed videos.")
            except json.JSONDecodeError:
                print("Warning: Existing JSON is corrupted or empty. Starting fresh.")

    for video_id, segments in tad_data["results"].items():

        segments = deduplicate_segments(segments)
        segments = [s for s in segments if s["score"] >= MIN_CONFIDENCE]
        segments = nms_per_class(segments)

        print(f"Processing video: {video_id} ({len(segments)} segments to check)")
        video_path = f"{VIDEO_ROOT}/{video_id}.mp4"

        # we skip entirely if the video file doesn't exist
        if not os.path.exists(video_path):
            print(f"  [skip] Video file not found: {video_path}")
            continue

        t_vid = time.time()

        # We ensure the video entry exists in results
        if video_id not in hybrid_results["results"]:
            hybrid_results["results"][video_id] = []

        completed_segments = {
            (round(s["segment"][0], 3), round(s["segment"][1], 3))
            for s in hybrid_results["results"][video_id]
        }

        for seg in segments:
            start, end = seg["segment"]
            tad_label = seg["label"]
            tad_score = seg["score"]
            seg_key = (round(start, 3), round(end, 3))

            # We skip already-processed segments
            if seg_key in completed_segments:
                print(f"  [skip] Segment [{start:.1f}s - {end:.1f}s] already processed.")
                continue

            # for short segments: we keep TAD result as-is
            if end - start < MIN_SEGMENT_DURATION:
                hybrid_results["results"][video_id].append(seg)
                completed_segments.add(seg_key)
                with open(OUTPUT_PATH, 'w') as f:
                    json.dump(hybrid_results, f, indent=4)
                print(f"  [{start:.1f}s - {end:.1f}s] Too short, kept TAD label: {tad_label} (Saved!)")
                continue

            try:
                messages = build_messages(video_path, start, end, tad_label, tad_score)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                vlm_decision = parse_vlm_response(output_text, tad_label, tad_score)

            except Exception as e:
                print(f"  [error] Inference failed for [{start:.1f}s - {end:.1f}s]: {e}. Falling back to TAD.")
                vlm_decision = {"label": tad_label, "confidence": tad_score}

            final_seg = {
                "segment": [start, end],
                "label": vlm_decision.get("label", tad_label),
                "score": vlm_decision.get("confidence", tad_score),
            }

            hybrid_results["results"][video_id].append(final_seg)
            completed_segments.add(seg_key)

            with open(OUTPUT_PATH, 'w') as f:
                json.dump(hybrid_results, f, indent=4)

            print(f"  [{start:.1f}s - {end:.1f}s] TAD: {tad_label} -> VLM: {final_seg['label']} (Saved!)")

        print(f"  Finished {video_id} in {time.time() - t_vid:.1f}s")

    print(f"Pipeline complete. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()