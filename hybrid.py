import json
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration # Adjust import if using a specific Qwen3 class
from qwen_vl_utils import process_vision_info
import time

# --- Configuration ---
VIDEO_ROOT = "../../../work/cs-503/sadgal/Videos_mp4/"
TAD_RESULTS_PATH = "TAD_full_run/outputs/predictions_canonical.json"
OUTPUT_PATH = "hybrid_pipeline_results.json"
MIN_SEGMENT_DURATION = 0.5  # seconds
NFRAMES_PER_CLIP = 8

def build_messages(video_path, start, end, tad_label, tad_score):
    """
    Constructs the structured prompt for Qwen3-VL, natively trimming via the video dict.
    """
    prompt_text = (
        f"This clip runs from {start:.1f} to {end:.1f} seconds. "
        f"An action detection model predicted the activity is '{tad_label}' (confidence {tad_score:.2f}).\n\n"
        "Does this match what you see? Return ONLY a valid JSON object in this exact format:\n"
        '{"label": "the correct activity you see", "confidence": 0.95, "agrees_with_tad": true}'
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

def parse_vlm_response(response_text, tad_label, tad_score):
    """
    Safely parses the VLM's JSON response and handles fallback logic.
    """
    try:
        # Strip out markdown formatting if the model wraps the JSON
        clean_text = response_text.replace("```json", "").replace("", "").strip()
        vlm_data = json.loads(clean_text)
        return vlm_data
    except json.JSONDecodeError:
        print(f"Failed to parse VLM response as JSON: {response_text}")
        # Fallback to TAD if the VLM hallucinates formatting
        return {"label": tad_label, "confidence": tad_score, "agrees_with_tad": True}

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
            messages = build_messages(video_path, start, end, tad_label, tad_score)
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
                "vlm_agreed": vlm_decision.get("agrees_with_tad", True)
            }
            final_video_timeline.append(final_seg)
            
            print(f"  [{start:.1f}s - {end:.1f}s] TAD: {tad_label} ({tad_score:.2f}) -> VLM: {final_seg['label']} ({final_seg['score']:.2f})")

        hybrid_results["results"][video_id] = final_video_timeline

    # 4. Save Final Assembly
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(hybrid_results, f, indent=4)
    print(f"Saved hybrid results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()