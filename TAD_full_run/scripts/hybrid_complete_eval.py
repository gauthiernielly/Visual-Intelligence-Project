import json
import os
import sys
import pandas as pd

# --- Configuration ---
JSON_PATH = "hybrid_pipeline_results.json"
ANNOTATIONS_DIR = "../../../../work/cs-503/sadgal/Annotation"
OUTPUT_CSV = "hybrid_complete_evaluation_results.csv"

def clean_label(label):
    return str(label).strip().lower().replace(".", "_").replace(" ", "_")

def get_csv_sequence(video_id: str) -> list | None:
    subject_id = video_id[:3] 
    csv_path = os.path.join(ANNOTATIONS_DIR, subject_id, f"{video_id}.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path).sort_values(by="start_frame")
        return [clean_label(x) for x in df["event"].tolist()]
    except Exception:
        return None

def get_json_sequence(json_data: dict, video_id: str) -> list:
    if video_id not in json_data["results"]:
        return []
    segments = sorted(json_data["results"][video_id], key=lambda x: x["segment"][0])
    raw_sequence = [clean_label(seg["label"]) for seg in segments]
    
    # Collapse consecutive duplicates
    collapsed_seq = []
    for action in raw_sequence:
        if not collapsed_seq or collapsed_seq[-1] != action:
            collapsed_seq.append(action)
    return collapsed_seq

def calculate_lcs_metrics(gt_seq: list, pred_seq: list):
    """Returns Recall, Precision, and F1 Score based on LCS length."""
    if not gt_seq or not pred_seq:
        return 0.0, 0.0, 0.0
        
    m, n = len(gt_seq), len(pred_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_seq[i - 1] == pred_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    lcs_length = dp[m][n]
    
    recall = (lcs_length / m) * 100
    precision = (lcs_length / n) * 100
    
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return recall, precision, f1

def main():
    if not os.path.exists(JSON_PATH):
        print(f"[Error] Could not find {JSON_PATH}.")
        sys.exit(1)

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        
    video_ids = list(data.get("results", {}).keys())
    print(f"Found {len(video_ids)} processed videos. Starting evaluation...\n")
    
    results_list = []
    
    for video_id in video_ids:
        gt_seq = get_csv_sequence(video_id)
        pred_seq = get_json_sequence(data, video_id)
        
        if gt_seq and len(gt_seq) > 0:
            recall, precision, f1 = calculate_lcs_metrics(gt_seq, pred_seq)
            results_list.append({
                "video_id": video_id,
                "gt_count": len(gt_seq),
                "pred_count": len(pred_seq),
                "ratio_pred_gt": round(len(pred_seq) / len(gt_seq), 2),
                "recall": round(recall, 2),
                "precision": round(precision, 2),
                "f1_score": round(f1, 2)
            })
        
    df = pd.DataFrame(results_list)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("=" * 70)
    print(f"{'Video ID':<12} | {'GT':<4} | {'Pred':<5} | {'Recall':<8} | {'Precision':<10} | {'F1':<5}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['video_id']:<12} | {row['gt_count']:<4} | {row['pred_count']:<5} | {row['recall']:<7.1f}% | {row['precision']:<9.1f}% | {row['f1_score']:.1f}%")
    print("=" * 70)
    
    print(f"Global Average Recall    : {df['recall'].mean():.1f}%")
    print(f"Global Average Precision : {df['precision'].mean():.1f}%")
    print(f"Global Average F1-Score  : {df['f1_score'].mean():.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()