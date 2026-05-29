import os

# ── Environment ──────────────────────────────────────────────────────────────
HF_HOME = "/work/cs-503/sadgal/hf_models_cache"
os.environ["HF_HOME"] = HF_HOME

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# ── Dataset ──────────────────────────────────────────────────────────────────
VIDEO_DIR = "/work/cs-503/sadgal/Videos_mp4"
GT_DIR = "/work/cs-503/sadgal/Annotation"
DATASET_FPS = 25

# ── Annotation normalisation ─────────────────────────────────────────────────
CLASS_ALIASES = {
    "Make_coffee.Get_water": "Get_water",
    "Make_tea.Insert_tea_bag": "Insert_tea_bag",
}

# ── Sliding-window inference ─────────────────────────────────────────────────
WINDOW_SEC = 60          # length of each window in seconds
OVERLAP_SEC = 0          # overlap between consecutive windows in seconds
WINDOW_FPS = 35 / 60     # ~0.58 fps → one frame every 1.71 seconds
MAX_NEW_TOKENS = 512

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"

# ── Evaluation Intersection over Union thresholds ────────────────────────────
MAP_IOU_THRESHOLDS = [0.1, 0.3, 0.5]   # Event-mAP thresholds
CLASS_RECALL_IOU = 0.05                 # Per-class recall match threshold
DURATION_RECALL_IOU = 0.1               # Recall by event duration match threshold
HALLU_IOU = 0.05                        # Hallucination detection threshold

# ── Bins for event duration recall ───────────────────────────────────────────
DURATION_BINS = [
    ("<1s",    0,  1),
    ("1-2s",   1,  2),
    ("2-10s",  2, 10),
    ("10-30s", 10, 30),
    (">30s",   30, float("inf")),
]

# ── Activity classes (TSU ontology) ──────────────────────────────────────────
TSU_CLASSES = [
    "Breakfast", "Breakfast.Cut_bread", "Breakfast.Eat_at_table", 
    "Breakfast.Spread_jam_or_butter", "Breakfast.Take_ham", "Clean_dishes", 
    "Clean_dishes.Clean_with_water", "Clean_dishes.Dry_up", 
    "Clean_dishes.Put_something_in_sink", "Cook", "Cook.Cut", "Cook.Stir", 
    "Cook.Use_oven", "Cook.Use_stove", "Drink.From_bottle", "Drink.From_can", 
    "Drink.From_cup", "Drink.From_glass", "Dump_in_trash", "Eat_snack", 
    "Enter", "Get_up", "Get_water", "Insert_tea_bag", "Lay_down", "Leave", 
    "Make_coffee", "Make_coffee.Pour_grains", "Make_coffee.Pour_water", 
    "Make_tea", "Make_tea.Boil_water", "Pour.From_bottle", "Pour.From_can", 
    "Pour.From_kettle", "Put_something_on_table", "Read", "Sit_down", 
    "Stir_coffee/tea", "Take_pills", "Take_something_off_table", "Use_Drawer", 
    "Use_cupboard", "Use_fridge", "Use_glasses", "Use_laptop", "Use_tablet", 
    "Use_telephone", "Walk", "Watch_TV", "Wipe_table", "Write"
]
