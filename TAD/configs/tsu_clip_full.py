# Template. The placeholder @@WORK_DIR@@ is substituted by run_pipeline.sh.

_base_ = [
    "../_base_/datasets/tsu/features_clip_full.py",   # TSU dataset config
    "../_base_/models/actionformer.py",                # standard ActionFormer model
]

# CLIP ViT-B/32 produces 512-dim features. TSU has 51 classes. max_seq_len=2304
# matches trunc_len in the dataset config so the transformer's positional
# capacity is sufficient for the longest TSU videos.
model = dict(
    projection=dict(in_channels=512, arch=(2, 2, 5), max_seq_len=2304),
    rpn_head=dict(num_classes=51),
)

solver = dict(
    # OpenTAD requires (val|test).batch_size % world_size == 0 at the
    # dataloader-build assertion. tools/train.py also builds the test_loader
    # internally with world_size=N_GPUS, so we set both val and test batch_size
    # to 2 to keep 2-GPU training valid. Each rank still loads one sample per
    # step, so variable-length feature sequences are fine. The standalone
    # tools/test.py in step 6 is launched with nproc_per_node=N_GPUS so the
    # same constraint holds there.
    train=dict(batch_size=2, num_workers=4),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    clip_grad_norm=1,
    ema=True,
    amp=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR",
                 warmup_epoch=3, max_epoch=40)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    pre_nms_topk=2000,
    pre_nms_thresh=0.001,
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,
    ),
    save_dict=True,         # write predictions to work_dir/result_detection.json
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=5,        # save every 5 epochs
    val_loss_interval=1,
    val_eval_interval=5,          # evaluate mAP every 5 epochs, full eval is slow
    val_start_epoch=10,           # skip very early epochs
    end_epoch=40,
)

work_dir = "@@WORK_DIR@@"
