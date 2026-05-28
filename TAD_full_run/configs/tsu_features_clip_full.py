# Template. The placeholders @@ANN_FILE@@, @@CLASS_MAP@@ and @@DATA_PATH@@
# are substituted by run_pipeline.sh before being copied into the OpenTAD
# config tree.

dataset_type    = "ThumosPaddingDataset"
annotation_path = "@@ANN_FILE@@"
class_map       = "@@CLASS_MAP@@"
data_path       = "@@DATA_PATH@@"
block_list      = None

trunc_len = 2304   # CLIP at stride 16 and 25 fps gives 0.64 s/snippet, so 2304
                   # is about 24 minutes and covers the longest TSU videos.

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=16,
        sample_stride=1,
        offset_frames=0,
        fps=25,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=trunc_len, trunc_thresh=0.5,
                 crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats",
                 keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=16,
        sample_stride=1,
        offset_frames=0,
        fps=25,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            # No padding here. We feed variable-length sequences at batch_size=1
            # to match Multi-THUMOS and avoid truncating long TSU videos.
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats",
                 keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="testing",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=16,
        sample_stride=1,
        offset_frames=0,
        fps=25,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.5, 0.7],
    ground_truth_filename=annotation_path,
)
