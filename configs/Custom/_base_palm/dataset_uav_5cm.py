# ==========================================================================
# _base_palm/dataset_uav_5cm.py
# --------------------------------------------------------------------------
# Shared UAV 5 cm dataset definition for the date palm backbone benchmark.
# This file is detector-agnostic — it defines dataloaders, pipelines, and
# evaluators only, and is imported by both SOLOv2 and Mask R-CNN configs.
#
# Expected dataset layout:
#   /workspace/datasets/UAV_5cm/
#     ├── train_UAV/              ← 1024x1024 tiles, 25% overlap (16,800)
#     ├── val_UAV/                ← 1024x1024 tiles, 0% overlap   (2,230)
#     ├── test_UAV/               ← 1024x1024 tiles, 0% overlap   (2,450)
#     └── Annotations/
#           ├── train_UAV.json
#           ├── val_UAV.json
#           └── test_UAV.json
# =============================================]\1=============================

dataset_type = 'CocoDataset'
data_root = '/workspace/datasets/COCO/UAV_5cm/'
classes = ('DatePalm',)

# --- Pipelines --------------------------------------------------------------
# Tiles are already 1024x1024 on disk, so no Resize is strictly required.
# A Resize + Pad pair is retained for robustness against occasional
# off-by-one tile dimensions produced by raster clipping.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # Pad guarantees a fixed 1024x1024 input after keep_ratio resizing.
    # Required for FPN stride-32 alignment and for stable small-object
    # feature extraction at the highest-resolution pyramid level (P2).
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # Pad must mirror the training pipeline to avoid a train/test input
    # geometry mismatch on non-square edge tiles.
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape',
                   'img_shape', 'scale_factor'),
    ),
]

# --- Dataloaders ------------------------------------------------------------
# batch_size=2 is the VRAM-limited maximum on a 24 GB TITAN RTX at 1024x1024
# for SOLOv2 and Mask R-CNN with mid-capacity backbones. Effective batch of 4
# is achieved via accumulative_counts=2 in the optimiser wrapper.
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='Annotations/train_UAV.json',
        data_prefix=dict(img='train_UAV/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='Annotations/val_UAV.json',
        data_prefix=dict(img='val_UAV/'),
        metainfo=dict(classes=classes),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='Annotations/test_UAV.json',
        data_prefix=dict(img='test_UAV/'),
        metainfo=dict(classes=classes),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

# --- Evaluators -------------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'Annotations/val_UAV.json',
    metric=['bbox', 'segm'],
    classwise=True,
    format_only=False,
    # COCO convention: AR@100 / AR@300 / AR@1000. Using (100, 300, 300)
    # double-counts the 300 slot and suppresses AR@1000, which matters
    # for dense palm plantation tiles.
    proposal_nums=(100, 300, 1000),
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'Annotations/test_UAV.json',
    metric=['bbox', 'segm'],
    classwise=True,
    format_only=False,
    proposal_nums=(100, 300, 1000),
)


