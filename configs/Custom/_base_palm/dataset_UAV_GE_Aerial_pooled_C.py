# Copyright (c) Stage C Date Palm Benchmark.
# dataset_UAV_GE_Aerial_pooled_C.py  (v3 - production-grade)
# ==========================================================================
# Three-source pooled training dataset with source-local batch construction
# for kernel page-cache locality across the UAV / GE / Aerial mounts.
#
# WHAT CHANGED RELATIVE TO v2
# --------------------------------------------------------------------------
#  1. batch_size ships as 1 with sampler chunk_size 4 (the fp32-safe
#     setting; AMP runs may raise batch_size as long as chunk_size stays a
#     multiple of it).
#  2. chunk_size now tracks batch_size via a single variable (no silent drift
#     between the two if batch_size is edited).
#  3. val_dataloader / test_dataloader use persistent_workers=False so their
#     workers are torn down after each pass, freeing resident RAM between the
#     5k-interval validations. Train workers stay persistent for throughput.
#  4. Machine knobs consolidated into ONE clearly-marked block.
#
# serialize_data=True is kept on every sub-dataset: the data_list is packed
# into a single shared-memory buffer once, so the multi-GB COCO index is NOT
# duplicated per worker (this is what keeps DataLoader RAM bounded and is the
# correct mitigation for the worker-RAM inflation noted on the A5000).

# ==========================================================================
# PER-MACHINE KNOBS  --  edit these THREE values when switching workstations.
#   TITAN RTX (64 GB RAM) : num_workers=2, pin_memory=True,  prefetch_factor=4
#   RTX A5000 (16 GB ctr) : num_workers=1, pin_memory=False, prefetch_factor=2
# ==========================================================================
num_workers     = 2
pin_memory      = True
prefetch_factor = 4

# Universally beneficial / do not change without reason:
persistent_workers = True   # TRAIN workers only; val/test set False below.
batch_size = 1              # fp32-safe default; see note 1 in the header.
chunk_size = 4     # MUST be a multiple of batch_size so each batch is one source.

# ==========================================================================
# Data roots -- point these at YOUR data layout, and keep them consistent
# with the roots the evaluation uses (configs/Custom/Evaluation/
# sensor_registry.py defaults to /workspace/datasets/COCO/...; this file
# was last used on a machine that mounted the same trees under
# /root/datasets/). The two must name the same trees or training and
# evaluation silently read different data.
# ==========================================================================
uav_root    = '/root/datasets/COCO/UAV_5cm/'
ge_root     = '/root/datasets/COCO/GE_15cm/'
aerial_root = '/root/datasets/COCO/Aerial_15cm/'

dataset_type = 'CocoDataset'
metainfo = dict(classes=('DatePalm',), palette=[(220, 20, 60)])
backend_args = None

# ==========================================================================
# Pipelines  --  ALIGNED TO STAGE B (dataset_MS15_pooled.py).
#
# v2 used keep_ratio=False with a single (horizontal) RandomFlip and no
# photometric augmentation. That diverged from Stage B on two axes that are
# NOT no-ops and that bear directly on the cross-sensor objective of Stage C:
#   - Stage B applies RandomFlip in BOTH horizontal and vertical directions
#     (p=0.5 each). Vertical flip is well-motivated for nadir imagery, where
#     there is no canonical "up".
#   - Stage B applies PhotoMetricDistortion (brightness/contrast/saturation/
#     hue jitter), the primary mechanism for radiometric robustness across the
#     pooled UAV / GE / Aerial sources.
# Both are now restored so the Stage C training condition matches Stage B.
#
# The Resize(keep_ratio=True) + Pad(1024,1024) pair also mirrors Stage B. For
# tiles already at 1024^2 this is a no-op; it only affects occasional
# off-by-one edge tiles, where it guarantees a fixed 1024^2 input (FPN
# stride-32 alignment) and a consistent train/test input geometry.
# ==========================================================================
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'),
    ),
]

# ==========================================================================
# Training datasets  --  three sources, ConcatDataset.
# ORDER MATTERS: it defines source indices used by SensorBalancedSamplerN.
#   Index 0: UAV    (5 cm,  ~16,800 tiles)
#   Index 1: GE     (15 cm, ~19,472 tiles)
#   Index 2: Aerial (15 cm, ~1,204  tiles)
# ==========================================================================
train_dataset_uav = dict(
    type=dataset_type, metainfo=metainfo, data_root=uav_root,
    ann_file='Annotations/train_UAV.json', data_prefix=dict(img='train_UAV/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline, serialize_data=True, backend_args=backend_args,
)
train_dataset_ge = dict(
    type=dataset_type, metainfo=metainfo, data_root=ge_root,
    ann_file='Annotations/train_GE.json', data_prefix=dict(img='train_GE/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline, serialize_data=True, backend_args=backend_args,
)
train_dataset_aerial = dict(
    type=dataset_type, metainfo=metainfo, data_root=aerial_root,
    ann_file='Annotations/train_aerial.json', data_prefix=dict(img='train_aerial/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline, serialize_data=True, backend_args=backend_args,
)

# ==========================================================================
# Worker-thread clamping  --  NOTE on worker_init_fn.
# --------------------------------------------------------------------------
# MMEngine's Runner.build_dataloader does NOT accept a user-supplied
# worker_init_fn through the dataloader config: it always constructs its own
# (from mmengine.dataset.worker_init_fn) for per-worker seeding, and it
# asserts callable(worker_init_fn) on whatever key remains, so passing a
# dict(type='function', qualname=...) raises AssertionError at loop build
# (see open-mmlab/mmengine issue #933). The cv2/OMP/MKL thread clamp that
# dataloader_worker_init.py provided is therefore routed through the channels
# MMEngine DOES honour:
#   - OpenCV threads: env_cfg['mp_cfg']['opencv_num_threads']=1 in the runtime.
#   - OMP/MKL threads: exported in the shell before launch (see header /
#     STAGE_C_REDESIGN.md): `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
# No worker_init_fn key is set on the dataloaders below.
# ==========================================================================

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    pin_memory_device='cuda' if pin_memory else '',
    prefetch_factor=prefetch_factor,
    sampler=dict(
        type='SensorBalancedSamplerN',
        weights=[1.0, 1.0, 1.0],   # proportional sampling -- see methods
        chunk_size=chunk_size,     # == batch_size: one source per batch
    ),
    batch_sampler=None,            # AspectRatioBatchSampler would cross-mix sources
    dataset=dict(
        type='ConcatDataset',
        datasets=[train_dataset_uav, train_dataset_ge, train_dataset_aerial],
        ignore_keys=['classes', 'palette'],
    ),
)

# ==========================================================================
# Validation  --  UAV_val + GE_val, separate CocoMetric per source.
# persistent_workers=False so val workers are released after each pass.
# ==========================================================================
val_dataset_uav = dict(
    type=dataset_type, metainfo=metainfo, data_root=uav_root,
    ann_file='Annotations/val_UAV.json', data_prefix=dict(img='val_UAV/'),
    test_mode=True, pipeline=test_pipeline, serialize_data=True,
    backend_args=backend_args,
)
val_dataset_ge = dict(
    type=dataset_type, metainfo=metainfo, data_root=ge_root,
    ann_file='Annotations/val_GE.json', data_prefix=dict(img='val_GE/'),
    test_mode=True, pipeline=test_pipeline, serialize_data=True,
    backend_args=backend_args,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=False,      # release between 5k-interval val passes
    pin_memory=pin_memory,
    pin_memory_device='cuda' if pin_memory else '',
    prefetch_factor=2,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[val_dataset_uav, val_dataset_ge],
        ignore_keys=['classes', 'palette'],
    ),
)

val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(type='CocoMetric',
             ann_file=uav_root + 'Annotations/val_UAV.json',
             metric=['bbox', 'segm'], format_only=False,
             proposal_nums=(100, 300, 1000), backend_args=backend_args),
        dict(type='CocoMetric',
             ann_file=ge_root + 'Annotations/val_GE.json',
             metric=['bbox', 'segm'], format_only=False,
             proposal_nums=(100, 300, 1000), backend_args=backend_args),
    ],
    dataset_prefixes=['UAV', 'GE'],
)

# ==========================================================================
# Test loop  --  single CocoDataset placeholder, rewritten at runtime by
# metrics_engine.patch_config() (which requires a single CocoDataset, not a
# ConcatDataset). persistent_workers=False here too.
# ==========================================================================
test_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=False,
    pin_memory=pin_memory,
    pin_memory_device='cuda' if pin_memory else '',
    prefetch_factor=2,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, metainfo=metainfo, data_root=ge_root,
        ann_file='Annotations/test_GE.json', data_prefix=dict(img='test_GE/'),
        test_mode=True, pipeline=test_pipeline, serialize_data=True,
        backend_args=backend_args,
    ),
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=ge_root + 'Annotations/test_GE.json',
    metric=['bbox', 'segm'], format_only=False,
    proposal_nums=(100, 300, 1000), backend_args=backend_args,
)
