# ==========================================================================
# _base_palm/dataset_MS15_pooled.py
# --------------------------------------------------------------------------
# Pooled GE 15 cm + Aerial 15 cm training corpus for the Stage B
# benchmark, with GE-only validation. Aerial is held out for post-hoc
# evaluation via a separate eval-only config (Evaluation/evaluate_model.py --sensors Aerial)
# invoked through tools/test.py against the best GE-val checkpoint.
# Stage B only — not imported by any Stage A config.
#
# -----------------------------------------------------------------------
# EVALUATION DESIGN:
#   train_dataloader : ConcatDataset(GE_train + Aerial_train) with
#                       SensorBalancedSampler(alpha=0.3)
#   val_dataloader   : GE-only (single CocoDataset)
#   val_evaluator    : bare CocoMetric (no MultiDatasetsEvaluator
#                       wrapper required for single-source val)
#   test_dataloader  : mirrors val (MMDet convention; Evaluation/evaluate_model.py --sensors Aerial
#                       overrides this for the Aerial test pass)
#
# Emitted val metric key: 'coco/segm_mAP_50' (no namespace prefix).
# CheckpointHook.save_best and EarlyStoppingHook.monitor in
# runtime_palm_ms15.py both reference this key.
#
# -----------------------------------------------------------------------
# PROPOSAL_NUMS = (100, 300, 1000) — full COCO default:
#
# Empirical instance-density analysis of the GE val partition
# (761 tiles, 27,342 annotations):
#   max=373, mean=35.9, p95=127, p99=234
#   70/761 tiles (9.2%) exceed 100 instances
#
# At proposal_nums=(100,), recall is capped at the top-100 predictions
# per tile. The 9.2% of GE tiles exceeding this threshold have true
# positives systematically suppressed, biasing mAP@50/75 downward by a
# tile-density-dependent amount. Since transformer and SSM backbones
# may produce denser proposal distributions than CNNs, this cap
# introduces an architecture-dependent bias that contaminates the
# cross-backbone ranking the benchmark is designed to produce.
#
# The full COCO default (100, 300, 1000) is therefore retained.
# The cost increase per val pass is on the order of 5-15% (dominated
# by cocoeval recall computation, not the forward pass), which is
# acceptable given the benchmark-integrity requirement.
#
# -----------------------------------------------------------------------
# MACHINE CONFIGURATION — edit this section when deploying to a different
# workstation. No CLI flags are needed.
#
#   TITAN RTX (64 GB RAM)   : num_workers=2, pin_memory=True
#   RTX A5000 (31.9 GB RAM) : num_workers=1, pin_memory=False
#
# Rationale (num_workers): each DataLoader worker inherits a full copy
# of the GE COCO annotation index (~3 GB expanded Python object graph).
# At num_workers=2 the A5000 peaks at ~12 GB worker RAM before any GPU
# activity, causing swap thrashing (CUDA utilisation drops to 0%).
# num_workers=1 halves worker RAM to ~6 GB while remaining GPU-bound
# at batch_size=2. persistent_workers=True is always True so the worker
# is spawned once and reused across validation passes.
#
# Rationale (pin_memory): on the A5000 with system RAM near capacity
# (~94% utilisation), pin_memory=True caused non-deterministic CUDA
# context corruption (RuntimeError: CUDA error: unknown error) when the
# OS could not honour pinned-page-lock requests. pin_memory=False is
# safe on both machines; pin_memory=True is retained on the TITAN RTX
# (64 GB RAM) where the pinning budget is sufficient.
# -----------------------------------------------------------------------
num_workers = 2      # <-- set to 1 on the RTX A5000 workstation
pin_memory  = True   # <-- set to False on the RTX A5000 workstation

dataset_type = 'CocoDataset'
classes = ('DatePalm',)

ge_root = '/workspace/datasets/COCO/GE_15cm/'
aerial_root = '/workspace/datasets/COCO/Aerial_15cm/'

# --- Pipelines ------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape',
                   'img_shape', 'scale_factor'),
    ),
]

# --- Per-source train dataset wrappers ------------------------------------
ge_train_dataset = dict(
    type=dataset_type,
    data_root=ge_root,
    ann_file='Annotations/train_GE.json',
    data_prefix=dict(img='train_GE/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=0),
    metainfo=dict(classes=classes),
    pipeline=train_pipeline,
)

aerial_train_dataset = dict(
    type=dataset_type,
    data_root=aerial_root,
    ann_file='Annotations/train_aerial.json',
    data_prefix=dict(img='train_aerial/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=0),
    metainfo=dict(classes=classes),
    pipeline=train_pipeline,
)

# --- Val dataset (GE only) ------------------------------------------------
ge_val_dataset = dict(
    type=dataset_type,
    data_root=ge_root,
    ann_file='Annotations/val_GE.json',
    data_prefix=dict(img='val_GE/'),
    metainfo=dict(classes=classes),
    pipeline=test_pipeline,
    test_mode=True,
)

# --- Train dataloader -----------------------------------------------------
# IMPORTANT: the order of datasets=[ge_train_dataset, aerial_train_dataset]
# defines sensor identity for the SensorBalancedSampler. Reordering without
# updating sampler.sensor_names will silently invert the rebalancing target.
train_dataloader = dict(
    batch_size=2,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=pin_memory,
    prefetch_factor=2,
    sampler=dict(
        type='SensorBalancedSampler',
        alpha=0.3,
        sensor_names=('GE', 'Aerial'),
        seed=0,
    ),
    dataset=dict(
        type='ConcatDataset',
        datasets=[ge_train_dataset, aerial_train_dataset],
    ),
)

# --- Val dataloader (GE only) ---------------------------------------------
val_dataloader = dict(
    batch_size=2,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=pin_memory,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ge_val_dataset,
)

# --- Val evaluator (bare CocoMetric) --------------------------------------
# Single-source GE validation: no MultiDatasetsEvaluator wrapper is
# needed. The emitted key is 'coco/segm_mAP_50', referenced by
# CheckpointHook.save_best and EarlyStoppingHook.monitor in the runtime.
val_evaluator = dict(
    type='CocoMetric',
    ann_file=ge_root + 'Annotations/val_GE.json',
    metric=['bbox', 'segm'],
    classwise=True,
    format_only=False,
    proposal_nums=(100, 300, 1000),
)

# --- Test dataloader / evaluator ------------------------------------------
# MMDet convention: mirror val by default. Aerial post-hoc evaluation
# is performed through Evaluation/evaluate_model.py --sensors Aerial which overrides these two
# keys to point at the Aerial partition. This keeps the training
# launch lightweight (no Aerial workers spawned at runner init) and
# isolates the "test" semantics to an explicit eval-only config.
test_dataloader = dict(
    batch_size=4,
    num_workers=num_workers,
    #persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    pin_memory_device='cuda' if pin_memory else '',
    #prefetch_factor=prefetch_factor,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,            # single CocoDataset — patch_config-compatible
        #metainfo=metainfo,
        data_root=ge_root,
        ann_file='Annotations/test_GE.json',
        data_prefix=dict(img='test_GE/'),
        test_mode=True,
        pipeline=test_pipeline,
        serialize_data=True,
        #backend_args=backend_args,
    ),
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=ge_root + 'Annotations/test_GE.json',
    metric=['bbox', 'segm'],
    classwise=True,
    format_only=False,
    proposal_nums=(100, 300, 1000),
)