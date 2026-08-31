# ==========================================================================
# maskrcnn_spatialmamba_s_finetune_hn.py
# --------------------------------------------------------------------------
# Operational hard-negative fine-tune of the deployed Stage C unified model
# (Spatial-Mamba-S, best_GE) to suppress desert false positives — palm-like
# shrubs and native trees (ghaf, acacia) that the Stage C training corpus
# never contained as labelled background.
#
# This is a DEPLOYMENT ADAPTATION, not a change to the benchmark. The Stage C
# checkpoints and their reported metrics are untouched; this produces a
# separate "Stage C + hard-negative adaptation" model used only for the
# country-scale inventory. Describe it in the manuscript in one paragraph as
# an operational post-processing step; keep it out of the backbone comparison.
#
# METHOD (why this works and is standard practice)
# --------------------------------------------------------------------------
# Mask R-CNN trains on empty images natively: on a tile with zero annotations
# every RPN anchor and every RoI proposal is an unmatched negative, so the
# tile is pure "nothing here is a palm" supervision for the two classification
# heads; the box/mask regression losses receive no positives and are inert.
# The benchmark's filter_empty_gt=True merely SKIPS such tiles — it is not a
# framework limit. Here we add a fourth source of hard-negative tiles with
# filter_empty_gt=False, so those tiles train the classifier against the exact
# confusers it fails on. This is hard-negative mining / active learning, the
# routine production fix for a domain-specific false-positive mode.
#
# CATASTROPHIC-FORGETTING GUARD
# --------------------------------------------------------------------------
# Fine-tuning only on negatives would erode recall. We therefore REPLAY the
# original GE-15cm positive training data alongside the negatives, at a
# controlled ratio (HN_WEIGHT), and validate on the untouched GE val split
# every VAL_INTERVAL iters. Watch GE recall: if it drops materially, lower
# HN_WEIGHT or shorten training.
#
# GE VAL IS ONLY HALF THE GATE, AND IT IS THE HALF THAT CANNOT SEE THIS WORK.
# GE val is farmland and holds no desert confusers, so the metric cannot move
# when the model stops hallucinating palms in the desert. In round 1 it did
# not move: the last seven evaluations were identical to four decimal places,
# and save_best therefore picked a checkpoint by noise. It happened to pick a
# good one. Do not rely on that twice. Choose the checkpoint by running
#   configs/Custom/Finetune_HN/eval_hard_negatives.py
# over the holdout tiles, then confirm recall on GE val with tools/test.py.
#
# RADIOMETRY CONSISTENCY (see make_hard_negative_coco.py header)
# --------------------------------------------------------------------------
# The negative tiles MUST be preprocessed identically to GE_train. GE_train was
# NOT contrast-stretched (the tiling pipeline stretched WorldView-3 only), and
# the country inference pipeline also does not stretch uint8 GE imagery — so
# training, negatives, and inference are all raw uint8, consistent. Generate
# the negatives with --stretch none (the default). Do NOT stretch GE negatives:
# that would create the very mismatch we are avoiding.
#
# CODEC CONSISTENCY (as important as the radiometry rule above)
# --------------------------------------------------------------------------
# GE_train is JPEG (COCO/GE_15cm/train_GE/JPEGImages, 19,472 tiles). If the
# negatives are written as lossless GeoTIFF, the classifier can separate the
# two sources by compression artefact alone -- 8x8 DCT blocking and chroma
# subsampling -- and satisfy the loss with "clean image = not a palm" without
# ever learning shrub rejection. Generate the negatives with
#   make_aoi_tiles.py --format jpg --jpeg-ref <a train_GE .jpg>
# so they inherit the corpus quantisation tables exactly.
#
# WHAT THIS RUN CAN AND CANNOT MEASURE
# --------------------------------------------------------------------------
# Validation is GE val only, and GE val contains no desert confusers. So the
# best-checkpoint gate proves ONLY that nothing was forgotten -- it does NOT
# show that the false positives went away. Empty-annotation tiles also cannot
# be scored by CocoMetric (no ground truth => no mAP), so there is no
# in-training metric for the fix itself. The fix is measured OPERATIONALLY,
# after the run: re-infer over the same AOIs and re-mine with
#   make_aoi_tiles.py --detections <new predictions> --dry-run
# and compare the reported false-positive count against the baseline.
#
# WHAT YOU MUST SET (three paths + two knobs below):
#   HN_ROOT_R1/R2 : hard-negative dataset dirs (make_aoi_tiles.py --emit-coco
#                   empty). Mine each round with --exclude pointing at the
#                   earlier rounds AND at the holdout, or the sets overlap:
#                   selection ranks by false-positive count and is
#                   deterministic, so a different --seed reproduces nearly the
#                   same tiles. A first attempt at a holdout set this way
#                   returned 1,070 of the 1,074 tiles already trained on.
#   GE_ROOT       : the real GE 15cm COCO root (positives to replay)
#   load_from     : the best_GE Stage C checkpoint to adapt
#   HN_WEIGHT     : COMPUTE IT -- see the formula at the knob below
#   max_iters     : long enough to bracket the optimum; the optimum itself
#                   is then MEASURED, not assumed -- see the knob below
# ==========================================================================

# ONE base only. maskrcnn_spatialmamba_s_stagec.py ALREADY inherits
# _base_maskrcnn_palm_stagec.py (plus the pooled Stage C dataset and runtime),
# so naming the detector base here as well made two sibling bases define
# `model` and MMEngine refused the config outright:
#     KeyError: Duplicate key is not allowed among bases. Duplicate keys: {'model'}
# Inheriting the Stage C config alone brings the detector, the Spatial-Mamba-S
# backbone and the FPN, which is everything this adaptation needs.
_base_ = ['../maskrcnn_palm_stagec/maskrcnn_spatialmamba_s_stagec.py']

# --------------------------------------------------------------------------
# OPTIONAL surgical mode (uncomment to freeze the backbone entirely). Hard-
# negative suppression is fundamentally a CLASSIFIER problem — the RPN/RoI
# heads decide palm-vs-background — so freezing the backbone + FPN and adapting
# only the heads is the lowest-risk, fastest-forgetting-proof option. Leave it
# commented to allow gentle feature adaptation (backbone lr x0.1, below); use
# it if you see any recall drop on GE val.
# --------------------------------------------------------------------------
# model = dict(backbone=dict(frozen_stages=4))   # freeze all 4 SSM stages

# --------------------------------------------------------------------------
# EDIT THESE
# --------------------------------------------------------------------------
GE_ROOT    = '/workspace/datasets/COCO/GE_15cm/'
# Two mining rounds, kept as separate folders rather than merged. Merging
# destroys the record of which tiles trained which model and breaks the
# --exclude bookkeeping that guarantees the holdout is disjoint. Folder names
# state the ROLE, not a version number, so nothing has to be remembered.
HN_ROOT_R1 = '/workspace/datasets/COCO/HardNeg/round1_train/'   # 3,083 tiles
HN_ROOT_R2 = '/workspace/datasets/COCO/HardNeg/round2_train/'   # 2,368 tiles
# never trained on; used only by eval_hard_negatives.py:
#   /workspace/datasets/COCO/HardNeg/holdout_eval/               2,098 tiles

# Restart from the Stage C base, NOT from the round-1 adapted checkpoint. Both
# negative sets are then new to the model, one run yields one model, and the
# method section describes a single adaptation instead of a chain. The learning
# saturates within a few thousand iterations, so restarting costs almost
# nothing.
load_from = ('work_dirs/Stage_C/maskrcnn_spatialmamba_s_stagec/'
             'best_GE_segm_mAP_50_iter_75001.pth')

# ---- HN_WEIGHT is a MULTIPLIER ON NATURAL SIZE, not a share ---------------
# SensorBalancedSamplerN allocates
#       quota_s = w_s * N_s / sum_t(w_t * N_t) * sum_t(N_t)
# so the weight scales each source's EXISTING size. For a target negative
# share p over a pool of N_neg tiles:
#       HN_WEIGHT = (p / (1 - p)) * (N_positives / N_neg)
# Measured: N_pos = 19472 (train_GE), N_neg = 3083 + 2368 = 5451, p = 0.25
#       -> (0.25/0.75) * (19472/5451) = 1.19
# The SAME weight goes on both rounds, which pools them: every negative tile
# is then equally likely to be drawn, and the larger round contributes
# proportionally more. Weighting the two rounds equally instead would
# over-sample the smaller one tile for tile, which is a claim about round 2
# being more informative that nothing in the data supports.
# Check the realised share against the sampler's first log lines: GE_train is
# filtered by filter_empty_gt/min_size, so its effective N is a little below
# 19472.
HN_WEIGHT = 1.19

# ---- max_iters: long enough to bracket the optimum, then MEASURE it -------
# The coverage rule (max_iters ~= N_negatives / p, here 5451/0.25 = 21.8k)
# is wrong for this problem. Round 1 measured the actual behaviour: the best
# false-positive suppression came at iteration 4000, having drawn only ~1000
# negative samples, and iteration 13000 was twice as DIRTY at identical
# recall. Hard-negative adaptation saturates early and then decays as the 75%
# positive replay pulls the classifier back toward detecting.
#
# So run long enough to contain the optimum and keep every checkpoint, then
# pick by measurement on the holdout set. Do NOT trust save_best here: it
# selects on GE val mAP, which in round 1 was identical to four decimal places
# across the last seven evaluations and therefore chose a checkpoint by noise.
max_iters    = 10000
val_interval = 1000

work_dir = 'work_dirs/Finetune_HN/spatialmamba_s_hn_round2'

# --------------------------------------------------------------------------
# The base backbone config replaces custom_imports / custom_hooks wholesale
# (MMEngine list-key replacement), so we must reproduce the module imports
# this fine-tune needs. Drop the training-only hooks (early stopping on the
# 2-sensor mean, per-sensor best) and keep just what a short GE-only
# validation loop needs.
# --------------------------------------------------------------------------
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.sensor_balanced_sampler_n',
        'configs.Custom._base_palm.nms_fp32_guard',
        'mmdet.models.backbones.spatialmamba_backbone',
    ],
    allow_failed_imports=False,
)

default_scope = 'mmdet'
metainfo = dict(classes=('DatePalm',), palette=[(220, 20, 60)])
backend_args = None

# --------------------------------------------------------------------------
# Pipelines — identical to Stage C (aligned augmentation), so the adapted
# model sees the same input geometry/radiometry it will meet at inference.
# --------------------------------------------------------------------------
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
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor')),
]

# --------------------------------------------------------------------------
# Source 0: GE positives (replay — prevents forgetting). filter_empty_gt=True.
# Source 1: hard negatives.               filter_empty_gt=FALSE  <-- the point.
# --------------------------------------------------------------------------
train_dataset_ge_pos = dict(
    type='CocoDataset', metainfo=metainfo, data_root=GE_ROOT,
    ann_file='Annotations/train_GE.json', data_prefix=dict(img='train_GE/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline, serialize_data=True, backend_args=backend_args,
)
train_dataset_hn_r1 = dict(
    type='CocoDataset', metainfo=metainfo, data_root=HN_ROOT_R1,
    ann_file='annotations/hard_neg.json', data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),   # KEEP empty tiles = negatives
    pipeline=train_pipeline, serialize_data=True, backend_args=backend_args,
)
train_dataset_hn_r2 = dict(
    type='CocoDataset', metainfo=metainfo, data_root=HN_ROOT_R2,
    ann_file='annotations/hard_neg.json', data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline, serialize_data=True, backend_args=backend_args,
)

train_dataloader = dict(
    batch_size=1,               # Spatial-Mamba-S memory (Stage C setting)
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    pin_memory_device='cuda',
    prefetch_factor=4,
    sampler=dict(
        type='SensorBalancedSamplerN',
        weights=[1.0, HN_WEIGHT, HN_WEIGHT],  # positives : r1 : r2
        chunk_size=4,               # == effective batch (accumulate 4)
    ),
    batch_sampler=None,
    dataset=dict(
        type='ConcatDataset',
        datasets=[train_dataset_ge_pos, train_dataset_hn_r1,
                  train_dataset_hn_r2],
        ignore_keys=['classes', 'palette'],
    ),
)

# --------------------------------------------------------------------------
# Validate on the UNTOUCHED GE val split — the forgetting monitor.
# --------------------------------------------------------------------------
# _delete_=True is REQUIRED here. Stage C's val dataset is a ConcatDataset
# (UAV + GE); MMEngine merges dicts key-by-key WITHOUT checking `type`, so
# without it this CocoDataset would silently inherit `datasets` and
# `ignore_keys` and blow up on unexpected kwargs at build time.
val_dataset_ge = dict(
    _delete_=True,
    type='CocoDataset', metainfo=metainfo, data_root=GE_ROOT,
    ann_file='Annotations/val_GE.json', data_prefix=dict(img='val_GE/'),
    test_mode=True, pipeline=test_pipeline, serialize_data=True,
    backend_args=backend_args,
)
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=False,
    pin_memory=True, pin_memory_device='cuda', prefetch_factor=2,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset_ge,
)
# _delete_=True for the same reason: Stage C's evaluator is a
# MultiDatasetsEvaluator, and its `metrics` / `dataset_prefixes` keys would
# otherwise survive into this single CocoMetric.
val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=GE_ROOT + 'Annotations/val_GE.json',
    metric=['bbox', 'segm'], format_only=False,
    proposal_nums=(100, 300, 1000), backend_args=backend_args,
)
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# --------------------------------------------------------------------------
# Schedule — SHORT and LOW-LR: adaptation, not training. AdamW at 1e-5 (10x
# below Stage C's 1e-4), backbone frozen-ish via the 0.1 multiplier, cosine
# to near-zero over a few thousand iters. Effective batch 4 (accumulate 4)
# matches Stage C so gradient statistics are comparable.
# --------------------------------------------------------------------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters,
                 val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=200),
    dict(type='CosineAnnealingLR', T_max=max_iters - 200, eta_min=1e-7,
         by_epoch=False, begin=200, end=max_iters),
]

optim_wrapper = dict(
    type='AmpOptimWrapper', dtype='float16',
    accumulative_counts=4,
    optimizer=dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999),
                   weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={
        'backbone':                     dict(lr_mult=0.1),
        '.bias':                        dict(decay_mult=0.0),
        '.norm':                        dict(decay_mult=0.0),
        'absolute_pos_embed':           dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'A_log':                        dict(decay_mult=0.0),
        'D':                            dict(decay_mult=0.0),
        'dt_proj':                      dict(decay_mult=0.0),
    }),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
auto_scale_lr = dict(enable=False, base_batch_size=4)

# --------------------------------------------------------------------------
# Hooks — periodic + best-on-GE checkpoints (the ship gate), by iteration.
# --------------------------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # Keep EVERY checkpoint. The stopping point is chosen afterwards by
    # measuring false positives on the holdout tiles, so discarding
    # intermediate checkpoints would throw away the candidates. save_best is
    # retained only as a label; it selects on GE val mAP, which cannot see
    # this adaptation and in round 1 chose by noise.
    checkpoint=dict(type='CheckpointHook', by_epoch=False,
                    interval=val_interval, max_keep_ckpts=-1,
                    save_best='coco/segm_mAP_50', rule='greater',
                    save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

# No training-only hooks needed for a short adaptation (the nms_fp32_guard
# module patches batched_nms on import; nothing else required).
custom_hooks = []

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=1),
    dist_cfg=dict(backend='nccl'),
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
log_level = 'INFO'
resume = False
