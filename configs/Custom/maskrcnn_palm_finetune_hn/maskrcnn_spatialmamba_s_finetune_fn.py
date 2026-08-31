# ==========================================================================
# maskrcnn_spatialmamba_s_finetune_fn.py
# --------------------------------------------------------------------------
# Operational HARD-POSITIVE fine-tune of the deployed Stage C unified model
# (Spatial-Mamba-S, best_GE) to recover FALSE NEGATIVES — date palms the model
# misses in area types the Stage C corpus under-represents (young/small crowns,
# dense overlapping canopy, dust-hazed or shadowed tiles, mixed orchards,
# atypical GE mosaic radiometry).
#
# This is a DEPLOYMENT ADAPTATION, not a change to the benchmark. Stage C
# checkpoints and their reported metrics are untouched. Describe it in the
# manuscript in one paragraph alongside the hard-negative round; keep it out of
# the backbone comparison.
#
# WHY THIS CONFIG IS NOT THE HARD-NEGATIVE ONE
# --------------------------------------------------------------------------
# maskrcnn_spatialmamba_s_finetune_hn.py fixes FALSE POSITIVES with empty-
# annotation tiles: every RoI on such a tile is an unmatched negative, so the
# classifier learns "shrub is not palm". That mechanism CANNOT fix false
# negatives — an unlabelled palm is trained as background, which makes recall
# strictly worse. Recall needs LABELLED POSITIVES drawn from the failing
# regime, i.e. the AOI tiles produced by Finetune_HN/make_aoi_tiles.py and
# annotated in LabelMe.
#
# Two further consequences, both reflected below:
#   1. NO backbone freezing. FP suppression is a classifier problem, so the HN
#      config offers frozen_stages=4. Recall in a new appearance regime is a
#      FEATURE problem — the crowns look different, not just the decision
#      boundary — so the backbone must be allowed to adapt (lr_mult 0.1).
#   2. The ship gate is the MEAN of GE-val and new-AOI-val mAP@50, not GE
#      alone. Selecting on the new domain only would let the model trade away
#      GE performance; selecting on GE only would reject the very improvement
#      being sought.
#
# ANNOTATION COMPLETENESS IS THE ONE HARD REQUIREMENT
# --------------------------------------------------------------------------
# Within every tile in HP_ROOT, EVERY palm must be labelled. A partially
# annotated tile teaches the model to suppress the palms you left out, which
# is exactly the failure you are trying to remove. If a tile is too dense or
# ambiguous to label exhaustively, DELETE the tile — do not ship it partial.
# (Tiles that genuinely contain zero palms are fine and useful: they are kept
# via filter_empty_gt=False and act as in-domain negatives.)
#
# RADIOMETRY AND CODEC CONSISTENCY
# --------------------------------------------------------------------------
# make_aoi_tiles.py defaults to --stretch none (raw uint8 passthrough), which
# matches GE_train and the GE inference pipeline. Do not stretch GE tiles.
#
# GE_train is also JPEG (COCO/GE_15cm/train_GE/JPEGImages, 19,472 tiles). New
# tiles written as lossless GeoTIFF would let the classifier separate the two
# sources by compression artefact alone, so cut them with
#   make_aoi_tiles.py --format jpg --jpeg-ref <a train_GE .jpg>
# which copies the corpus quantisation tables verbatim. This matters more here
# than for the negatives: these tiles carry the POSITIVE examples whose
# appearance the model must generalise from.
#
# WHAT YOU MUST SET (three paths + two knobs below):
#   GE_ROOT     : the real GE 15 cm COCO root (positives to replay)
#   HP_ROOT     : --out dir from make_aoi_tiles.py, after labelme2coco
#   load_from   : the checkpoint to adapt (see note on chaining HN -> FN)
#   HP_WEIGHT   : over-sampling multiplier for the new tiles (see formula)
#   max_iters   : short — this is adaptation, not training (start 6000)
# ==========================================================================

# ONE base only. maskrcnn_spatialmamba_s_stagec.py ALREADY inherits
# _base_maskrcnn_palm_stagec.py, so naming the detector base here as well made
# two sibling bases define `model` and MMEngine refused the config:
#     KeyError: Duplicate key is not allowed among bases. Duplicate keys: {'model'}
_base_ = ['../maskrcnn_palm_stagec/maskrcnn_spatialmamba_s_stagec.py']

# --------------------------------------------------------------------------
# EDIT THESE
# --------------------------------------------------------------------------
GE_ROOT = '/workspace/datasets/COCO/GE_15cm/'
HP_ROOT = '/workspace/datasets/COCO/HardPos_GE/'

# Start from the ORIGINAL Stage C best_GE for a standalone recall round, or
# from the hard-negative-adapted checkpoint to CHAIN both fixes (recommended
# once the FP round is validated — chaining preserves the FP gains):
load_from = ('work_dirs/Stage_C/maskrcnn_spatialmamba_s_stagec/'
             'best_GE_segm_mAP_50_iter_75001.pth')
# load_from = ('work_dirs/Finetune_HN/maskrcnn_spatialmamba_s_finetune_hn/'
#              'best_coco_segm_mAP_50_iter_XXXX.pth')

# ---- HP_WEIGHT: over-sampling the new tiles -------------------------------
# SensorBalancedSamplerN weights are MULTIPLIERS ON NATURAL SIZE, not shares:
#       quota_s = w_s * N_s / sum_t(w_t * N_t) * sum_t(N_t)
# So with weights [1, 1] a 250-tile HP set against a 2500-tile GE set is only
# ~9% of batches — far too dilute to move recall. To make the new tiles a
# target SHARE p of the batches:
#       HP_WEIGHT = (p / (1 - p)) * (N_GE / N_HP)
# MEASURED N_GE = 19472 (train_GE). So for p = 0.35 and N_HP = 400:
#       (0.35/0.65) * (19472/400) = 0.538 * 48.7 = 26.2
# The multiplier is large precisely because the replay corpus is large -- do
# not mistake it for an error. Read N_HP off your train_hardpos.json
# (len(images)) and recompute; the value below assumes 400 annotated tiles.
# Guardrail: p above ~0.5 risks over-fitting a few hundred tiles; 0.3-0.4 is
# the sane band. Print the sampler's per-source quota lines at startup and
# confirm the realised share before letting the run go long.
HP_WEIGHT = 26.2

# max_iters must let each new tile be seen SEVERAL times -- unlike the
# hard-negative round, these tiles carry positive supervision and a few
# hundred of them is a small signal:
#       max_iters ~= k * N_HP / p     with k = 8-15 passes
# For N_HP = 400, p = 0.35, k = 10  ->  ~11k. Keep it under ~15k: this is
# adaptation, and GE val is the only thing standing between you and drift.
max_iters = 11000
val_interval = 1000

work_dir = 'work_dirs/Finetune_HN/maskrcnn_spatialmamba_s_finetune_fn'

# --------------------------------------------------------------------------
# The backbone config replaces custom_imports / custom_hooks wholesale
# (MMEngine list-key replacement), so reproduce exactly the modules this
# fine-tune needs: the N-source sampler, the fp32 NMS guard, the backbone,
# and the mean-metric hook used as the ship gate.
# --------------------------------------------------------------------------
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.sensor_balanced_sampler_n',
        'configs.Custom._base_palm.mean_sensor_metric_hook',
        'configs.Custom._base_palm.per_sensor_best_checkpoint_hook',
        'configs.Custom._base_palm.nms_fp32_guard',
        'mmdet.models.backbones.spatialmamba_backbone',
    ],
    allow_failed_imports=False,
)

default_scope = 'mmdet'
metainfo = dict(classes=('DatePalm',), palette=[(220, 20, 60)])
backend_args = None

# --------------------------------------------------------------------------
# Pipelines — identical to Stage C, so the adapted model sees the same input
# geometry/radiometry it will meet at inference. Deliberately NOT adding new
# augmentation: the point is to add real examples of the failing regime, and
# a changed augmentation policy would confound the before/after comparison.
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
# Source 0: GE positives (replay — prevents drift/forgetting).
# Source 1: new AOI hard positives. filter_empty_gt=False so exhaustively
#           checked palm-free tiles inside the AOIs are retained as in-domain
#           negatives instead of being silently dropped.
# --------------------------------------------------------------------------
train_dataset_ge_pos = dict(
    type='CocoDataset', metainfo=metainfo, data_root=GE_ROOT,
    ann_file='Annotations/train_GE.json', data_prefix=dict(img='train_GE/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline, serialize_data=True, backend_args=backend_args,
)
train_dataset_hardpos = dict(
    type='CocoDataset', metainfo=metainfo, data_root=HP_ROOT,
    ann_file='annotations/train_hardpos.json',
    data_prefix=dict(img='images_train/'),
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
        weights=[1.0, HP_WEIGHT],   # GE replay : new AOI positives
        chunk_size=4,               # == effective batch (accumulate 4)
    ),
    batch_sampler=None,             # AspectRatioBatchSampler would cross-mix
    dataset=dict(
        type='ConcatDataset',
        datasets=[train_dataset_ge_pos, train_dataset_hardpos],
        ignore_keys=['classes', 'palette'],
    ),
)

# --------------------------------------------------------------------------
# Validation on TWO sources:
#   GE   — the untouched GE val split: the drift/forgetting monitor.
#   HPOS — held-out WHOLE AOIs from make_aoi_tiles.py --val-frac: the only
#          honest measure of whether the recall problem actually improved.
#          (Held out by AOI, not by tile, so overlapping neighbours cannot
#          leak and inflate the gain.)
# --------------------------------------------------------------------------
val_dataset_ge = dict(
    type='CocoDataset', metainfo=metainfo, data_root=GE_ROOT,
    ann_file='Annotations/val_GE.json', data_prefix=dict(img='val_GE/'),
    test_mode=True, pipeline=test_pipeline, serialize_data=True,
    backend_args=backend_args,
)
val_dataset_hardpos = dict(
    type='CocoDataset', metainfo=metainfo, data_root=HP_ROOT,
    ann_file='annotations/val_hardpos.json',
    data_prefix=dict(img='images_val/'),
    test_mode=True, pipeline=test_pipeline, serialize_data=True,
    backend_args=backend_args,
)

# _delete_=True on both dataset blocks: MMEngine merges dicts key-by-key
# without checking `type`, and this config is also assigned to test_dataloader,
# whose inherited dataset is a single CocoDataset. Without it, that
# CocoDataset's ann_file / data_prefix / data_root keys survive into this
# ConcatDataset and the build fails on unexpected kwargs.
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=False,
    pin_memory=True, pin_memory_device='cuda', prefetch_factor=2,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[val_dataset_ge, val_dataset_hardpos],
        ignore_keys=['classes', 'palette'],
    ),
)

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(type='CocoMetric',
             ann_file=GE_ROOT + 'Annotations/val_GE.json',
             metric=['bbox', 'segm'], format_only=False,
             proposal_nums=(100, 300, 1000), backend_args=backend_args),
        dict(type='CocoMetric',
             ann_file=HP_ROOT + 'annotations/val_hardpos.json',
             metric=['bbox', 'segm'], format_only=False,
             proposal_nums=(100, 300, 1000), backend_args=backend_args),
    ],
    dataset_prefixes=['GE', 'HPOS'],
)
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# --------------------------------------------------------------------------
# Schedule — SHORT and LOW-LR: adaptation, not training. Same 1e-5 / cosine
# recipe as the HN round so the two adaptations are comparable, effective
# batch 4 (accumulate 4) matching Stage C.
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
        # 0.1 (not frozen): new-regime recall needs feature adaptation.
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
# Hooks — the ship gate is best-on-MEAN(GE, HPOS) segm mAP@50, so a recall
# gain bought by degrading GE cannot be selected.
# --------------------------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False,
                    interval=val_interval, max_keep_ckpts=3,
                    save_best='mean/segm_mAP_50', rule='greater',
                    save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

custom_hooks = [
    # Runs at HIGHEST priority, injecting 'mean/segm_mAP_50' into the metrics
    # dict before CheckpointHook reads it for save_best (same wiring Stage C
    # uses to feed EarlyStoppingHook).
    dict(type='MeanSensorMetricHook',
         sensor_keys=['GE/coco/segm_mAP_50', 'HPOS/coco/segm_mAP_50'],
         out_key='mean/segm_mAP_50',
         rule='greater'),
    # Also keep the two per-source bests, so the trade-off is inspectable:
    # best_GE = least forgetting, best_HPOS = most recall recovered. Compare
    # all three before deciding what to deploy.
    dict(type='PerSensorBestCheckpointHook',
         monitors=dict(GE='GE/coco/segm_mAP_50',
                       HPOS='HPOS/coco/segm_mAP_50'),
         rule='greater'),
]

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=1),
    dist_cfg=dict(backend='nccl'),
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
log_level = 'INFO'
resume = False
