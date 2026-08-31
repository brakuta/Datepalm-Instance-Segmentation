# ==========================================================================
# 3_unified_multisource/maskrcnn_swin_s_stagec.py   (v1)
# --------------------------------------------------------------------------
# Mask R-CNN + Swin-S on the unified Stage C corpus
# (UAV 5 cm + GE 15 cm + Aerial 15 cm, proportional sampling).
# Target workstation: RTX A5000 (portable; batch 2, no memory fix).
#
# Stage C counterpart of maskrcnn_swin_s_ms15.py. Backbone and neck inherited
# verbatim from Stage A/B (architecture untouched).
#
# WHAT CHANGED RELATIVE TO STAGE B
# --------------------------------------------------------------------------
#  1. _base_ chain -> Stage C bases; schedule_stagec.py left commented
#     (runtime owns precision/optimiser/schedule).
#  2. custom_imports / custom_hooks are now DECLARED EXPLICITLY. This differs
#     from the Stage B Swin-S config, which declared neither and inherited
#     both from the runtime (SwinTransformer is native to MMDet, so no
#     backbone import is needed). In Stage C that inheritance is undesirable:
#     the Stage C runtime hook stack still carries MemProbeHook, so inheriting
#     would give Swin-S a hook stack DIFFERENT from the rest of the cohort
#     (which redeclare and omit it). Both keys are therefore reproduced here
#     to keep the cohort uniform. custom_imports lists the six hook/util
#     modules only — no backbone import, because Swin is native.
#  3. work_dir relocated to native overlay storage (/root/work_dirs/...).
#  4. NO batch fix: per-GPU batch 2, accumulate 2 -> effective batch 4,
#     inherited from the runtime under PRECISION='amp_fp16'.
#  5. compute_flops=True retained: torch.jit.trace succeeds on Swin's graph.
#
# >>> VERIFY BEFORE LAUNCH <<<
#  - Checkpoint path is RELATIVE ('checkpoints/...'), resolved against the
#    launch cwd (the mmdetection root on the A5000). Confirm it exists:
#      ls checkpoints/swin_small_p4_w7_backbone_only.pth
#  - A5000 dataloader knobs: in dataset_UAV_GE_Aerial_pooled_C.py set
#    num_workers=1, pin_memory=False, prefetch_factor=2 (16 GB container).
#  - FPN NECK divergence (carried verbatim from Stage A/B): this neck OMITS
#    add_extra_convs / relu_before_extra_convs, so the 5th level (P6) is a
#    max-pool of the last output. The Mamba-family Stage C configs set
#    add_extra_convs='on_output' (P6 via conv). This cross-backbone neck
#    difference predates Stage C; it is preserved here for within-backbone
#    cross-stage comparability. Confirm it is intended for the reported matrix.
#
# Architecture (verbatim): embed_dims=96, depths=[2,2,18,2] (vs Tiny's
# [2,2,6,2]), drop_path_rate=0.3, ImageNet-1k pretraining only;
# FPN in_channels=[96,192,384,768] (same as Swin-T — not a config error).
# ==========================================================================

_base_ = [
    '../_base_palm/_base_maskrcnn_palm_stagec.py',
    '../_base_palm/dataset_UAV_GE_Aerial_pooled_C.py',
    #'../_base_palm/schedule_stagec.py',
    '../_base_palm/runtime_palm_stagec.py',
]

# Six hook/util modules only -- Swin is native to MMDet (no backbone import).
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler_n',
        'configs.Custom._base_palm.per_sensor_best_checkpoint_hook',
        'configs.Custom._base_palm.mean_sensor_metric_hook',
        'configs.Custom._base_palm.dataloader_worker_init',
        'configs.Custom._base_palm.nms_fp32_guard',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(
        type='MeanSensorMetricHook',
        sensor_keys=['UAV/coco/segm_mAP_50', 'GE/coco/segm_mAP_50'],
        out_key='mean/segm_mAP_50',
    ),
    dict(
        type='PerSensorBestCheckpointHook',
        monitors=dict(UAV='UAV/coco/segm_mAP_50', GE='GE/coco/segm_mAP_50'),
        rule='greater',
    ),
    dict(
        type='EarlyStoppingHook',
        monitor='mean/segm_mAP_50',
        rule='greater',
        min_delta=0.001,
        patience=10,
    ),
    dict(
        type='PalmBenchmarkLoggingHook',
        input_shape=(3, 1024, 1024),
        compute_flops=True,
        save_json=True,
    ),
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/swin_small_p4_w7_backbone_only.pth',
        ),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],   # same as Swin-T -- not a config error
        out_channels=256,
        num_outs=5,
        # NOTE: add_extra_convs / relu_before_extra_convs intentionally absent
        # (verbatim from Stage A/B). P6 via max-pool, not conv. See header.
    ),
)

work_dir = '/root/work_dirs/Stage_C/maskrcnn_swin_s_stagec'
