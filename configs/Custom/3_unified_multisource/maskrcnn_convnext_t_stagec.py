# ==========================================================================
# 3_unified_multisource/maskrcnn_convnext_t_stagec.py   (v1)
# --------------------------------------------------------------------------
# Mask R-CNN + ConvNeXt-T on the unified Stage C corpus
# (UAV 5 cm + GE 15 cm + Aerial 15 cm, proportional sampling).
# Target workstation: RTX A5000 (portable; batch 2, no memory fix).
#
# Stage C counterpart of maskrcnn_convnext_t_ms15.py. Backbone and neck
# inherited verbatim from Stage A/B (architecture untouched).
#
# WHAT CHANGED RELATIVE TO STAGE B
# --------------------------------------------------------------------------
#  1. _base_ chain -> Stage C bases; schedule_stagec.py left commented
#     (runtime owns precision/optimiser/schedule).
#  2. custom_imports / custom_hooks rebuilt to the full Stage C stack
#     (MMEngine replaces these list keys). sensor_balanced_sampler ->
#     sensor_balanced_sampler_n. Backbone import 'mmpretrain.models.
#     backbones.convnext' retained.
#  3. work_dir relocated to native overlay storage (/root/work_dirs/...).
#  4. NO batch fix: per-GPU batch 2, accumulate 2 -> effective batch 4.
#  5. compute_flops=True retained: torch.jit.trace succeeds on the pure-CNN
#     graph. ConvNeXt uses LayerNorm; the '.norm' zero-decay key in the
#     runtime paramwise_cfg covers it.
#
# >>> VERIFY BEFORE LAUNCH <<<
#  - Relative checkpoint path: ls checkpoints/convnext_tiny_backbone_only.pth
#  - A5000 dataloader knobs in dataset_UAV_GE_Aerial_pooled_C.py:
#    num_workers=1, pin_memory=False, prefetch_factor=2.
#  - FPN NECK divergence (verbatim): add_extra_convs / relu_before_extra_convs
#    absent -> P6 via max-pool, unlike the Mamba-family configs (P6 via conv).
#    See the Swin-S Stage C header for the full note. Confirm intended.
#
# Architecture (verbatim): arch='tiny' (depths=[3,3,9,3],
# channels=[96,192,384,768]), drop_path_rate=0.4, layer_scale_init_value=1.0;
# FPN in_channels=[96,192,384,768].
# ==========================================================================

_base_ = [
    '../_base_palm/_base_maskrcnn_palm_stagec.py',
    '../_base_palm/dataset_UAV_GE_Aerial_pooled_C.py',
    #'../_base_palm/schedule_stagec.py',
    '../_base_palm/runtime_palm_stagec.py',
]

custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler_n',
        'configs.Custom._base_palm.per_sensor_best_checkpoint_hook',
        'configs.Custom._base_palm.mean_sensor_metric_hook',
        'configs.Custom._base_palm.dataloader_worker_init',
        'configs.Custom._base_palm.nms_fp32_guard',
        'mmpretrain.models.backbones.convnext',
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
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/convnext_tiny_backbone_only.pth',
        ),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        # NOTE: add_extra_convs / relu_before_extra_convs intentionally absent
        # (verbatim from Stage A/B). P6 via max-pool, not conv. See header.
    ),
)

work_dir = '/root/work_dirs/Stage_C/maskrcnn_convnext_t_stagec'
