# ==========================================================================
# 3_unified_multisource/maskrcnn_mambaout_s_stagec.py   (v1)
# --------------------------------------------------------------------------
# Mask R-CNN + MambaOut-Small on the unified Stage C corpus
# (UAV 5 cm + GE 15 cm + Aerial 15 cm, proportional sampling).
#
# Stage C counterpart of maskrcnn_mambaout_s_ms15.py. Backbone and neck
# inherited verbatim from Stage B (architecture untouched).
#
# WHAT CHANGED RELATIVE TO STAGE B
# --------------------------------------------------------------------------
#  1. _base_ chain -> Stage C bases; schedule_stagec.py left commented
#     (runtime owns precision/optimiser/schedule).
#  2. custom_imports / custom_hooks rebuilt to the full Stage C stack
#     (MMEngine replaces these list keys). sensor_balanced_sampler ->
#     sensor_balanced_sampler_n.
#  3. work_dir relocated to native overlay storage (/root/work_dirs/...).
#  4. NO batch fix. MambaOut is a gated-CNN (the SSM-ablation arm; no SSM
#     recurrence, no hybrid attention), so it runs at the light-backbone
#     condition: per-GPU batch 2, accumulate 2 -> effective batch 4, inherited
#     from the runtime under PRECISION='amp_fp16'. Portable to either machine.
#  5. compute_flops=True retained: torch.jit.trace succeeds on the gated-CNN
#     forward graph.
#
# Architecture (verbatim): variant='small' (timm/mambaout_small.in1k),
# stage channels [96,192,384,576] (SAME as Tiny — capacity scales via depth);
# pretrained via timm (cache populated in Stage A, no network access needed).
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
        'mmdet.models.backbones.mambaout_backbone',
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
        type='MambaOutBackbone',
        variant='small',
        pretrained=True,
        init_cfg=None,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 576],   # same as Tiny -- not a config error
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = '/root/work_dirs/Stage_C/maskrcnn_mambaout_s_stagec'
