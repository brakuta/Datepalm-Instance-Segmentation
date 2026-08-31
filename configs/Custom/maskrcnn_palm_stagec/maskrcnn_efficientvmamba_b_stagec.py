# ==========================================================================
# maskrcnn_palm_stagec/maskrcnn_efficientvmamba_b_stagec.py   (v1)
# --------------------------------------------------------------------------
# Mask R-CNN + EfficientVMamba-Base on the unified Stage C corpus
# (UAV 5 cm + GE 15 cm + Aerial 15 cm, proportional sampling). TITAN RTX only.
#
# Stage C counterpart of maskrcnn_efficientvmamba_b_ms15.py. Backbone and neck
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
#  4. SSM MEMORY FIX (per-GPU batch 1, accumulate 4 -> effective batch 4),
#     applied precautionarily: EfficientVMamba-B was not memory-profiled in
#     Stage C but is a hybrid Mamba-CNN at dims=96 over GE/Aerial 15 cm tiles.
#     Launch with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
#  5. compute_flops=True retained.
#
# CHECKPOINT PATH (corrected)
#   pretrained backbone init:
#     /workspace/mmdetection/checkpoints/efficientvmamba/efficient_vmamba_base.ckpt
#   History: an earlier version used the '/workspace/project/...' prefix,
#   which was the A5000's 9p mount (G:\). EfficientVMamba-B is now TITAN-only
#   and the TITAN mount is '/workspace' (the host workspace volume), so the
#   '/workspace/project' prefix no longer resolves and has been removed. All
#   backbones now share '/workspace/mmdetection/checkpoints/'. Verify with:
#       ls -lh /workspace/mmdetection/checkpoints/efficientvmamba/
#   (One-time read at init; this init is overwritten by the Stage C weights
#   at inference, so a miss only matters when training from ImageNet.)
#
# Architecture (verbatim): dims=96, depths=(2,2,9,2), ssm_d_state=16,
# mlp_ratio=0.0 (no MLP after SSM), window_size=2 (atrous skip-sampling),
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
        'mmdet.models.backbones.efficientvmamba_backbone',
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
        type='MM_EFFVSSM',
        out_indices=(0, 1, 2, 3),
        pretrained='/workspace/mmdetection/checkpoints/efficientvmamba/efficient_vmamba_base.ckpt',
        depths=(2, 2, 9, 2),
        dims=96,
        ssm_d_state=16,
        ssm_dt_rank='auto',
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version='v1',
        patchembed_version='v1',
        window_size=2,
        drop_path_rate=0.2,
        frozen_stages=0,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = '/root/work_dirs/Stage_C/maskrcnn_efficientvmamba_b_stagec'

# ==========================================================================
# SSM MEMORY FIX (Stage C, TITAN RTX, 24 GB)  --  EfficientVMamba-B.
# Per-GPU batch 1 + accumulate 4 -> effective batch 4 (Stage B comparable).
# chunk_size widened to 4 so each optimiser step draws same-source tiles.
# Launch with: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ==========================================================================
train_dataloader = dict(
    batch_size=1,
    sampler=dict(chunk_size=4),
)
optim_wrapper = dict(accumulative_counts=4)
