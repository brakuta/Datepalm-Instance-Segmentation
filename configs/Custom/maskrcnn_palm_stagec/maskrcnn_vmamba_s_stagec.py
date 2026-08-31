# ==========================================================================
# maskrcnn_palm_stagec/maskrcnn_vmamba_s_stagec.py   (v4)
# --------------------------------------------------------------------------
# Mask R-CNN + VMamba-S on the unified Stage C corpus
# (UAV 5 cm + GE 15 cm + Aerial 15 cm, proportional sampling). TITAN RTX only.
#
# WHAT CHANGED RELATIVE TO v2
# --------------------------------------------------------------------------
#  - No body change. Backbone and neck inherited verbatim from Stage B.
#    custom_imports / custom_hooks reproduced in full (with the VMamba import);
#    the hook stack is already correct (no save_last).
#  - Optimizer / precision / schedule inherited from runtime_palm_stagec.py.
#    This backbone was a primary trigger of the reported host freeze (63.3 GB
#    RAM) under v2's FP32. It is resolved at source: PRECISION='amp_fp16'
#    reproduces the Stage B training condition (AdamW lr=1e-4, effective batch
#    4); PRECISION='fp32' falls back to per-GPU batch 1. The Mamba zero-decay
#    keys (A_log, D, dt_proj) in the runtime paramwise_cfg apply to this
#    backbone.
#
# WHAT CHANGED RELATIVE TO v3
# --------------------------------------------------------------------------
#  - SSM MEMORY FIX added (per-GPU batch 1, accumulate 4). Stage B trained
#    VMamba-S within 24 GB at batch 2, but that condition was UAV-light /
#    multi-scale; in Stage C the pooled GE + Aerial 15 cm tiles drive VMamba-S
#    to ~23.7/24 GB at batch 2 under amp_fp16, after which heavy (high-RoI)
#    iterations spill into system RAM via the WSL2/NVIDIA sysmem fallback and
#    collapse throughput to ~13 s/iter. Dropping per-GPU batch to 1 holds the
#    working set at ~13-17 GB; accumulate 4 keeps the EFFECTIVE batch at 4, so
#    the training condition stays comparable to Stage B and to the light
#    backbones on every axis except per-step micro-batch size. This is the
#    same lever the runtime applies automatically under PRECISION='fp32'; here
#    it is applied unconditionally for VMamba-S because the spill occurs in
#    amp_fp16 mode too. Always launch with
#    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
#  - work_dir relocated to native overlay storage (/root/work_dirs/...) so
#    checkpoints and the 'last_checkpoint' resume pointer are not lost on a
#    9p Windows mount (Root Cause 4.3).
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
        'mmdet.models.backbones.vmamba_backbone',
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
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained='/workspace/mmdetection/checkpoints/vmamba/vmamba_small_imagenet1k.pth',
        dims=96,
        depths=(2, 2, 15, 2),
        ssm_d_state=1,
        ssm_dt_rank='auto',
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type='v05_noz',
        mlp_ratio=4.0,
        downsample_version='v3',
        patchembed_version='v2',
        drop_path_rate=0.3,
        norm_layer='ln2d',
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

work_dir = '/root/work_dirs/Stage_C/maskrcnn_vmamba_s_stagec'

# ==========================================================================
# SSM MEMORY FIX (Stage C, TITAN RTX, 24 GB)  --  VMamba-S only.
# --------------------------------------------------------------------------
# Per-GPU batch 1 + accumulate 4 -> effective batch 4 (Stage B comparable).
# Required because the pooled GE + Aerial 15 cm tiles push VMamba-S to
# ~23.7/24 GB at batch 2 under amp_fp16 and spill into system RAM on heavy
# iterations. chunk_size is widened to 4 so every optimiser step still draws
# same-source tiles (page-cache locality; no cross-source micro-batches inside
# an accumulation window). Both dicts deep-merge over the dataset / runtime
# dataloader and optimiser definitions; type, optimizer, paramwise_cfg,
# clip_grad and dtype are inherited unchanged.
#
# This override is unconditional for VMamba-S: it must hold in PRECISION=
# 'amp_fp16' (the recommended mode), where the runtime's fp32-only override
# does not fire. Launch with:
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ==========================================================================
train_dataloader = dict(
    batch_size=1,
    sampler=dict(chunk_size=4),
)
optim_wrapper = dict(accumulative_counts=4)