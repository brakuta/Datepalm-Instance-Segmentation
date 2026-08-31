# ==========================================================================
# 2_pooled_15cm_ge_aerial/maskrcnn_efficientvmamba_b_ms15.py
# --------------------------------------------------------------------------
# Mask R-CNN + EfficientVMamba-Base on the pooled MS-15 cm corpus
# (GE 15 cm + Aerial 15 cm, sensor-balanced sampling at alpha=0.3).
#
# Stage B counterpart of maskrcnn_efficientvmamba_b_uav5cm.py. The backbone
# and neck blocks are inherited verbatim from Stage A to preserve
# cross-stage architectural comparability.
#
# IMPORTANT — checkpoint path:
#   EfficientVMamba checkpoints are stored at a different path from all
#   other backbones in this benchmark:
#     /workspace/mmdetection/checkpoints/efficientvmamba/
#   (other backbones use /workspace/mmdetection/checkpoints/)
#   Verify this path is accessible inside the container before launching.
#
# EfficientVMamba-B architecture (MM_EFFVSSM backbone):
#   dims                : 96
#   depths              : (2, 2, 9, 2)
#   ssm_d_state         : 16
#   ssm_ratio           : 2.0
#   mlp_ratio           : 0.0              (no MLP after SSM)
#   downsample_version  : 'v1'
#   patchembed_version  : 'v1'
#   window_size         : 2                (atrous skip-sampling)
#   drop_path_rate      : 0.2
# Stage output channels : [96, 192, 384, 768]
#
# Note: Unlike EfficientVMamba-S, the Base variant uses dims=96 and
# [96, 192, 384, 768] FPN in_channels. This matches the upstream
# checkpoint and does not require the dims correction applied to Small.
#
# Mamba-specific design decisions (inherited verbatim from Stage A):
#   - frozen_stages=0 (full fine-tuning). Benchmark invariant across
#     all Mamba backbones in both Stage A and Stage B.
#
# -----------------------------------------------------------------------
# CRITICAL — custom_imports overrides runtime imports entirely:
#   MMEngine replaces (does not merge) the top-level custom_imports key
#   when set in a child config. The full import list from
#   runtime_palm_ms15.py must therefore be reproduced here, with the
#   addition of 'mmdet.models.backbones.efficientvmamba_backbone'.
#
# CRITICAL — custom_hooks overrides runtime hooks entirely:
#   Same replacement behaviour. The full hook stack from
#   runtime_palm_ms15.py must be reproduced here.
#   compute_flops=True: torch.jit.trace succeeds on EfficientVMamba's
#   pure-SSM forward graph.
#
# Reference upstream config:
#   /opt/efficientvmamba/detection/configs/efficient/
#       mask_rcnn_vssm_fpn_coco_efficient_2292_96.py
# ==========================================================================

_base_ = [
    '../1_single_sensor_uav_5cm/_base_maskrcnn_palm_ms15.py',
    '../_base_palm/dataset_MS15_pooled.py',
    '../_base_palm/schedule_unified_MS_80k.py',
    '../_base_palm/runtime_palm_ms15.py',
]

# --- Custom imports -------------------------------------------------------
# Reproduces runtime_palm_ms15.py imports plus the EfficientVMamba backbone.
# MMEngine replaces custom_imports entirely when set in a child config.
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler',
        'mmdet.models.backbones.efficientvmamba_backbone',
    ],
    allow_failed_imports=False,
)

# --- Custom hooks ---------------------------------------------------------
# Reproduces the full hook stack from runtime_palm_ms15.py.
# MMEngine replaces custom_hooks entirely when set in a child config.
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/segm_mAP_50',
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

# --- Backbone + neck override (verbatim from Stage A EfficientVMamba-B) ---
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

work_dir = './work_dirs/maskrcnn_efficientvmamba_b_ms15'