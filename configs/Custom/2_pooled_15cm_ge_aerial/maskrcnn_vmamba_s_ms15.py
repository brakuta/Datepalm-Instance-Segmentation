# ==========================================================================
# 2_pooled_15cm_ge_aerial/maskrcnn_vmamba_s_ms15.py
# --------------------------------------------------------------------------
# Mask R-CNN + VMamba-Small on the pooled MS-15 cm corpus
# (GE 15 cm + Aerial 15 cm, sensor-balanced sampling at alpha=0.3).
#
# Stage B counterpart of maskrcnn_vmamba_s_uav5cm.py. The backbone and
# neck blocks are inherited verbatim from Stage A to preserve cross-stage
# architectural comparability.
#
# Role within Stage B:
#   - Scaling check for VMamba within the Mamba cohort.
#   - Comparison targets:
#       - maskrcnn_vmamba_t_ms15.py (same family, smaller capacity)
#       - maskrcnn_mambavision_s_ms15.py (parameter-matched Mamba
#         reference with hybrid SSM+attention stages)
#
# VMamba-S architecture:
#   dims       : 96
#   depths     : (2, 2, 15, 2)   (deeper stage 3 vs Tiny's 8 blocks)
#   ssm_ratio  : 2.0             (higher than Tiny's 1.0)
#   ssm_d_state: 1
#   drop_path  : 0.3             (higher than Tiny's 0.2)
# Stage output channels: [96, 192, 384, 768]
#
# Mamba-specific design decisions (inherited verbatim from Stage A):
#   - frozen_stages=0 (full fine-tuning). Benchmark invariant across
#     all Mamba backbones in both Stage A and Stage B.
#   - VMamba-S stage-3 has 15 blocks vs Tiny's 8, approximately doubling
#     stage-3 activation memory. If OOM occurs, the first remediation
#     is reducing accumulative_counts from 2 to 1 in
#     schedule_unified_MS_80k.py (or a per-config override).
#     Tile size 1024x1024 is a benchmark invariant and must not be
#     changed.
#   - Pretrained checkpoint path is the same container-local path used
#     in Stage A.
#
# -----------------------------------------------------------------------
# CRITICAL — custom_imports overrides runtime imports entirely:
#   MMEngine replaces (does not merge) the top-level custom_imports key
#   when set in a child config. The full import list from
#   runtime_palm_ms15.py must therefore be reproduced here, with the
#   addition of 'mmdet.models.backbones.vmamba_backbone'.
#
# CRITICAL — custom_hooks overrides runtime hooks entirely:
#   Same replacement behaviour. The full hook stack from
#   runtime_palm_ms15.py must be reproduced here.
#   compute_flops=True: torch.jit.trace succeeds on VMamba's pure-SSM
#   forward graph (no hybrid attention stages).
# ==========================================================================

_base_ = [
    '../1_single_sensor_uav_5cm/_base_maskrcnn_palm_ms15.py',
    '../_base_palm/dataset_MS15_pooled.py',
    '../_base_palm/schedule_unified_MS_80k.py',
    '../_base_palm/runtime_palm_ms15.py',
]

# --- Custom imports -------------------------------------------------------
# Reproduces runtime_palm_ms15.py imports plus the VMamba backbone.
# MMEngine replaces custom_imports entirely when set in a child config.
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler',
        'mmdet.models.backbones.vmamba_backbone',
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

# --- Backbone + neck override (verbatim from Stage A VMamba-S) ------------
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

work_dir = './work_dirs/maskrcnn_vmamba_s_ms15'