# ==========================================================================
# maskrcnn_palm/maskrcnn_vmamba_s_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + VMamba-Small on UAV 5 cm.
#
# Role: Scaling check for VMamba under Mask R-CNN. Optional for the
# primary bottleneck ablation but included for completeness since
# working legacy checkpoints exist.
#
# VMamba-S architecture:
#   dims       : 96
#   depths     : (2, 2, 15, 2)   (deeper stage 3 vs Tiny)
#   ssm_ratio  : 2.0             (higher vs Tiny 1.0)
#   ssm_d_state: 1
#   drop_path  : 0.3             (higher vs Tiny 0.2)
# Stage output channels: [96, 192, 384, 768]
#
# Protocol notes:
#   - frozen_stages=0 (legacy used 2 — a benchmark confound).
#   - VMamba-S stage-3 has 15 blocks (vs Tiny's 8), roughly doubling
#     stage-3 activation memory. If OOM occurs, the first remediation
#     is enabling gradient checkpointing via the backbone module's
#     use_checkpoint flag (requires the module to support it); the
#     second is reverting accumulative_counts from 2 to 1.
# ==========================================================================

_base_ = [
    './_base_maskrcnn_palm.py',
    '../_base_palm/dataset_uav_5cm.py',
    '../_base_palm/schedule_mamba_120k.py',
    '../_base_palm/runtime_palm.py',
]

custom_imports = dict(
    imports=['mmdet.models.backbones.vmamba_backbone',
            'configs.Custom._base_palm.benchmark_logging_hook'],
    allow_failed_imports=False,
)

# --- Backbone + neck override ---------------------------------------------
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

work_dir = './work_dirs/maskrcnn_vmamba_s_uav5cm'
