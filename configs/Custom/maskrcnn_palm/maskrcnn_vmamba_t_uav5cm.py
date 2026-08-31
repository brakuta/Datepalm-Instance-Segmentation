# ==========================================================================
# maskrcnn_palm/maskrcnn_vmamba_t_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + VMamba-Tiny on UAV 5 cm.
#
# Role: Primary bottleneck-ablation Mamba config. Paired with
# maskrcnn_r50_uav5cm.py to establish the RoIAlign bottleneck effect.
# Also paired with solov2_vmamba_t_uav5cm.py for the head-ablation
# comparison.
#
# VMamba-T architecture (MM_VSSM backbone):
#   dims       : 96
#   depths     : (2, 2, 8, 2)
#   ssm_ratio  : 1.0
#   ssm_d_state: 1
#   drop_path  : 0.2
# Stage output channels: [96, 192, 384, 768]
#
# Protocol notes:
#   - frozen_stages=0 for benchmark consistency (legacy used 1)
#   - Gradient checkpointing via backbone flag not yet supported in the
#     vmamba_backbone module used here; VRAM fits without it at batch=2
#     because VMamba-T activations are smaller than MambaVision-T.
#   - Pretrained checkpoint path matches the existing legacy location.
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
        pretrained='/workspace/mmdetection/checkpoints/vmamba/vmamba_tiny_imagenet1k.pth',
        dims=96,
        depths=(2, 2, 8, 2),
        ssm_d_state=1,
        ssm_dt_rank='auto',
        ssm_ratio=1.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type='v05_noz',
        mlp_ratio=4.0,
        downsample_version='v3',
        patchembed_version='v2',
        drop_path_rate=0.2,
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

work_dir = './work_dirs/maskrcnn_vmamba_t_uav5cm'
