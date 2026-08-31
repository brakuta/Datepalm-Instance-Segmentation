# ==========================================================================
# maskrcnn_palm/maskrcnn_efficientvmamba_s_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + EfficientVMamba-Small on UAV 5 cm.
#
# IMPORTANT: The released ImageNet checkpoint `efficient_vmamba_small.ckpt`
# was trained with dims=64 (NOT 96 as listed in the upstream
# detection config or the classification YAML). Verified empirically from
# checkpoint shapes:
#   patch_embed.0.weight = [64, 3, 4, 4]   ->  dims=64
#   stage 1 in_proj.weight = [256, 64]
#   stage 2 in_proj.weight = [512, 128]
#   etc. (channel doubling: 64 -> 128 -> 256 -> 512)
#
# EfficientVMamba-S architecture:
#   dims                : 64               (matches released checkpoint)
#   depths              : (2, 2, 4, 2)
#   ssm_d_state         : 16
#   ssm_ratio           : 2.0
#   mlp_ratio           : 0.0
#   downsample_version  : 'v1'
#   patchembed_version  : 'v1'
#   window_size         : 2
#   drop_path_rate      : 0.2
# Stage output channels : [64, 128, 256, 512]
# ==========================================================================
_base_ = [
    './_base_maskrcnn_palm.py',
    '../_base_palm/dataset_uav_5cm.py',
    '../_base_palm/schedule_mamba_120k.py',
    '../_base_palm/runtime_palm.py',
]
custom_imports = dict(
    imports=['mmdet.models.backbones.efficientvmamba_backbone',
    'configs.Custom._base_palm.benchmark_logging_hook'
    ],
    allow_failed_imports=False,
)

# --- Backbone + neck override ---------------------------------------------
model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_EFFVSSM',
        out_indices=(0, 1, 2, 3),
        pretrained='/workspace/mmdetection/checkpoints/efficientvmamba/efficient_vmamba_small.ckpt',
        depths=(2, 2, 4, 2),
        dims=64,                # <-- corrected from 96
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
        in_channels=[64, 128, 256, 512],   # <-- corrected from [96,192,384,768]
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = './work_dirs/maskrcnn_efficientvmamba_s_uav5cm'