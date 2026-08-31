# ==========================================================================
# maskrcnn_palm/maskrcnn_efficientvmamba_b_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + EfficientVMamba-Base on UAV 5 cm.
#
# EfficientVMamba-B architecture (MM_EFFVSSM backbone):
#   dims                : 96
#   depths              : (2, 2, 9, 2)
#   ssm_d_state         : 16
#   ssm_ratio           : 2.0
#   mlp_ratio           : 0.0       (no MLP after SSM)
#   downsample_version  : 'v1'
#   patchembed_version  : 'v1'
#   window_size         : 2         (atrous skip-sampling -- core contribution)
#   drop_path_rate      : 0.2
# Stage output channels : [96, 192, 384, 768]
# Total params          : ~33M (Mask R-CNN backbone)
#
# Reference upstream config:
#   /opt/efficientvmamba/detection/configs/efficient/
#       mask_rcnn_vssm_fpn_coco_efficient_2292_96.py
# ==========================================================================

_base_ = [
    './_base_maskrcnn_palm.py',
    '../_base_palm/dataset_uav_5cm.py',
    '../_base_palm/schedule_mamba_120k.py',
    '../_base_palm/runtime_palm.py',
]

custom_imports = dict(
    imports=['mmdet.models.backbones.efficientvmamba_backbone',
    'configs.Custom._base_palm.benchmark_logging_hook'],
    allow_failed_imports=False,
)

# --- Backbone + neck override ---------------------------------------------
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

work_dir = './work_dirs/maskrcnn_efficientvmamba_b_uav5cm'
