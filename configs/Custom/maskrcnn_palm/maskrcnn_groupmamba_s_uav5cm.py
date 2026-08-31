# ==========================================================================
# maskrcnn_palm/optional/maskrcnn_groupmamba_s_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + GroupMamba-Small on UAV 5 cm.
#
# Status: OPTIONAL.
#
# GroupMamba-S architecture:
#   embed_dims      : (64, 128, 348, 512)   (wider stage 3 vs Tiny's 448)
#   depths          : (3, 4, 16, 3)
#   stem_hidden_dim : 64
#   drop_path       : 0.3
# Stage output channels: [64, 128, 348, 512]
# ==========================================================================


_base_ = [
    './_base_maskrcnn_palm.py',
    '../_base_palm/dataset_uav_5cm.py',
    '../_base_palm/schedule_mamba_120k.py',
    '../_base_palm/runtime_palm.py',
]
custom_imports = dict(
    imports=['mmdet.models.backbones.groupmamba_backbone',
             'configs.Custom._base_palm.benchmark_logging_hook'],
    allow_failed_imports=False,
)

# --- Backbone + neck override ---------------------------------------------
model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_GroupMamba',
        out_indices=(0, 1, 2, 3),
        pretrained='/workspace/mmdetection/checkpoints/groupmamba/groupmamba_small.pth',
        embed_dims=(64, 128, 348, 512),
        depths=(3, 4, 16, 3),
        mlp_ratios=(8, 8, 4, 4),
        stem_hidden_dim=64,
        drop_path_rate=0.3,
        frozen_stages=-1,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 348, 512],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = './work_dirs/maskrcnn_groupmamba_s_uav5cm'
