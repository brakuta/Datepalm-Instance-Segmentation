# ==========================================================================
# maskrcnn_palm/optional/maskrcnn_groupmamba_t_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + GroupMamba-Tiny on UAV 5 cm.
#
# Status: OPTIONAL. Not in primary benchmark matrix. Migrated for
# infrastructure consistency only.
#
# GroupMamba-T architecture (MM_GroupMamba backbone):
#   embed_dims      : (64, 128, 348, 448)   (non-standard widths by design)
#   depths          : (3, 4, 9, 3)
#   mlp_ratios      : (8, 8, 4, 4)
#   stem_hidden_dim : 32
#   drop_path       : 0.2
# Stage output channels: [64, 128, 348, 448]
#
# Architectural note: GroupMamba uses 4 channel groups per block, each
# scanning one of 4 directions (vs VMamba's 4 full-channel scans). The
# unusual channel width 348 at stage 2 is a consequence of this grouping
# (348 = 4 x 87, divisible by the 4-way group structure). FPN in_channels
# must match exactly.
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
        pretrained='/workspace/mmdetection/checkpoints/groupmamba/groupmamba_tiny.pth',
        embed_dims=(64, 128, 348, 448),
        depths=(3, 4, 9, 3),
        mlp_ratios=(8, 8, 4, 4),
        stem_hidden_dim=32,
        drop_path_rate=0.2,
        frozen_stages=-1,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 348, 448],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = './work_dirs/maskrcnn_groupmamba_t_uav5cm'
