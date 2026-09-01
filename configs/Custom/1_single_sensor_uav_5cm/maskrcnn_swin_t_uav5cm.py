# ==========================================================================
# 1_single_sensor_uav_5cm/maskrcnn_swin_t_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + Swin-T on UAV 5 cm.
#
# Role: Local-window transformer baseline. Parameter-matched to VMamba-T
# (~48 M vs ~50 M) and MambaVision-T, providing the primary Mamba-versus-
# transformer comparison axis at the Tiny capacity band.
#
# Swin-T architecture:
#   embed_dims     : 96
#   depths         : [2, 2, 6, 2]
#   num_heads      : [3, 6, 12, 24]
#   window_size    : 7
#   drop_path_rate : 0.2
# Stage output channels: [96, 192, 384, 768]
#
# CRITICAL: This config uses ImageNet-1k pretrained weights, NOT the
# ImageNet-22k variant. 22k weights are available and produce ~1-2 mAP
# higher performance but would introduce a pretraining confound in the
# backbone comparison. 22k variants may be reported in a supplementary
# table but must not drive the headline results.
# ==========================================================================

_base_ = [
    './_base_maskrcnn_palm.py',
    '../_base_palm/dataset_uav_5cm.py',
    '../_base_palm/schedule_standard_80k.py',
    '../_base_palm/runtime_palm.py',
]

# --- Backbone + neck override ---------------------------------------------
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/'
                       'download/v1.0.0/swin_tiny_patch4_window7_224.pth',
        ),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
    ),
)

work_dir = './work_dirs/maskrcnn_swin_t_uav5cm'
