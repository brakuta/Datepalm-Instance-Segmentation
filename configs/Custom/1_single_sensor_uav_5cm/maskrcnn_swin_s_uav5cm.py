# ==========================================================================
# 1_single_sensor_uav_5cm/maskrcnn_swin_s_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + Swin-S on UAV 5 cm.
#
# Role: Base-tier transformer baseline. Parameter-matched to VMamba-S
# (~69 M vs ~75 M) to extend the Mamba-versus-transformer comparison
# to the base capacity band. Connects the current benchmark to the prior
# UAV Mask R-CNN result where Swin outperformed ResNet-50.
#
# Swin-S differs from Swin-T only in:
#   depths         : [2, 2, 18, 2]   (vs Tiny's [2, 2, 6, 2])
#   drop_path_rate : 0.3             (vs Tiny's 0.2)
#
# All other architecture parameters (embed_dims, num_heads, window_size)
# are identical to Swin-T. FPN in_channels therefore matches Swin-T exactly.
#
# ImageNet-1k pretraining (NOT 22k) — see maskrcnn_swin_t_uav5cm.py notes.
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
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/'
                       'download/v1.0.0/swin_small_patch4_window7_224.pth',
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

work_dir = './work_dirs/maskrcnn_swin_s_uav5cm'
