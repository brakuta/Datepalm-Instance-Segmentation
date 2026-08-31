# ==========================================================================
# 1_single_sensor_uav_5cm/maskrcnn_pvtv2_b2_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + PVT-v2-B2 on UAV 5 cm.
#
# Role: Linear-attention transformer baseline. PVT-v2 uses overlapping
# patch embeddings and linear spatial-reduction attention (SRA), providing
# a mechanistically distinct global-context family for comparison against
# Swin's local-window attention. It is one of the most frequently cited
# transformer baselines in remote sensing instance segmentation literature,
# strengthening the benchmark's external validity.
#
# PVT-v2-B2 architecture:
#   embed_dims    : [64, 128, 320, 512]
#   num_layers    : [3, 4, 6, 3]
#   num_heads     : [1, 2, 5, 8]
#   mlp_ratios    : [8, 8, 4, 4]
#   sr_ratios     : [8, 4, 2, 1]
#   drop_path_rate: 0.1
#   qkv_bias      : True
# Stage output channels: [64, 128, 320, 512]
# Params: ~45M (comparable to Swin-T at ~48M)
#
# Pretraining: ImageNet-1k (NOT 22k) — consistent with all other
# transformer baselines in the benchmark to avoid pretraining confound.
#
# Protocol: 80k iterations, schedule_standard_80k.py, AdamW,
#           frozen_stages=0 (PVT-v2 uses no stem freeze convention;
#           full fine-tuning with reduced backbone LR×0.1 is standard).
# ==========================================================================
_base_ = [
    './_base_maskrcnn_palm.py',
    '../_base_palm/dataset_uav_5cm.py',
    '../_base_palm/schedule_standard_80k.py',
    '../_base_palm/runtime_palm.py',
]

# PVT-v2 is registered in mmdet via mmdet.models.backbones.pvt
custom_imports = dict(
    imports=['mmdet.models.backbones.pvt'],
    allow_failed_imports=False,
)

# --- Backbone + neck override ---------------------------------------------
model = dict(
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/'
                       'pvt/pvt-v2-b2_3rdparty_in1k_20220501-d7f3e26e.pth',
        ),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5,
    ),
)

work_dir = './work_dirs/maskrcnn_pvtv2_b2_uav5cm'
