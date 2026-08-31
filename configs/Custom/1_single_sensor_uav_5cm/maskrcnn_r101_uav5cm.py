# ==========================================================================
# maskrcnn_palm/maskrcnn_r101_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + ResNet-101 on UAV 5 cm.
#
# Role: CNN depth ablation baseline. Directly comparable to ResNet-50 to
# isolate the effect of backbone depth within the CNN family. Both R50 and
# R101 use identical training settings — the only variable is depth.
#
# ResNet-101 architecture vs ResNet-50:
#   Depth        : 101 layers  (vs 50)
#   Block config : [3, 4, 23, 3]  (vs [3, 4, 6, 3])
#   Stage channels: [256, 512, 1024, 2048]  (identical to R50)
#   FPN in_channels: [256, 512, 1024, 2048] (identical — no neck change)
#   Params       : ~63M  (vs ~44M for R50)
#
# Protocol: 80k iterations, schedule_standard_80k.py, AdamW,
#           frozen_stages=1 (stem only — standard ImageNet transfer).
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
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet101',
        ),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
)

work_dir = './work_dirs/maskrcnn_r101_uav5cm'
