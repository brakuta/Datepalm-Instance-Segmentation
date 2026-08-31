# ==========================================================================
# maskrcnn_palm/maskrcnn_r50_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + ResNet-50 on UAV 5 cm — bottleneck ablation baseline.
#
# Role: Reference baseline for the Mask R-CNN head. Compared against
# VMamba-T and MambaVision-S under Mask R-CNN to establish the RoIAlign
# bottleneck effect quantitatively.
#
# Protocol: 80k iterations, standard schedule, AdamW + AMP, full
# fine-tuning (frozen_stages=1 for ResNet's stem, which is the standard
# ImageNet transfer convention and does not undertrain the backbone).
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
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50',
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

work_dir = './work_dirs/maskrcnn_r50_uav5cm'
