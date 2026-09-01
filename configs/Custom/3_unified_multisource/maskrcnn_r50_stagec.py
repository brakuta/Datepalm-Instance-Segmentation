# ==========================================================================
# 3_unified_multisource/maskrcnn_r50_stagec.py   (v3)
# --------------------------------------------------------------------------
# Mask R-CNN + ResNet-50 on the unified Stage C corpus
# (UAV 5 cm + GE 15 cm + Aerial 15 cm, proportional sampling).
#
# WHAT CHANGED RELATIVE TO v2
# --------------------------------------------------------------------------
#  - No body change. Backbone and neck are inherited verbatim from Stage B.
#  - This config does NOT redeclare custom_hooks, so it inherits the runtime
#    hook stack. Under the v2 runtime that stack was UNBUILDABLE because the
#    runtime passed an invalid save_last kwarg to PerSensorBestCheckpointHook
#    (TypeError at runner construction). The revised runtime removes that
#    kwarg, so this config now builds and runs correctly with no per-file edit.
#  - Optimizer / precision / schedule are inherited from runtime_palm_stagec.py
#    (unified AdamW + Stage B paramwise + PRECISION switch). v2's runtime used
#    SGD lr=0.02; the revised runtime matches Stage B (AdamW lr=1e-4).
# ==========================================================================

_base_ = [
    '../_base_palm/_base_maskrcnn_palm_stagec.py',
    '../_base_palm/dataset_UAV_GE_Aerial_pooled_C.py',
    '../_base_palm/runtime_palm_stagec.py',
]

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

work_dir = './work_dirs/Stage_C/maskrcnn_r50_stagec'
