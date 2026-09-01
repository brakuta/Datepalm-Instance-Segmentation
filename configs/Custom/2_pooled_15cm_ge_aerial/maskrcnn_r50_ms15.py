# ==========================================================================
# 2_pooled_15cm_ge_aerial/maskrcnn_r50_ms15.py
# --------------------------------------------------------------------------
# Mask R-CNN + ResNet-50 on the pooled MS-15 cm corpus
# (GE 15 cm + Aerial 15 cm, sensor-balanced sampling at alpha=0.3).
#
# Stage B counterpart of maskrcnn_r50_uav5cm.py. The ResNet-50 backbone
# and FPN neck are inherited verbatim from Stage A; the detector-level
# differences (anchor scales, training schedule, cross-sensor model
# selection) are encapsulated in the Stage-B-scoped base files below:
#
#   - _base_maskrcnn_palm_ms15.py:    Stage B detector base
#                                       (anchor scales [2, 4, 8])
#   - dataset_MS15_pooled.py:         pooled GE+Aerial, alpha=0.3
#   - schedule_unified_MS_80k.py:   100k iters
#   - runtime_palm_ms15.py:           Stage B runtime
#                                       (mean-sensor save_best,
#                                        EarlyStopping patience=10)
#
# All other detector hyperparameters (RoI/RPN heads, mask_size, NMS
# thresholds, score_thr, max_per_img) are inherited unchanged from
# _base_maskrcnn_palm_ms15.py.
#
# Stage A reproducibility:
#   The Stage A bases (`_base_maskrcnn_palm.py`, `runtime_palm.py`) are
#   preserved untouched. New UAV-5 cm runs continue to use those bases
#   without modification.
# ==========================================================================

_base_ = [
    '../1_single_sensor_uav_5cm/_base_maskrcnn_palm_ms15.py',
    '../_base_palm/dataset_MS15_pooled.py',
    '../_base_palm/schedule_unified_MS_80k.py',
    '../_base_palm/runtime_palm_ms15.py',
]

# --- Backbone + neck override (verbatim from Stage A R50 config) ----------
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

work_dir = './work_dirs/maskrcnn_r50_ms15'
