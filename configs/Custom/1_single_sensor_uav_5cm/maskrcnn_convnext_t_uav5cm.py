# ==========================================================================
# 1_single_sensor_uav_5cm/maskrcnn_convnext_t_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + ConvNeXt-T on UAV 5 cm.
#
# Role: Modernised CNN baseline. ConvNeXt is a pure-CNN architecture
# designed to match transformer performance under equivalent training
# recipes, making it the strongest non-transformer non-Mamba foil in the
# matrix. Its inclusion isolates the SSM-versus-modern-CNN question from
# the legacy-CNN-versus-modern confound that ResNet-50 introduces.
#
# ConvNeXt-T architecture:
#   arch              : 'tiny'  (depths=[3,3,9,3], channels=[96,192,384,768])
#   drop_path_rate    : 0.4     (per MMDet ConvNeXt-T detection recipe)
# Stage output channels: [96, 192, 384, 768]
#
# Note: ConvNeXt uses LayerNorm (not BatchNorm). The '.norm' weight-decay
# exclusion in schedule_standard_80k.py covers this correctly.
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
        type='ConvNeXt',
        arch='tiny',
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/'
                       'v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
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

work_dir = './work_dirs/maskrcnn_convnext_t_uav5cm'
