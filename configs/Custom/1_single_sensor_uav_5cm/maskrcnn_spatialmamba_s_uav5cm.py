# ==========================================================================
# 1_single_sensor_uav_5cm/maskrcnn_spatialmamba_s_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + Spatial-Mamba-Small on UAV 5 cm.
#
# Role: Scaling variant of Spatial-Mamba-T under Mask R-CNN. Primary
# scientific interest is under SOLOv2; this variant is generated for
# completeness.
#
# Spatial-Mamba-S architecture:
#   dims       : 64               (same as Tiny — Spatial-Mamba convention)
#   depths     : (2, 4, 21, 5)    (deeper stage 3 and 4 vs Tiny)
#   d_state    : 1
#   drop_path  : 0.3
#   norm_layer : 'ln'
# Stage output channels: [64, 128, 256, 512]  (SAME as Tiny — dims=64)
#
# Key architectural note: unlike VMamba and MambaVision, Spatial-Mamba-S
# does NOT widen channels relative to Tiny. Capacity scaling is achieved
# purely through stage-3 depth (21 blocks vs 8). FPN in_channels therefore
# matches Tiny exactly. This is not a config error.
# ==========================================================================

_base_ = [
    './_base_maskrcnn_palm.py',
    '../_base_palm/dataset_uav_5cm.py',
    '../_base_palm/schedule_mamba_120k.py',
    '../_base_palm/runtime_palm.py',
]

custom_imports = dict(
    imports=['mmdet.models.backbones.spatialmamba_backbone',
             'configs.Custom._base_palm.benchmark_logging_hook'],
    allow_failed_imports=False,
)

# --- Backbone + neck override ---------------------------------------------
model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        pretrained='/workspace/mmdetection/checkpoints/spatialmamba/spatialmamba_small_in1k.pth',
        dims=64,
        depths=(2, 4, 21, 5),
        d_state=1,
        drop_path_rate=0.3,
        mlp_ratio=4.0,
        norm_layer='ln',
        frozen_stages=0,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = './work_dirs/maskrcnn_spatialmamba_s_uav5cm'
