# ==========================================================================
# maskrcnn_palm/maskrcnn_spatialmamba_t_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + Spatial-Mamba-Tiny on UAV 5 cm.
#
# Role: Secondary Mamba candidate (structure-aware SSM) under Mask R-CNN.
# Primary scientific value of this backbone sits under SOLOv2; the
# Mask R-CNN variant is included for completeness and as an optional
# entry in the bottleneck ablation.
#
# Spatial-Mamba-T architecture (MM_SpatialMamba backbone):
#   dims       : 64               (narrower than VMamba/MambaVision)
#   depths     : (2, 4, 8, 4)
#   d_state    : 1
#   drop_path  : 0.2
#   norm_layer : 'ln'
# Stage output channels: [64, 128, 256, 512]
#
# Note: Spatial-Mamba adopts narrower channel widths than VMamba by design,
# relying on its dilated-convolution prior for local-global balance rather
# than raw channel capacity. FPN in_channels must reflect this.
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
        pretrained='/workspace/mmdetection/checkpoints/spatialmamba/spatialmamba_tiny_in1k.pth',
        dims=64,
        depths=(2, 4, 8, 4),
        d_state=1,  
        drop_path_rate=0.2,
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

work_dir = './work_dirs/maskrcnn_spatialmamba_t_uav5cm'
