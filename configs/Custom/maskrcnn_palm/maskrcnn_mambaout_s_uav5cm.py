# ==========================================================================
# maskrcnn_palm/maskrcnn_mambaout_s_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + MambaOut-Small on UAV 5 cm.
#
# Role: Scaling check for the MambaOut ablation. Paired with
# maskrcnn_vmamba_s_uav5cm.py to verify the SSM ablation finding holds
# at the Small capacity band as well as Tiny.
#
# Backbone:
#   class        : MambaOutBackbone
#   variant      : 'small'  ->  timm/mambaout_small.in1k  (HF)
#   stage chans  : [96, 192, 384, 576]
#   stage strides: [4, 8, 16, 32]  -> FPN P2..P5
# ==========================================================================

_base_ = [
    './_base_maskrcnn_palm.py',
    '../_base_palm/dataset_uav_5cm.py',
    '../_base_palm/schedule_mamba_120k.py',
    '../_base_palm/runtime_palm.py',
]

custom_imports = dict(
    imports=['mmdet.models.backbones.mambaout_backbone',
    'configs.Custom._base_palm.benchmark_logging_hook'],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        _delete_=True,
        type='MambaOutBackbone',
        variant='small',
        pretrained=True,
        init_cfg=None,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 576],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = './work_dirs/maskrcnn_mambaout_s_uav5cm'
