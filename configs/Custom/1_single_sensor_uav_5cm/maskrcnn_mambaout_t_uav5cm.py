# ==========================================================================
# maskrcnn_palm/maskrcnn_mambaout_t_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + MambaOut-Tiny on UAV 5 cm.
#
# Role: CRITICAL ABLATION ANCHOR. Paired with maskrcnn_vmamba_t_uav5cm.py
# to isolate the contribution of the selective-scan SSM in VMamba. If
# MambaOut-T matches or exceeds VMamba-T under an identical training
# protocol, the SSM mechanism itself — rather than the surrounding
# gated-convolution block design — is falsified for this task.
#
# Backbone:
#   class        : MambaOutBackbone
#   variant      : 'tiny'  ->  timm/mambaout_tiny.in1k  (HF)
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
        variant='tiny',
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

work_dir = './work_dirs/maskrcnn_mambaout_t_uav5cm'
