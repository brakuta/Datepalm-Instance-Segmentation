# ==========================================================================
# maskrcnn_palm_ms15/maskrcnn_convnext_t_ms15.py
# --------------------------------------------------------------------------
# Mask R-CNN + ConvNeXt-T on the pooled MS-15 cm corpus
# (GE 15 cm + Aerial 15 cm, sensor-balanced sampling at alpha=0.3).
#
# Stage B counterpart of maskrcnn_convnext_t_uav5cm.py. The backbone and
# neck blocks are inherited verbatim from Stage A to preserve cross-stage
# architectural comparability.
#
# Role within Stage B:
#   Modernised CNN baseline. ConvNeXt is a pure-CNN architecture designed
#   to match transformer performance under equivalent training recipes,
#   making it the strongest non-transformer non-Mamba foil in the matrix.
#   Its inclusion isolates the SSM-versus-modern-CNN question from the
#   legacy-CNN-versus-modern confound that ResNet introduces. The cross-
#   sensor dimension of Stage B additionally tests whether ConvNeXt's
#   large-kernel depthwise convolutions generalise more consistently
#   across GE and Aerial sensor domains than SSM or attention alternatives.
#
# ConvNeXt-T architecture:
#   arch              : 'tiny'  (depths=[3,3,9,3], channels=[96,192,384,768])
#   drop_path_rate    : 0.4     (per MMDet ConvNeXt-T detection recipe)
#   layer_scale_init_value: 1.0
# Stage output channels: [96, 192, 384, 768]
#
# Note: ConvNeXt uses LayerNorm (not BatchNorm). The '.norm' weight-decay
# exclusion in schedule_unified_MS_80k.py covers this correctly.
#
# Checkpoint path: relative to /workspace/mmdetection/ (same as Stage A).
# Verify the file exists before launching:
#   ls checkpoints/convnext_tiny_backbone_only.pth
#
# NOTE — FPN neck:
#   add_extra_convs and relu_before_extra_convs are absent, preserved
#   verbatim from Stage A maskrcnn_convnext_t_uav5cm.py. Verify against
#   the Stage A config if this is unexpected.
#
# -----------------------------------------------------------------------
# CRITICAL — custom_imports overrides runtime imports entirely:
#   MMEngine replaces (does not merge) the top-level custom_imports key
#   when set in a child config. The full import list from
#   runtime_palm_ms15.py must therefore be reproduced here, with the
#   addition of 'mmpretrain.models.backbones.convnext'.
#
# CRITICAL — custom_hooks overrides runtime hooks entirely:
#   Same replacement behaviour. The full hook stack from
#   runtime_palm_ms15.py must be reproduced here.
#   compute_flops=True: torch.jit.trace succeeds on ConvNeXt's
#   pure-CNN forward graph.
# ==========================================================================

_base_ = [
    '../maskrcnn_palm/_base_maskrcnn_palm_ms15.py',
    '../_base_palm/dataset_MS15_pooled.py',
    '../_base_palm/schedule_unified_MS_80k.py',
    '../_base_palm/runtime_palm_ms15.py',
]

# --- Custom imports -------------------------------------------------------
# Reproduces runtime_palm_ms15.py imports plus the ConvNeXt backbone.
# MMEngine replaces custom_imports entirely when set in a child config.
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler',
        'mmpretrain.models.backbones.convnext',
    ],
    allow_failed_imports=False,
)

# --- Custom hooks ---------------------------------------------------------
# Reproduces the full hook stack from runtime_palm_ms15.py.
# MMEngine replaces custom_hooks entirely when set in a child config.
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/segm_mAP_50',
        rule='greater',
        min_delta=0.001,
        patience=10,
    ),
    dict(
        type='PalmBenchmarkLoggingHook',
        input_shape=(3, 1024, 1024),
        compute_flops=True,
        save_json=True,
    ),
]

# --- Backbone + neck override (verbatim from Stage A ConvNeXt-T) ----------
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/convnext_tiny_backbone_only.pth',
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

work_dir = './work_dirs/maskrcnn_convnext_t_ms15'