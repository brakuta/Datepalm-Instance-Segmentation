# ==========================================================================
# maskrcnn_palm_ms15/maskrcnn_mambaout_s_ms15.py
# --------------------------------------------------------------------------
# Mask R-CNN + MambaOut-Small on the pooled MS-15 cm corpus
# (GE 15 cm + Aerial 15 cm, sensor-balanced sampling at alpha=0.3).
#
# Stage B counterpart of maskrcnn_mambaout_s_uav5cm.py. The backbone and
# neck blocks are inherited verbatim from Stage A to preserve cross-stage
# architectural comparability.
#
# Role within Stage B:
#   Scaling check for the MambaOut ablation. Paired with
#   maskrcnn_vmamba_s_ms15.py to verify that the SSM ablation finding
#   from the Tiny pair holds at the Small capacity band. The cross-sensor
#   dimension of Stage B additionally tests whether any SSM advantage
#   or disadvantage is consistent across GE and Aerial sources.
#
# Backbone:
#   class        : MambaOutBackbone
#   variant      : 'small'  ->  timm/mambaout_small.in1k
#   stage chans  : [96, 192, 384, 576]  (same as Tiny)
#   stage strides: [4, 8, 16, 32]  -> FPN P2..P5
#
# Note: MambaOut-T and MambaOut-S share identical stage output channels
# [96, 192, 384, 576]. This is not a config error — it mirrors the
# MambaOut architecture convention where capacity scaling is achieved
# through depth rather than channel width.
#
# Pretrained weights: loaded via timm at runtime (pretrained=True).
# The timm cache was populated during Stage A training and is available
# in the container without network access.
#
# -----------------------------------------------------------------------
# CRITICAL — custom_imports overrides runtime imports entirely:
#   MMEngine replaces (does not merge) the top-level custom_imports key
#   when set in a child config. The full import list from
#   runtime_palm_ms15.py must therefore be reproduced here, with the
#   addition of 'mmdet.models.backbones.mambaout_backbone'.
#
# CRITICAL — custom_hooks overrides runtime hooks entirely:
#   Same replacement behaviour. The full hook stack from
#   runtime_palm_ms15.py must be reproduced here.
#   compute_flops=True: torch.jit.trace succeeds on MambaOut's
#   gated-CNN forward graph (no SSM recurrence, no hybrid attention).
# ==========================================================================

_base_ = [
    '../maskrcnn_palm/_base_maskrcnn_palm_ms15.py',
    '../_base_palm/dataset_MS15_pooled.py',
    '../_base_palm/schedule_unified_MS_80k.py',
    '../_base_palm/runtime_palm_ms15.py',
]

# --- Custom imports -------------------------------------------------------
# Reproduces runtime_palm_ms15.py imports plus the MambaOut backbone.
# MMEngine replaces custom_imports entirely when set in a child config.
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler',
        'mmdet.models.backbones.mambaout_backbone',
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

# --- Backbone + neck override (verbatim from Stage A MambaOut-S) ----------
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
        in_channels=[96, 192, 384, 576],   # same as Tiny -- not a config error
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = './work_dirs/maskrcnn_mambaout_s_ms15'