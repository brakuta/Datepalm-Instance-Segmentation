# ==========================================================================
# maskrcnn_palm_ms15/maskrcnn_swin_s_ms15.py
# --------------------------------------------------------------------------
# Mask R-CNN + Swin-S on the pooled MS-15 cm corpus
# (GE 15 cm + Aerial 15 cm, sensor-balanced sampling at alpha=0.3).
#
# Stage B counterpart of maskrcnn_swin_s_uav5cm.py. The backbone and
# neck blocks are inherited verbatim from Stage A to preserve cross-stage
# architectural comparability.
#
# Role within Stage B:
#   Base-tier transformer baseline. Parameter-matched to VMamba-S to
#   extend the Mamba-versus-transformer comparison to the base capacity
#   band on the pooled MS-15 cm corpus.
#
# Swin-S differs from Swin-T only in:
#   depths         : [2, 2, 18, 2]   (vs Tiny's [2, 2, 6, 2])
#   drop_path_rate : 0.3             (vs Tiny's 0.2)
#
# All other architecture parameters (embed_dims, num_heads, window_size)
# are identical to Swin-T. FPN in_channels therefore matches Swin-T
# exactly — this is not a config error.
#
# CRITICAL — pretraining confound (inherited from Stage A):
#   ImageNet-1k pretrained weights only (NOT 22k). See
#   maskrcnn_swin_t_ms15.py header for full rationale.
#
# Checkpoint path: relative to /workspace/mmdetection/ (same as Stage A).
# Verify the file exists before launching:
#   ls checkpoints/swin_small_p4_w7_backbone_only.pth
#
# -----------------------------------------------------------------------
# NOTE — custom_imports and custom_hooks:
#   SwinTransformer is registered natively in mmdet; no per-config
#   custom_imports block is required. Since neither custom_imports nor
#   custom_hooks is defined in this file, MMEngine inherits both keys
#   from runtime_palm_ms15.py without replacement. This is the only
#   backbone in the Stage B cohort where this holds; all other backbones
#   requiring custom_imports must also reproduce custom_hooks in full
#   (see runtime_palm_ms15.py header for explanation).
#
# NOTE — FPN neck:
#   add_extra_convs and relu_before_extra_convs are absent, preserved
#   verbatim from Stage A maskrcnn_swin_s_uav5cm.py. Verify against
#   the Stage A config if this is unexpected.
# ==========================================================================

_base_ = [
    '../maskrcnn_palm/_base_maskrcnn_palm_ms15.py',
    '../_base_palm/dataset_MS15_pooled.py',
    '../_base_palm/schedule_unified_MS_80k.py',
    '../_base_palm/runtime_palm_ms15.py',
]

# --- Backbone + neck override (verbatim from Stage A Swin-S) --------------
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/swin_small_p4_w7_backbone_only.pth',
        ),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],   # same as Swin-T -- not a config error
        out_channels=256,
        num_outs=5,
    ),
)

work_dir = './work_dirs/maskrcnn_swin_s_ms15'