# ==========================================================================
# 2_pooled_15cm_ge_aerial/maskrcnn_pvtv2_b2_ms15.py
# --------------------------------------------------------------------------
# Mask R-CNN + PVT-v2-B2 on the pooled MS-15 cm corpus
# (GE 15 cm + Aerial 15 cm, sensor-balanced sampling at alpha=0.3).
#
# Stage B counterpart of maskrcnn_pvtv2_b2_uav5cm.py. The backbone and
# neck blocks are inherited verbatim from Stage A to preserve cross-stage
# architectural comparability.
#
# Role within Stage B:
#   Linear-attention transformer baseline. PVT-v2 uses overlapping patch
#   embeddings and linear spatial-reduction attention (SRA), providing a
#   mechanistically distinct global-context family for comparison against
#   Swin's local-window attention and SSM-based backbones. The cross-sensor
#   dimension of Stage B additionally tests whether PVT-v2's global SRA
#   mechanism generalises more or less consistently across GE and Aerial
#   sensor domains relative to local-attention (Swin) and SSM alternatives.
#
# PVT-v2-B2 architecture:
#   embed_dims    : [64, 128, 320, 512]
#   num_layers    : [3, 4, 6, 3]
#   num_heads     : [1, 2, 5, 8]
#   mlp_ratios    : [8, 8, 4, 4]
#   sr_ratios     : [8, 4, 2, 1]
#   drop_path_rate: 0.1
#   qkv_bias      : True
# Stage output channels: [64, 128, 320, 512]
# Params: ~45M (comparable to Swin-T at ~48M)
#
# CRITICAL — pretraining confound (inherited from Stage A):
#   ImageNet-1k pretrained weights only (NOT 22k), consistent with all
#   other transformer baselines in the benchmark.
#
# Checkpoint path: relative to /workspace/mmdetection/ (same as Stage A).
# Verify the file exists before launching:
#   ls checkpoints/pvtv2_b2_backbone_only.pth
#
# NOTE — FPN neck:
#   add_extra_convs and relu_before_extra_convs are absent here,
#   preserved verbatim from Stage A maskrcnn_pvtv2_b2_uav5cm.py.
#   Verify against the Stage A config if this is unexpected.
#
# -----------------------------------------------------------------------
# CRITICAL — custom_imports overrides runtime imports entirely:
#   MMEngine replaces (does not merge) the top-level custom_imports key
#   when set in a child config. The full import list from
#   runtime_palm_ms15.py must therefore be reproduced here, with the
#   addition of 'mmdet.models.backbones.pvt'.
#
# CRITICAL — custom_hooks overrides runtime hooks entirely:
#   Same replacement behaviour. The full hook stack from
#   runtime_palm_ms15.py must be reproduced here.
#   compute_flops=True: torch.jit.trace succeeds on PVT-v2's
#   transformer forward graph.
# ==========================================================================

_base_ = [
    '../1_single_sensor_uav_5cm/_base_maskrcnn_palm_ms15.py',
    '../_base_palm/dataset_MS15_pooled.py',
    '../_base_palm/schedule_unified_MS_80k.py',
    '../_base_palm/runtime_palm_ms15.py',
]

# --- Custom imports -------------------------------------------------------
# Reproduces runtime_palm_ms15.py imports plus the PVT-v2 backbone.
# MMEngine replaces custom_imports entirely when set in a child config.
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler',
        'mmdet.models.backbones.pvt',
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

# --- Backbone + neck override (verbatim from Stage A PVT-v2-B2) -----------
model = dict(
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/pvtv2_b2_backbone_only.pth',
        ),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5,
    ),
)

work_dir = './work_dirs/maskrcnn_pvtv2_b2_ms15'