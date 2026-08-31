# ==========================================================================
# maskrcnn_palm_ms15/maskrcnn_spatialmamba_s_ms15.py
# --------------------------------------------------------------------------
# Mask R-CNN + Spatial-Mamba-Small on the pooled MS-15 cm corpus
# (GE 15 cm + Aerial 15 cm, sensor-balanced sampling at alpha=0.3).
#
# Stage B counterpart of maskrcnn_spatialmamba_s_uav5cm.py. The backbone
# and neck blocks are inherited verbatim from Stage A to preserve
# cross-stage architectural comparability.
#
# Spatial-Mamba-S architecture:
#   dims       : 64               (SAME as Tiny — Spatial-Mamba convention)
#   depths     : (2, 4, 21, 5)    (deeper stage 3 and 4 vs Tiny)
#   d_state    : 1
#   drop_path  : 0.3
#   norm_layer : 'ln'
# Stage output channels: [64, 128, 256, 512]  (SAME as Tiny — dims=64)
#
# Key architectural note: unlike VMamba and MambaVision, Spatial-Mamba-S
# does NOT widen channels relative to Tiny. Capacity scaling is achieved
# purely through stage-3 depth (21 blocks vs 8) and stage-4 depth
# (5 blocks vs 4). FPN in_channels therefore matches Tiny exactly.
# This is not a config error — it is by architectural design.
#
# Practical implication: OOM risk on Small is higher than Tiny despite
# identical channel widths, because stage-3 activation memory scales
# with depth (21 vs 8 blocks). If OOM occurs, reduce
# accumulative_counts from 2 to 1 via a per-config optim_wrapper
# override. Tile size 1024x1024 is a benchmark invariant.
#
# Mamba-specific design decisions (inherited verbatim from Stage A):
#   - frozen_stages=0 (full fine-tuning). Benchmark invariant across
#     all Mamba backbones in both Stage A and Stage B.
#   - Pretrained checkpoint path is the same container-local path used
#     in Stage A (offline container, no network access).
#
# -----------------------------------------------------------------------
# CRITICAL — custom_imports overrides runtime imports entirely:
#   MMEngine replaces (does not merge) the top-level custom_imports key
#   when set in a child config. The full import list from
#   runtime_palm_ms15.py must therefore be reproduced here, with the
#   addition of 'mmdet.models.backbones.spatialmamba_backbone'.
#
# CRITICAL — custom_hooks overrides runtime hooks entirely:
#   Same replacement behaviour. The full hook stack from
#   runtime_palm_ms15.py must be reproduced here.
#   compute_flops=True: torch.jit.trace succeeds on Spatial-Mamba's
#   pure-SSM forward graph.
# ==========================================================================

_base_ = [
    '../maskrcnn_palm/_base_maskrcnn_palm_ms15.py',
    '../_base_palm/dataset_MS15_pooled.py',
    '../_base_palm/schedule_unified_MS_80k.py',
    '../_base_palm/runtime_palm_ms15.py',
]

# --- Custom imports -------------------------------------------------------
# Reproduces runtime_palm_ms15.py imports plus the SpatialMamba backbone.
# MMEngine replaces custom_imports entirely when set in a child config.
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler',
        'mmdet.models.backbones.spatialmamba_backbone',
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

# --- Backbone + neck override (verbatim from Stage A SpatialMamba-S) ------
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
        in_channels=[64, 128, 256, 512],   # same as Tiny -- not a config error
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = './work_dirs/maskrcnn_spatialmamba_s_ms15'