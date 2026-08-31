# ==========================================================================
# maskrcnn_palm_ms15/optional/maskrcnn_groupmamba_s_ms15.py
# --------------------------------------------------------------------------
# Mask R-CNN + GroupMamba-Small on the pooled MS-15 cm corpus
# (GE 15 cm + Aerial 15 cm, sensor-balanced sampling at alpha=0.3).
#
# Stage B counterpart of maskrcnn_groupmamba_s_uav5cm.py.
#
# Status: OPTIONAL — supplementary only. Not in the primary Stage B
# benchmark matrix. Run only after the primary cohort is complete and
# disk/compute budget permits.
#
# GroupMamba-S architecture:
#   embed_dims      : (64, 128, 348, 512)   (wider stage 3 vs Tiny's 448)
#   depths          : (3, 4, 16, 3)         (deeper stage 2 vs Tiny's 9)
#   mlp_ratios      : (8, 8, 4, 4)
#   stem_hidden_dim : 64                    (wider stem vs Tiny's 32)
#   drop_path       : 0.3
# Stage output channels: [64, 128, 348, 512]
#
# Architectural notes:
#   - Stage 2 channel width 348 is shared with Tiny (348 = 4 x 87).
#     Stage 3 widens to 512 (vs Tiny's 448), the only channel difference
#     between variants. FPN in_channels must match exactly.
#   - frozen_stages=-1: GroupMamba backbone module's convention for no
#     frozen stages. Preserved verbatim from Stage A.
#   - Pretrained checkpoint is loaded from the same container-local path
#     used in Stage A (offline container, no network access).
#   - OOM risk on Small is higher than Tiny due to deeper stage 2
#     (16 blocks vs 9) and wider stage 3 stem. If OOM occurs, reduce
#     accumulative_counts from 2 to 1 in a per-config optim_wrapper
#     override.
#     Tile size 1024x1024 is a benchmark invariant.
#
# -----------------------------------------------------------------------
# CRITICAL — custom_imports overrides runtime imports entirely:
#   MMEngine replaces (does not merge) the top-level custom_imports key
#   when set in a child config. The full import list from
#   runtime_palm_ms15.py must therefore be reproduced here, with the
#   addition of 'mmdet.models.backbones.groupmamba_backbone'.
#
# CRITICAL — custom_hooks overrides runtime hooks entirely:
#   Same replacement behaviour. The full hook stack from
#   runtime_palm_ms15.py must be reproduced here.
#   compute_flops=True: torch.jit.trace succeeds on GroupMamba's
#   pure-SSM forward graph.
# ==========================================================================

_base_ = [
    '../maskrcnn_palm/_base_maskrcnn_palm_ms15.py',
    '../_base_palm/dataset_MS15_pooled.py',
    '../_base_palm/schedule_unified_MS_80k.py',
    '../_base_palm/runtime_palm_ms15.py',
]

# --- Custom imports -------------------------------------------------------
# Reproduces runtime_palm_ms15.py imports plus the GroupMamba backbone.
# MMEngine replaces custom_imports entirely when set in a child config.
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler',
        'mmdet.models.backbones.groupmamba_backbone',
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

# --- Backbone + neck override (verbatim from Stage A GroupMamba-S) --------
model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_GroupMamba',
        out_indices=(0, 1, 2, 3),
        pretrained='/workspace/mmdetection/checkpoints/groupmamba/groupmamba_small.pth',
        embed_dims=(64, 128, 348, 512),
        depths=(3, 4, 16, 3),
        mlp_ratios=(8, 8, 4, 4),
        stem_hidden_dim=64,
        drop_path_rate=0.3,
        frozen_stages=-1,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 348, 512],   # non-standard widths -- see header
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = './work_dirs/maskrcnn_groupmamba_s_ms15'