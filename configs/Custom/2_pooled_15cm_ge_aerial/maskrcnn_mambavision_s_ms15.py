# ==========================================================================
# maskrcnn_palm_ms15/maskrcnn_mambavision_s_ms15.py
# --------------------------------------------------------------------------
# Mask R-CNN + MambaVision-S on the pooled MS-15 cm corpus
# (GE 15 cm + Aerial 15 cm training, GE-only validation;
# Aerial held out for post-hoc evaluation via eval_aerial_ms15.py).
# SensorBalancedSampler at alpha=0.3.
#
# -----------------------------------------------------------------------
# SCHEDULE: schedule_unified_MS_80k.py (80k iterations, identical
# across all Stage B backbones). Stage A best-iteration analysis
# across 16 backbones supports this unified schedule; no backbone-
# family-specific schedule is used.
#
# -----------------------------------------------------------------------
# CRITICAL — custom_hooks overrides runtime hooks entirely:
#   MMEngine replaces (does not merge) the top-level custom_hooks key
#   when set in a child config. The full hook stack from
#   runtime_palm_ms15.py must therefore be reproduced here, with one
#   MambaVision-S-specific modification:
#     PalmBenchmarkLoggingHook: compute_flops=False
#     torch.jit.trace fails on MambaVision's hybrid SSM+attention
#     forward graph. FLOPs are taken from the published reference
#     (Hatamizadeh and Kautz, 2024) scaled to 1024x1024.
#
# CRITICAL — custom_imports overrides runtime imports entirely:
#   Same replacement behaviour. The full import list from
#   runtime_palm_ms15.py must be reproduced here, with the addition
#   of 'mmdet.models.backbones.mambavision_backbone'.
#
# CRITICAL — gradient_checkpointing=False (LOCKED):
#   AMP + gradient checkpointing is incompatible with MambaVision's
#   hybrid SSM+attention stages (dtype divergence float32/float16 in
#   recomputation pass). Do NOT change to True.
#
# -----------------------------------------------------------------------
# CRITICAL — TITAN RTX MEMORY MANAGEMENT (MambaVision-S only):
#
# MambaVision-S under AMP on the TITAN RTX (24 GB VRAM) exhibits three
# coupled memory pathways. Together, these reduce steady-state peak
# memory from ~20 GB to ~14-15 GB, restoring ~9 GB of headroom:
#
#   (i)   RPN target-assignment activation peak — addressed by
#         train_cfg.rpn.sampler.num = 128 (was 256).
#   (ii)  ROI-head activation memory scales linearly with the number
#         of proposals — addressed by rpn_proposal.max_per_img = 512
#         (was 1000) and rpn_proposal.nms_pre = 500 (was 1000).
#   (iii) cuDNN workspace re-allocation triggered by variable ROI-head
#         input shapes — addressed by env_cfg.cudnn_benchmark = False.
#
# Companion environment variable (set in shell, not config):
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#
# Launch:
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   python tools/train.py \
#       configs/Custom/maskrcnn_palm_ms15/maskrcnn_mambavision_s_ms15.py
#
# All three RPN-side reductions, the cudnn_benchmark override, and the
# allocator configuration must be applied together. These collectively
# constitute the "MambaVision-S memory accommodation" reported in the
# manuscript.
#
# Throughput impact: cudnn_benchmark=False costs approximately 5-10%
# training throughput.
# -----------------------------------------------------------------------
#
# Workstation: TITAN RTX (24 GB VRAM, 64 GB system RAM).
# ==========================================================================

_base_ = [
    '../1_single_sensor_uav_5cm/_base_maskrcnn_palm_ms15.py',
    '../_base_palm/dataset_MS15_pooled.py',
    '../_base_palm/schedule_unified_MS_80k.py',
    '../_base_palm/runtime_palm_ms15.py',
]

# --- Custom imports -------------------------------------------------------
# Reproduces runtime_palm_ms15.py imports plus the MambaVision backbone.
# MMEngine replaces custom_imports entirely when set in a child config.
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler',
        'mmdet.models.backbones.mambavision_backbone',
    ],
    allow_failed_imports=False,
)

# --- Custom hooks ---------------------------------------------------------
# Reproduces the full hook stack from runtime_palm_ms15.py with one
# MambaVision-S-specific modification: compute_flops=False (see header).
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
        compute_flops=False,
        save_json=True,
    ),
]

# --- Environment override -------------------------------------------------
# Disables cuDNN workspace re-benchmarking for variable input shapes.
# See header section (iii) for justification. Other env_cfg fields
# match runtime_palm_ms15.py.
env_cfg = dict(
    cudnn_benchmark=False,
    mp_start_method='spawn',
    dist_cfg=dict(backend='nccl'),
)

# --- Model overrides ------------------------------------------------------
model = dict(
    backbone=dict(
        _delete_=True,
        type='mamba_small_vision_timm',
        drop_path_rate=0.2,
        frozen_stages=0,
        gradient_checkpointing=False,
        local_files_only=True,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
    # Memory accommodations for MambaVision-S only (see header).
    train_cfg=dict(
        rpn=dict(
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
        ),
        rpn_proposal=dict(
            nms_pre=500,
            max_per_img=512,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
    ),
)

work_dir = './work_dirs/maskrcnn_mambavision_s_ms15'