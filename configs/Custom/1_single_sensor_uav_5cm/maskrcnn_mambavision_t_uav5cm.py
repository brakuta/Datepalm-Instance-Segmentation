# ==========================================================================
# 1_single_sensor_uav_5cm/maskrcnn_mambavision_t_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + MambaVision-Tiny on UAV 5 cm.
#
# Role: Bottleneck ablation anchor (primary Mamba candidate).
# Comparison target: maskrcnn_r50_uav5cm.py.
# The pair demonstrates whether MambaVision retains its advantage
# when the detection head crops features to a 14x14 RoI grid.
#
# MambaVision-T stage output channels: [80, 160, 320, 640]
# Drop path rate: 0.1 (appropriate for Tiny variant)
#
# --------------------------------------------------------------------------
# REVISION — resolved CheckpointError + AMP incompatibility
# --------------------------------------------------------------------------
# The initial configuration enabled gradient_checkpointing=True. This
# produced a reproducible CheckpointError during the first backward pass:
#
#   torch.utils.checkpoint.CheckpointError: Recomputed values for the
#   following tensors have different metadata than during the forward pass.
#   saved metadata:    {dtype: torch.float32, ...}
#   recomputed metadata: {dtype: torch.float16, ...}
#
# Cause: MambaVision's stage-3 attention blocks force certain tensors to
# float32 for numerical stability under AMP autocast. The gradient-
# checkpointing recomputation pass does not re-enter the autocast context
# with identical state, producing dtype divergence between the saved and
# recomputed activations. This is a known incompatibility between PyTorch's
# legacy (reentrant) gradient checkpointing and AMP in hybrid SSM+attention
# backbones.
#
# Resolution: gradient_checkpointing=False. AMP is retained for its VRAM
# and throughput benefits. At 1024x1024, batch=2, AMP alone, the backbone
# activation footprint fits within 24 GB on a TITAN RTX.
#
# If OOM occurs at this setting, the fallback is to reduce
# accumulative_counts from 2 to 1 in schedule_mamba_120k.py, lowering
# effective batch from 4 to 2. Tile size 1024x1024 is NOT negotiable --
# it is a benchmark invariant (see research plan Section 3.2).
# --------------------------------------------------------------------------

_base_ = [
    './_base_maskrcnn_palm.py',
    '../_base_palm/dataset_uav_5cm.py',
    '../_base_palm/schedule_mamba_120k.py',
    '../_base_palm/runtime_palm.py',
]

custom_imports = dict(
    imports=['mmdet.models.backbones.mambavision_backbone',
              'configs.Custom._base_palm.benchmark_logging_hook'],
    allow_failed_imports=False,
)

# --- Backbone + neck override ---------------------------------------------
model = dict(
    backbone=dict(
        _delete_=True,
        type='mamba_tiny_vision_timm',
        drop_path_rate=0.1,
        # Gradient checkpointing DISABLED -- incompatible with AMP on
        # MambaVision's hybrid SSM+attention stages. See revision notes
        # in the header above.
        gradient_checkpointing=False,
        # frozen_stages=0 -- full fine-tuning. This is a benchmark
        # invariant: all Mamba backbones must be trained with identical
        # freezing policies to prevent undertraining confounds in the
        # architecture comparison.
        frozen_stages=0,
        local_files_only=False,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[80, 160, 320, 640],   # MambaVision-T stage dims
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = './work_dirs/maskrcnn_mambavision_t_uav5cm'