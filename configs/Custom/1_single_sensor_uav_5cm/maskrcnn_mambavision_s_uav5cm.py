# ==========================================================================
# maskrcnn_palm/maskrcnn_mambavision_s_uav5cm.py
# --------------------------------------------------------------------------
# Mask R-CNN + MambaVision-Small on UAV 5 cm.
#
# Role: Scaling check for MambaVision under Mask R-CNN.
# Comparison targets:
#   - maskrcnn_mambavision_t_uav5cm.py (same family, smaller capacity)
#   - maskrcnn_r50_uav5cm.py (parameter-matched CNN reference)
#
# MambaVision-S stage output channels: [96, 192, 384, 768]
# Drop path rate: 0.2 (standard for Small variant; deeper stochasticity)
# ==========================================================================

_base_ = [
    './_base_maskrcnn_palm.py',
    '../_base_palm/dataset_uav_5cm.py',
    '../_base_palm/schedule_mamba_120k.py',
    '../_base_palm/runtime_palm.py',
]

custom_imports = dict(
    imports=[
        'mmdet.models.backbones.mambavision_backbone',
        'configs.Custom._base_palm.benchmark_logging_hook',
    ],
    allow_failed_imports=False,
)



# --- Backbone + neck override ---------------------------------------------
model = dict(
    backbone=dict(
        _delete_=True,
        type='mamba_small_vision_timm',
        drop_path_rate=0.2,
        # Gradient checkpointing is MANDATORY for MambaVision-S at 1024x1024
        # on 24 GB VRAM. Channel widths 96->768 (vs Tiny 80->640) produce
        # substantially larger activations at stages 3-4.
        gradient_checkpointing=False,
        # frozen_stages=0 for protocol consistency with MambaVision-T.
        # Gradient checkpointing provides sufficient VRAM headroom for
        # full fine-tuning even on the Small variant.
        frozen_stages=0,
        local_files_only=True,   # ← add this line
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],   # MambaVision-S stage dims
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = './work_dirs/maskrcnn_mambavision_s_uav5cm'
