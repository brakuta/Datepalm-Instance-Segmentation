# ==========================================================================
# _base_palm/schedule_unified_MS_80k.py
# --------------------------------------------------------------------------
# Unified Stage B (MS-15 cm) training schedule for all backbones
# (CNN, transformer, and Mamba families). 80,000 iterations.
#
# This file is a proportional reduction of the prior Stage B 150k
# Mamba schedule, preserving all optimizer hyperparameters, AMP
# settings, gradient accumulation, paramwise_cfg, and the Mamba-
# specific zero-decay keys. The Mamba-specific keys (A_log, D,
# dt_proj) are no-ops for non-Mamba backbones — the paramwise key
# matcher finds no matching parameters and applies no override.
#
# -----------------------------------------------------------------------
# RATIONALE — 80k unified schedule:
#
# Stage A (UAV 5 cm) best-iteration analysis across 16 backbones:
#   CNN family       (3 variants): best-iter range  10,000 - 30,000
#   Transformer fam. (3 variants): best-iter range  20,000 - 80,000
#   Mamba family    (10 variants): best-iter range  15,000 - 75,000
#
# All but two backbones (Swin-Small at 80k, EfficientMamba-Small at
# 75k) peaked at or below iter 50,000. Median best iteration across
# all 16 backbones was approximately 25,000.
#
# Scaling by the Stage B corpus growth factor (20,676 vs 16,800
# tiles, 1.24x), the expected peak range shifts to approximately
# 12,000 - 62,000 iterations. An 80,000-iteration schedule:
#   - exceeds the 1.24x-scaled upper bound by ~30%, providing every
#     backbone ample opportunity to reach its peak;
#   - matches the Stage A schedule that CNN and transformer backbones
#     used successfully;
#   - holds schedule length constant across architectures, eliminating
#     a confound in the cross-backbone comparison.
#
# Edge case: EfficientMamba-Small peaked at iter 75,000 in Stage A
# (0.625 of its 120k schedule). If included in Stage B, this backbone
# may not fully converge under 80k. Documented as a known limitation.
#
# -----------------------------------------------------------------------
# EFFECTIVE BATCH SIZE:
#
# train_dataloader.batch_size = 2 (in dataset_MS15_pooled.py)
# optim_wrapper.accumulative_counts = 2
# --> Effective batch size = 4
#
# This matches the prior 150k schedule and the auto_scale_lr
# base_batch_size value of 4.
#
# -----------------------------------------------------------------------
# WARMUP:
#
# LinearLR over the first 1,500 iterations (start_factor=0.001), then
# CosineAnnealingLR from iter 1,500 to 80,000, decaying to eta_min=1e-6.
# Warmup length is held at 1,500 iterations (not proportionally scaled)
# because warmup serves to stabilise the AMP loss scale and gradient
# statistics at the start of training, which is independent of total
# schedule length.
# ==========================================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=80000,
    val_interval=5000,
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    accumulative_counts=2,
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone':                       dict(lr_mult=0.1),
            '.bias':                          dict(decay_mult=0.0),
            '.norm':                          dict(decay_mult=0.0),
            'absolute_pos_embed':             dict(decay_mult=0.0),
            'relative_position_bias_table':   dict(decay_mult=0.0),
            'A_log':                          dict(decay_mult=0.0),
            'D':                              dict(decay_mult=0.0),
            'dt_proj':                        dict(decay_mult=0.0),
        },
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1500,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=78500,
        eta_min=1e-6,
        by_epoch=False,
        begin=1500,
        end=80000,
    ),
]

auto_scale_lr = dict(enable=False, base_batch_size=4)
