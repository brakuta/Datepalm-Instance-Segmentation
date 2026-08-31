# ==========================================================================
# _base_palm/schedule_stagec.py
# --------------------------------------------------------------------------
# Stage C unified training schedule for all backbones
# (CNN, Transformer, and Mamba families). 120,000 iterations.
#
# This file mirrors the role of schedule_unified_MS_80k.py in Stage B:
# it owns all training-condition decisions — precision, optimizer,
# paramwise_cfg, LR schedule, effective batch size. The runtime
# (runtime_palm_stagec.py) owns hooks, environment, and logging.
# The dataset (dataset_UAV_GE_Aerial_pooled_C.py) owns data loading.
#
# -----------------------------------------------------------------------
# PRECISION SWITCH  --  change ONE line before copying to a workstation.
# -----------------------------------------------------------------------
#  'amp_fp16' : AmpOptimWrapper(float16), per-GPU batch 2, accumulate 2
#               -> effective batch 4. Mirrors Stage B exactly. Fits all
#               backbones including VMamba-S and SpatialMamba-S inside
#               24 GB on the TITAN RTX. RECOMMENDED.
#
#  'fp32'     : OptimWrapper (full FP32), per-GPU batch 1, accumulate 4
#               -> effective batch 4. Use only if an AMP-incompatible
#               custom op prevents amp_fp16.
#
# Both modes use identical optimizer, paramwise_cfg, schedule and effective
# batch (4), so results are mutually comparable and comparable to Stage B
# on every axis except numerical precision.
#
# IMPORTANT: set the same PRECISION on BOTH workstations. Mixed-precision
# runs across machines produce non-comparable results.
# -----------------------------------------------------------------------
PRECISION = 'amp_fp16'      # <-- only line that changes per workstation

assert PRECISION in ('amp_fp16', 'fp32'), f'invalid PRECISION={PRECISION!r}'
_USE_AMP = (PRECISION == 'amp_fp16')
_BATCH   = 2 if _USE_AMP else 1
_ACCUM   = 2 if _USE_AMP else 4   # effective batch = _BATCH * _ACCUM = 4

# ==========================================================================
# Training loops
# ==========================================================================
max_iters    = 120_000
val_interval = 5_000

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=val_interval,
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ==========================================================================
# LR schedule  --  Linear warmup 1.5k + CosineAnnealing (Stage B shape).
# Warmup length is kept at 1500 iterations regardless of total schedule
# length; it stabilises the AMP loss scale at the start of training and is
# independent of the horizon.
# ==========================================================================
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
        T_max=max_iters - 1500,
        eta_min=1e-6,
        by_epoch=False,
        begin=1500,
        end=max_iters,
    ),
]

# ==========================================================================
# Optimizer  --  AdamW + Stage B paramwise_cfg, precision-toggled wrapper.
# --------------------------------------------------------------------------
# Mamba-specific zero-decay keys (A_log, D, dt_proj) are no-ops for
# non-Mamba backbones: the paramwise matcher finds no matching parameters.
# backbone lr_mult=0.1 applies to all backbones (fine-tuning at 10% of the
# head LR), matching the Stage B unified schedule.
# ==========================================================================
optim_wrapper = dict(
    type='AmpOptimWrapper' if _USE_AMP else 'OptimWrapper',
    accumulative_counts=_ACCUM,
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone':                     dict(lr_mult=0.1),
            '.bias':                        dict(decay_mult=0.0),
            '.norm':                        dict(decay_mult=0.0),
            'absolute_pos_embed':           dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'A_log':                        dict(decay_mult=0.0),
            'D':                            dict(decay_mult=0.0),
            'dt_proj':                      dict(decay_mult=0.0),
        },
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
if _USE_AMP:
    optim_wrapper['dtype'] = 'float16'

# Disabled to match Stage B and to avoid implicit LR rescaling.
auto_scale_lr = dict(enable=False, base_batch_size=4)

# ==========================================================================
# DataLoader override for FP32 mode.
# --------------------------------------------------------------------------
# The dataset file defaults to batch_size=2 (AMP). Under fp32 we drop to
# batch_size=1 so the FP32 activation memory fits in 24 GB, and widen the
# sampler chunk to the accumulation window (4) so each optimizer step draws
# same-source tiles -- preserving page-cache locality and ensuring no
# cross-source micro-batches within an accumulation window.
# This is a partial deep-merge override; all other dataloader settings are
# inherited from dataset_UAV_GE_Aerial_pooled_C.py.
# ==========================================================================
if not _USE_AMP:
    train_dataloader = dict(
        batch_size=_BATCH,
        sampler=dict(chunk_size=_ACCUM),
    )
