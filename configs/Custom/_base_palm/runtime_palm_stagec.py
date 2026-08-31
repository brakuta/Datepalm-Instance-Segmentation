# Copyright (c) Stage C Date Palm Benchmark.
# runtime_palm_stagec.py  (v4 - production-grade, comparability-aligned)
# ==========================================================================
# Stage C runtime: 120k iters, val every 5k, EarlyStopping on the mean of the
# two sensor mAP@50 scores (patience 10), per-sensor best checkpoints.
#
# WHAT CHANGED RELATIVE TO v2 (see STAGE_C_REDESIGN.md for the full rationale)
# --------------------------------------------------------------------------
#  1. PRECISION switch (amp_fp16 | fp32). The reported host hang and the
#     ~8.7 s/iter slowdown are caused by FP32 1024^2 exceeding 24 GB on the
#     TITAN RTX, after which the WSL2/NVIDIA driver silently spills VRAM into
#     system RAM (sysmem fallback). Either precision mode below keeps the
#     working set inside 24 GB, so the fallback never triggers.
#  2. Optimizer realigned to Stage B: AdamW lr=1e-4 + the Stage B
#     paramwise_cfg (backbone lr_mult=0.1, zero-decay norm/bias and Mamba
#     A_log/D/dt_proj). v2 used SGD lr=0.02 with NO paramwise_cfg, which is
#     a divergence from Stage B AND an unsuitable optimiser for the SSM /
#     transformer backbones (pretrained SSM weights at SGD 0.02 with full
#     weight decay tend to diverge or underfit).
#  3. accumulative_counts restored so the EFFECTIVE batch size is 4 in both
#     precision modes, matching Stage B (batch 2 x accumulate 2). v2 had no
#     accumulation -> effective batch 2 (another silent Stage B divergence).
#  4. LR schedule shape aligned to Stage B (Linear warmup 1.5k + Cosine to
#     eta_min=1e-6) instead of v2's MultiStepLR. Reversible; see notes.
#  5. auto_scale_lr disabled (enable=False) to match Stage B and to stop v2
#     from scaling AdamW lr by 0.125 (base_batch_size=16, eff batch 2).
#  6. env_cfg.mp_cfg correctly NESTED (v2 was correct; this restates it).
#     mp_start_method='fork' is retained because Stage A/B effectively ran
#     under fork (their top-level 'spawn' key was mis-placed and ignored by
#     MMEngine), and both stages completed all backbones successfully.
#  7. BUG FIX: PerSensorBestCheckpointHook no longer receives save_last=False.
#     The hook __init__ does not accept that kwarg; v2 passed it, which would
#     raise TypeError at runner build for any config that inherits this hook
#     stack without redeclaring it (R50, R101, Swin-S). Removed.
#  8. BUG FIX (Root Cause 4.3): the standard CheckpointHook now sets
#     save_last=True and by_epoch=False. v3 emitted only periodic iter_*.pth
#     and the per-sensor best_*.pth, but NOT the 'last_checkpoint' pointer that
#     MMEngine reads to honour resume=True; interrupted runs were therefore
#     not resumable after a reboot. save_last=True writes that pointer at every
#     interval; by_epoch=False makes interval/save_last count in iterations,
#     matching the IterBasedTrainLoop. work_dir must be on native overlay
#     storage (/root/work_dirs/...) for the pointer write to survive — see the
#     per-backbone configs and the standard launch command.
# ==========================================================================

default_scope = 'mmdet'

# ==========================================================================
# PRECISION / MEMORY MODE  --  single switch.
# --------------------------------------------------------------------------
#  'amp_fp16' : AmpOptimWrapper(float16), batch_size=2, accumulate=2.
#               Mirrors the Stage B training condition
#               (schedule_unified_MS_80k.py: AmpOptimWrapper, dtype=float16,
#                accumulative_counts=2, AdamW lr=1e-4). Stage B trained every
#                Stage C backbone (incl. VMamba-S, SpatialMamba-S,
#                GroupMamba-S, MambaVision-S) on the same 24 GB cards under
#                this condition, so it is demonstrably VRAM-feasible and is
#                the recommended setting.
#  'fp32'     : OptimWrapper (full FP32), batch_size=1, accumulate=4.
#               Retains the SESSION_CONTEXT "FP32 throughout" decision.
#               Per-GPU batch is dropped to 1 so the FP32 working set fits in
#               24 GB; accumulate=4 keeps the effective batch at 4.
#
# In BOTH modes the optimiser, paramwise_cfg, schedule and EFFECTIVE batch
# (=4) are identical, so the two modes remain mutually comparable, and each
# is comparable to Stage B, on every axis except numerical precision.
# ==========================================================================
PRECISION = 'amp_fp16'          # 'amp_fp16' (recommended) | 'fp32'

assert PRECISION in ('amp_fp16', 'fp32'), f'invalid PRECISION={PRECISION!r}'
_USE_AMP = (PRECISION == 'amp_fp16')
_BATCH = 2 if _USE_AMP else 1
_ACCUM = 2 if _USE_AMP else 4   # effective batch = _BATCH * _ACCUM = 4

# ==========================================================================
# Schedule  --  120k iters, val every 5k. Linear warmup + Cosine (Stage B).
# ==========================================================================
max_iters = 120_000
val_interval = 5000

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=val_interval,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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
# The Mamba-specific zero-decay keys (A_log, D, dt_proj) are no-ops for
# non-Mamba backbones: the matcher finds no matching parameters.
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
    clip_grad=dict(max_norm=1.0, norm_type=2),  # Stage B value (v2 used 35)
)
if _USE_AMP:
    optim_wrapper['dtype'] = 'float16'

# Disabled to match Stage B and to avoid re-scaling the AdamW lr.
auto_scale_lr = dict(enable=False, base_batch_size=4)

# ==========================================================================
# DataLoader override driven by PRECISION.
# The dataset file (dataset_UAV_GE_Aerial_pooled_C.py) defaults to
# batch_size=2 (AMP). Under FP32 we drop per-GPU batch to 1 and WIDEN the
# sampler chunk to the accumulation window (4) so that (i) every optimiser
# step still draws same-source tiles -> kernel page-cache locality preserved,
# and (ii) no cross-source micro-batches are introduced. This is a partial
# (deep-merge) override; the rest of the dataloader definition is inherited.
# ==========================================================================
if not _USE_AMP:
    train_dataloader = dict(
        batch_size=_BATCH,
        sampler=dict(chunk_size=_ACCUM),
    )

# ==========================================================================
# Default hooks
# ==========================================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,        # iter-based loop: interval/save_last counted in iters
        interval=val_interval,
        max_keep_ckpts=3,
        save_last=True,        # writes 'last_checkpoint' pointer; required for resume=True
        save_best=None,        # per-sensor best handled by custom hook below
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

# ==========================================================================
# Custom imports / hooks
# ==========================================================================
custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler_n',
        'configs.Custom._base_palm.per_sensor_best_checkpoint_hook',
        'configs.Custom._base_palm.mean_sensor_metric_hook',
        'configs.Custom._base_palm.dataloader_worker_init',
        'configs.Custom._base_palm.nms_fp32_guard',
        'configs.Custom._base_palm.mem_probe_hook',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(
        type='MeanSensorMetricHook',
        sensor_keys=['UAV/coco/segm_mAP_50', 'GE/coco/segm_mAP_50'],
        out_key='mean/segm_mAP_50',
    ),
    dict(
        type='PerSensorBestCheckpointHook',
        monitors=dict(
            UAV='UAV/coco/segm_mAP_50',
            GE='GE/coco/segm_mAP_50',
        ),
        rule='greater',
        # NOTE: no save_last kwarg -- the hook does not accept it (v2 bug).
    ),
    dict(
        type='EarlyStoppingHook',
        monitor='mean/segm_mAP_50',
        rule='greater',
        min_delta=0.001,
        patience=10,
    ),
    dict(
        type='PalmBenchmarkLoggingHook',
        input_shape=(3, 1024, 1024),
        compute_flops=True,        # overridden to False per-backbone (MambaVision-S)
        save_json=True,
    ),
    dict(type='MemProbeHook'),
]

# ==========================================================================
# Environment  --  throughput-critical settings.
# cudnn_benchmark=True: input shape is constant (1024x1024) so the autotuner
# pays off. opencv_num_threads=1 + per-worker clamp prevents cv2 thread
# oversubscription. fork is retained (proven in Stage A/B).
# ==========================================================================
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(
        mp_start_method='fork',
        opencv_num_threads=1,
    ),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
log_level = 'INFO'

randomness = dict(seed=0, deterministic=False)
resume = False
load_from = None