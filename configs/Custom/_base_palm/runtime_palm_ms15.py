# ==========================================================================
# _base_palm/runtime_palm_ms15.py
# --------------------------------------------------------------------------
# Stage B (MS-15 cm) runtime configuration for the date palm backbone
# benchmark. Covers: environment, hooks, visualiser, logging.
#
# This file is a Stage-B-scoped fork of runtime_palm.py. The Stage A
# runtime (runtime_palm.py) is preserved unchanged.
#
# -----------------------------------------------------------------------
# STAGE B EVALUATION DESIGN:
#
#   Training:   pooled GE + Aerial with SensorBalancedSampler(alpha=0.3)
#   Validation: GE-only (single bare CocoMetric)
#   Aerial:     held out for post-hoc evaluation via Evaluation/evaluate_model.py
#               invoked through tools/test.py against the best GE-val
#               checkpoint; not monitored during training
#
# Model-selection key:
#   'coco/segm_mAP_50'  (bare CocoMetric, no MultiDatasetsEvaluator
#                        wrapper, since validation is single-source)
#
# CheckpointHook.save_best and EarlyStoppingHook.monitor both reference
# this key. MeanSensorMetricHook is NOT used: under GE-only validation
# there is no second source to mean against.
#
# -----------------------------------------------------------------------
# MACHINE CONFIGURATION — edit this section when deploying to a
# different workstation. No CLI flags are needed.
#
#   TITAN RTX (64 GB RAM)   : no changes needed
#   RTX A5000 (31.9 GB RAM) : no changes needed (num_workers and
#                              pin_memory controlled in
#                              dataset_MS15_pooled.py)
#
# NOTE — OMP_NUM_THREADS / MKL_NUM_THREADS:
#   These environment variables are intentionally NOT set here. Stage A
#   training ran without them on both workstations. Setting them via
#   env_cfg caused non-deterministic CUDA errors on the RTX A5000
#   (observed as grad_norm: nan followed by RuntimeError: CUDA error:
#   unknown error at iter ~850 during Swin-S training). The conflict
#   arises because forcing OMP_NUM_THREADS inside a container interferes
#   with PyTorch's internal thread pool management on certain CUDA driver
#   versions. Leaving them unset preserves Stage A behaviour and avoids
#   this instability.
#
# seed: fixed for reproducibility across all backbones and both machines.
# resume: set to True only when relaunching an interrupted run.
#   MMEngine reads work_dir/last_checkpoint automatically when True.
#   Reset to False for every fresh run.
# -----------------------------------------------------------------------
seed   = 0
resume = True  # auto-resume: continues from the latest checkpoint in
               # work_dir if one exists; a fresh run is unaffected

# --- Environment ----------------------------------------------------------
# Matches Stage A environment exactly. No thread-count overrides.
env_cfg = dict(
    cudnn_benchmark=True,
    mp_start_method='spawn',
    dist_cfg=dict(backend='nccl'),
)

custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler',
    ],
    allow_failed_imports=False,
)

# --- Default hooks --------------------------------------------------------
# CRITICAL: save_best='coco/segm_mAP_50' matches the key emitted by the
# bare CocoMetric in dataset_MS15_pooled.py.
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        save_best='coco/segm_mAP_50',
        rule='greater',
        max_keep_ckpts=3,
        save_last=True,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

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

# --- Visualiser -----------------------------------------------------------
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
    name='visualizer',
)

# --- Logging --------------------------------------------------------------
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
log_level = 'INFO'
default_scope = 'mmdet'

# --- Reproducibility ------------------------------------------------------
randomness = dict(seed=seed, deterministic=False)

# --- Resume ---------------------------------------------------------------
load_from = None