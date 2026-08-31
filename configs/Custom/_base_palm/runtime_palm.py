# ==========================================================================
# _base_palm/runtime_palm.py
# --------------------------------------------------------------------------
# Shared runtime configuration for the date palm backbone benchmark.
# Covers: environment, hooks, visualiser, logging.
# Detector-agnostic and source-agnostic — every config in the benchmark
# imports this file.
#
# Stage B modifications (vs the Stage A version):
#   - sensor_balanced_sampler added to custom_imports so the Stage B
#     dataloader can resolve type='SensorBalancedSampler' from the
#     DATA_SAMPLERS registry.
#   - PalmBenchmarkLoggingHook.compute_flops re-enabled. FLOPs are
#     computed at a fixed (3, 1024, 1024) input shape and are therefore
#     identical between Stage A and Stage B for any given backbone.
#     Re-enabling produces a one-time GFLOPs entry in benchmark_record.json
#     and matches the FLOPs column already reported in the Stage A
#     results table.
#   - EarlyStoppingHook.monitor and CheckpointHook.save_best are keyed
#     on 'GE_val/coco/segm_mAP_50' to match the MultiDatasetsEvaluator
#     output prefix in dataset_MS15_pooled.py. GE-val is preferred over
#     Aerial-val as the trigger key because (i) it has 7.2x more tiles
#     (896 vs 125), giving lower variance and more reliable stopping
#     decisions at min_delta=0.001, and (ii) Aerial-val is single-
#     location and would bias best-checkpoint selection toward
#     location-specific overfitting.
#
#     Stage A configs (UAV-5 cm, single-source val) continue to use the
#     legacy 'coco/segm_mAP_50' key. To run a Stage A config under this
#     runtime, override the two monitor keys via --cfg-options.
# ==========================================================================

# --- Environment ----------------------------------------------------------
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
# save_best='GE_val/coco/segm_mAP_50' selects the checkpoint that
# maximises segmentation mAP@50 on the GE val partition. Aerial_val
# metrics appear in the log every validation interval but are not
# coupled to checkpoint selection.
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        # Aligned with val_interval=5000 in all schedule files so that
        # every validation is paired with a scheduled checkpoint candidate.
        interval=5000,
        save_best='GE_val/coco/segm_mAP_50',
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
        monitor='GE_val/coco/segm_mAP_50',
        rule='greater',
        min_delta=0.001,
        patience=8,
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
# Seed is intentionally NOT fixed here. For the benchmark paper, each
# backbone may be run with multiple seeds to report mean ± std. Fix the
# seed via the --cfg-options CLI flag, e.g.:
#   python tools/train.py <config> --cfg-options randomness.seed=0
randomness = dict(seed=None, deterministic=False)

# --- Resume (default off) -------------------------------------------------
# Individual per-run configs may override these to resume.
resume = False
load_from = None
