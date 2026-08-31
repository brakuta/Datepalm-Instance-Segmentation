# ==========================================================================
# _base_palm/runtime_ge30sim_stage1.py   (Stage-1 runtime)
# --------------------------------------------------------------------------
# Single-source (GE-30sim) runtime. CheckpointHook interval = val_interval
# (2000); EarlyStoppingHook patience=6 (6 x 2000 = 12,000-iter plateau,
# appropriate for a long 60k full-training run). Per-backbone configs that
# declare their own custom_hooks (SSM/ConvNeXt) must reproduce patience here.
# ==========================================================================

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000,
                    max_keep_ckpts=3, save_last=True,
                    save_best='coco/segm_mAP_50', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

custom_imports = dict(
    imports=['configs.Custom._base_palm.benchmark_logging_hook',
             'configs.Custom._base_palm.nms_fp32_guard'],
    allow_failed_imports=False)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='coco/segm_mAP_50', rule='greater',
         min_delta=0.001, patience=6),
    dict(type='PalmBenchmarkLoggingHook', input_shape=(3, 512, 512),
         compute_flops=True, save_json=True)]

env_cfg = dict(cudnn_benchmark=True,
               mp_cfg=dict(mp_start_method='fork', opencv_num_threads=1),
               dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends,
                  name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
log_level = 'INFO'
randomness = dict(seed=0, deterministic=False)
resume = False
load_from = None
