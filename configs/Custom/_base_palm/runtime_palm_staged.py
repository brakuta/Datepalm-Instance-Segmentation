# ==========================================================================
# _base_palm/runtime_palm_staged.py   (Stage D, v2)
# --------------------------------------------------------------------------
# Single-sensor (WV-3) runtime. v2: CheckpointHook.interval 500 -> 1000 (=
# val_interval) and EarlyStoppingHook patience 8 -> 4 (4 x 1000 = 4000-iter
# plateau). Backbones that declare their OWN custom_hooks (non-native:
# convnext_t, spatialmamba_s, mambavision_s, ...) REPLACE this list and must
# set patience=4 themselves. R50 and Swin inherit this file.
# ==========================================================================

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=25),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000,
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
         min_delta=0.001, patience=4),
    dict(type='PalmBenchmarkLoggingHook', input_shape=(3, 1024, 1024),
         compute_flops=True, save_json=True)]

env_cfg = dict(cudnn_benchmark=True,
               mp_cfg=dict(mp_start_method='fork', opencv_num_threads=1),
               dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends,
                  name='visualizer')
log_processor = dict(type='LogProcessor', window_size=25, by_epoch=False)
log_level = 'INFO'
randomness = dict(seed=0, deterministic=False)
resume = False
load_from = None
