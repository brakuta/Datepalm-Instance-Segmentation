# ==========================================================================
# maskrcnn_palm_stagec/maskrcnn_spatialmamba_s_stagec.py   (v1)
# --------------------------------------------------------------------------
# Mask R-CNN + Spatial-Mamba-Small on the unified Stage C corpus
# (UAV 5 cm + GE 15 cm + Aerial 15 cm, proportional sampling). TITAN RTX only.
#
# Stage C counterpart of maskrcnn_spatialmamba_s_ms15.py. Backbone and neck
# are inherited verbatim from Stage B (architecture untouched); only the
# Stage C scaffolding changes.
#
# WHAT CHANGED RELATIVE TO STAGE B
# --------------------------------------------------------------------------
#  1. _base_ chain points at the Stage C bases. schedule_stagec.py is left
#     commented: runtime_palm_stagec.py is the single owner of precision /
#     optimiser / schedule / effective batch (Design B). Do not re-enable the
#     schedule import without removing those blocks from the runtime, or the
#     PRECISION switch exists in two places.
#  2. custom_imports / custom_hooks rebuilt to the full Stage C stack
#     (MeanSensorMetricHook, PerSensorBestCheckpointHook, EarlyStopping on the
#     UAV+GE mean, PalmBenchmarkLoggingHook). MMEngine REPLACES these list
#     keys, so the full stack must be reproduced here. sensor_balanced_sampler
#     -> sensor_balanced_sampler_n (SensorBalancedSamplerN, three sources).
#  3. work_dir relocated to native overlay storage (/root/work_dirs/...).
#  4. SSM MEMORY FIX (per-GPU batch 1, accumulate 4 -> effective batch 4).
#     Spatial-Mamba-S is the worst backbone by memory (Stage B: median 25 GB,
#     p95 28.5 GB, max 35.4 GB at batch 2; reached only iter 10,300 before it
#     was stopped). Per-GPU batch 1 holds the working set near ~13-18 GB while
#     keeping the effective batch at 4. Launch with
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
#  5. compute_flops=True retained: torch.jit.trace succeeds on Spatial-Mamba's
#     pure-SSM forward graph.
#
# Architecture (verbatim from Stage A/B): dims=64 (SAME as Tiny — capacity
# scales via stage-3/4 depth, not channel width), depths=(2,4,21,5),
# d_state=1, FPN in_channels=[64,128,256,512]. Not a config error.
# ==========================================================================

_base_ = [
    '../_base_palm/_base_maskrcnn_palm_stagec.py',
    '../_base_palm/dataset_UAV_GE_Aerial_pooled_C.py',
    #'../_base_palm/schedule_stagec.py',
    '../_base_palm/runtime_palm_stagec.py',
]

custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.sensor_balanced_sampler_n',
        'configs.Custom._base_palm.per_sensor_best_checkpoint_hook',
        'configs.Custom._base_palm.mean_sensor_metric_hook',
        'configs.Custom._base_palm.dataloader_worker_init',
        'configs.Custom._base_palm.nms_fp32_guard',
        'mmdet.models.backbones.spatialmamba_backbone',
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
        monitors=dict(UAV='UAV/coco/segm_mAP_50', GE='GE/coco/segm_mAP_50'),
        rule='greater',
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
        compute_flops=True,
        save_json=True,
    ),
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        pretrained='/workspace/mmdetection/checkpoints/spatialmamba/spatialmamba_small_in1k.pth',
        dims=64,
        depths=(2, 4, 21, 5),
        d_state=1,
        drop_path_rate=0.3,
        mlp_ratio=4.0,
        norm_layer='ln',
        frozen_stages=0,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 256, 512],   # same as Tiny -- not a config error
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = '/root/work_dirs/Stage_C/maskrcnn_spatialmamba_s_stagec'

# ==========================================================================
# SSM MEMORY FIX (Stage C, TITAN RTX, 24 GB)  --  Spatial-Mamba-S.
# Per-GPU batch 1 + accumulate 4 -> effective batch 4 (Stage B comparable).
# chunk_size widened to 4 so each optimiser step draws same-source tiles.
# Both dicts deep-merge over the dataset / runtime definitions.
# Launch with: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ==========================================================================
train_dataloader = dict(
    batch_size=1,
    sampler=dict(chunk_size=4),
)
optim_wrapper = dict(accumulative_counts=4)

# ==========================================================================
# SECOND LEVER (commented)  --  proposal cap. Enable ONLY if, after the
# batch-1 fix, VRAM still grazes 24 GB (verify in the first 5 minutes via
# nvidia-smi). This is the identical cap MambaVision-S carries. Uncomment
# and relaunch; the dict deep-merges over the base detector train_cfg.
# ==========================================================================
# model = dict(
#     train_cfg=dict(
#         rpn=dict(sampler=dict(num=128)),
#         rpn_proposal=dict(nms_pre=500, max_per_img=512),
#     ),
# )
