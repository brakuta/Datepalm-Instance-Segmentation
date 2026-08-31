# ==========================================================================
# 3_unified_multisource/maskrcnn_mambavision_s_stagec.py   (v1)
# --------------------------------------------------------------------------
# Mask R-CNN + MambaVision-S on the unified Stage C corpus
# (UAV 5 cm + GE 15 cm + Aerial 15 cm, proportional sampling). TITAN RTX only.
#
# Stage C counterpart of maskrcnn_mambavision_s_ms15.py. Backbone and neck
# inherited verbatim from Stage B (architecture untouched).
#
# WHAT CHANGED RELATIVE TO STAGE B
# --------------------------------------------------------------------------
#  1. _base_ chain -> Stage C bases; schedule_stagec.py left commented
#     (runtime owns precision/optimiser/schedule).
#  2. custom_imports / custom_hooks rebuilt to the full Stage C stack
#     (MMEngine replaces these list keys). sensor_balanced_sampler ->
#     sensor_balanced_sampler_n. compute_flops=False is preserved (see below).
#  3. work_dir relocated to native overlay storage (/root/work_dirs/...).
#  4. NO batch fix. MambaVision-S keeps the light-backbone condition (per-GPU
#     batch 2, accumulate 2 -> effective batch 4). Its memory is governed by
#     the proposal cap below (RPN sampler 128, nms_pre 500, max_per_img 512),
#     which in Stage B held steady-state peak at ~14-15 GB. This is the lever
#     instead of batch 1.
#  5. env_cfg override restated in the CORRECT Stage C (nested) form: only
#     cudnn_benchmark is flipped to False; mp_cfg (fork, opencv_num_threads=1)
#     and dist_cfg are reproduced from the runtime. The Stage B file placed
#     mp_start_method='spawn' at the top level of env_cfg, where MMEngine
#     ignored it (it must sit under mp_cfg); fork is the proven Stage A/B
#     behaviour and is used here explicitly.
#
# LOCKED / DO NOT CHANGE
# --------------------------------------------------------------------------
#  - compute_flops=False: torch.jit.trace fails on MambaVision's hybrid
#    SSM+attention graph. FLOPs are taken from the published reference
#    (Hatamizadeh and Kautz, 2024) scaled to 1024x1024.
#  - gradient_checkpointing=False: AMP + checkpointing diverges (float32/
#    float16 mismatch) on the hybrid stages.
#  - The proposal cap, cudnn_benchmark=False, and the allocator env var
#    together constitute the reported "MambaVision-S memory accommodation".
#  - Launch with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
#    cudnn_benchmark=False costs ~5-10% throughput.
#
# Architecture (verbatim): type='mamba_small_vision_timm', drop_path_rate=0.2,
# frozen_stages=0, local_files_only=True; FPN in_channels=[96,192,384,768].
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
        'mmdet.models.backbones.mambavision_backbone',
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
        compute_flops=False,       # hybrid SSM+attention: jit.trace fails
        save_json=True,
    ),
]

# --- Environment override (Stage C nested form) ---------------------------
# Only cudnn_benchmark differs from the runtime; mp_cfg/dist_cfg reproduced.
env_cfg = dict(
    cudnn_benchmark=False,         # variable ROI-head shapes -> avoid workspace re-alloc
    mp_cfg=dict(
        mp_start_method='fork',
        opencv_num_threads=1,
    ),
    dist_cfg=dict(backend='nccl'),
)

model = dict(
    backbone=dict(
        _delete_=True,
        type='mamba_small_vision_timm',
        drop_path_rate=0.2,
        frozen_stages=0,
        gradient_checkpointing=False,
        local_files_only=True,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
    # Memory accommodations for MambaVision-S only (see header).
    train_cfg=dict(
        rpn=dict(
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
        ),
        rpn_proposal=dict(
            nms_pre=500,
            max_per_img=512,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
    ),
)

work_dir = '/root/work_dirs/Stage_C/maskrcnn_mambavision_s_stagec'
