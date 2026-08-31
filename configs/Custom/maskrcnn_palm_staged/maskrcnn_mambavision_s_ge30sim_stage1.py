# ==========================================================================
# maskrcnn_palm_staged/maskrcnn_mambavision_s_ge30sim_stage1.py  (Stage D, A1)
# --------------------------------------------------------------------------
# A1 (Stage 1 of the 30 cm curriculum): train MambaVision-S on the simulated-
# 30 cm Google Earth corpus (GE-30sim, 19,472 tiles), INITIALISED AT LAUNCH
# from the MambaVision-S Stage B (15 cm) checkpoint via --cfg-options load_from.
#
# LAUNCH (WS1 / TITAN RTX):
#   python tools/train.py \
#     configs/Custom/maskrcnn_palm_staged/maskrcnn_mambavision_s_ge30sim_stage1.py \
#     --work-dir /workspace/mmdetection/work_dirs/Stage_D/mambavision_s_ge30sim_stage1 \
#     --cfg-options \
#       load_from=work_dirs/Stage_B/maskrcnn_mambavision_s_ms15/best_coco_segm_mAP_50_iter_75000.pth
#
# Backbone/neck and MEMORY ACCOMMODATIONS verbatim from
# maskrcnn_mambavision_s_staged_ft.py: proposal cap (nms_pre=500,
# max_per_img=512) governs memory (NOT batch 1); RandomSampler(num=128);
# compute_flops=False (hybrid SSM+attention graph fails torch.jit.trace);
# cudnn_benchmark=False; gradient_checkpointing=False (AMP+ckpt diverges,
# and here fp32 is used regardless). Anchors scales=[2,4]; fp32 OptimWrapper.
# ==========================================================================

_base_ = [
    '../_base_palm/_base_maskrcnn_palm_stagec.py',
    '../_base_palm/dataset_ge30sim.py',
    '../_base_palm/schedule_ge30sim_stage1.py',
    '../_base_palm/runtime_ge30sim_stage1.py',
]

custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.nms_fp32_guard',
        'mmdet.models.backbones.mambavision_backbone',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='coco/segm_mAP_50', rule='greater',
         min_delta=0.001, patience=4),
    dict(type='PalmBenchmarkLoggingHook', input_shape=(3, 512, 512),
         compute_flops=False, save_json=True),   # hybrid graph not traceable
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=1),
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
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4],                 # MATCH Stage B / A2 -> clean transfer
            ratios=[0.7, 1.0, 1.4],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
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

max_iters = 30_000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters,
                 val_interval=2_000)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=max_iters - 1000, eta_min=1e-6,
         by_epoch=False, begin=1000, end=max_iters),
]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999),
                   weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        '.bias': dict(decay_mult=0.0), '.norm': dict(decay_mult=0.0),
        'A_log': dict(decay_mult=0.0), 'D': dict(decay_mult=0.0),
        'dt_proj': dict(decay_mult=0.0)}),
    accumulative_counts=2,
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/mambavision_s_ge30sim_stage1'
