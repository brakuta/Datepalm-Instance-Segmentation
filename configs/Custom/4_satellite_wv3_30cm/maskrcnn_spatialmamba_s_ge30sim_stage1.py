# ==========================================================================
# maskrcnn_palm_staged/maskrcnn_spatialmamba_s_ge30sim_stage1.py
# --------------------------------------------------------------------------
# STAGE 1 of the 30cm curriculum: train SpatialMamba on the LARGE simulated-
# 30cm GE corpus (19,472 tiles), initialised from Stage B GE weights. Learns
# palm-at-30cm features from abundance. The resulting checkpoint initialises
# Stage 2 (fine-tune on 267 real WV-3 tiles).
#
# CRITICAL: anchors = [2,4] (same as the WV-3 v2 / Stage-2 config) so the
# RPN heads transfer cleanly Stage 1 -> Stage 2 (no reinit between stages).
# Backbone/neck VERBATIM from Stage C/D for checkpoint compatibility.
#
# Batch 1 + accumulate 4 (SSM memory). patience=6 (own custom_hooks).
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
        'mmdet.models.backbones.spatialmamba_backbone',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='coco/segm_mAP_50', rule='greater',
         min_delta=0.001, patience=6),
    dict(type='PalmBenchmarkLoggingHook', input_shape=(3, 512, 512),
         compute_flops=True, save_json=True),
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        pretrained='checkpoints/spatialmamba/spatialmamba_small_in1k.pth',
        dims=64, depths=(2, 4, 21, 5), d_state=1,
        drop_path_rate=0.3, mlp_ratio=4.0, norm_layer='ln', frozen_stages=0,
    ),
    neck=dict(
        _delete_=True, type='FPN', in_channels=[64, 128, 256, 512],
        out_channels=256, num_outs=5, add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4],                 # MATCH Stage 2 -> clean transfer
            ratios=[0.7, 1.0, 1.4],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
    roi_head=dict(
        mask_head=dict(
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True,
                           loss_weight=1.5),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7,
                          neg_iou_thr=0.3, min_pos_iou=0.2,
                          match_low_quality=True, ignore_iof_thr=-1),
            allowed_border=-1, pos_weight=-1, debug=False),
        rpn_proposal=dict(nms_pre=2000, max_per_img=2000,
                          nms=dict(type='nms', iou_threshold=0.7),
                          min_bbox_size=0),
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=2000, max_per_img=2000,
                 nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, max_per_img=1000,
                  nms=dict(type='nms', iou_threshold=0.5),
                  mask_thr_binary=0.5),
    ),
)

# SSM memory: batch 1 + accumulate 4
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(accumulative_counts=2)

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/spatialmamba_ge30sim_stage1'

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999),
                   weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        '.bias': dict(decay_mult=0.0), '.norm': dict(decay_mult=0.0),
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'A_log': dict(decay_mult=0.0), 'D': dict(decay_mult=0.0),
        'dt_proj': dict(decay_mult=0.0)}),
    accumulative_counts=2,
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
