# Copyright (c) Stage C Date Palm Benchmark.
# _base_maskrcnn_palm_stagec.py
#
# Mask R-CNN detector base for Stage C. Inherits the Stage B detector
# architecture and overrides only the anchor scales to cover the union GT
# distribution across UAV (5cm), GE (15cm), and Aerial (15cm).
#
# Anchor scale derivation
# -----------------------
# Stage A (UAV 5cm) used scales=[4, 8, 16] (≈ 32-128 px effective).
# Stage B (GE+Aerial 15cm) used scales=[2, 4, 8] (≈ 16-64 px effective).
#
# Stage C pools all three at native GSD. The union distribution spans:
#   - UAV: p5≈40 px, p50≈75 px, p95≈140 px (Stage A statistics)
#   - GE+Aerial: p5≈24 px, p50≈44 px, p95≈79 px (Stage B statistics, n=830,830)
#
# Effective anchor sizes at strides [4, 8, 16, 32, 64] with scales=[2, 4, 8]:
#   level P2 (stride 4):  8-32 px
#   level P3 (stride 8):  16-64 px
#   level P4 (stride 16): 32-128 px
#   level P5 (stride 32): 64-256 px
#   level P6 (stride 64): 128-512 px
#
# This coverage spans 8-512 px, which encompasses:
#   - GE p5 (24 px) at P3
#   - GE p95 (79 px) at P4
#   - UAV p5 (40 px) at P3/P4
#   - UAV p95 (140 px) at P5
#
# The Stage B scales=[2, 4, 8] therefore cover the union without modification.
# This is the most defensible choice — it does not require re-derivation, and
# it lets us claim that Stage C uses the same anchor design as Stage B
# (one fewer free parameter in the cross-stage comparison).

model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32,
    ),
    # backbone is set per-config (R50, Swin-S, etc.)
    backbone=None,
    # neck is set per-config (most backbones use FPN with default in/out channels)
    neck=None,
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8],            # ← inherited from Stage B (union-defensible)
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,             # DatePalm only
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0,
            ),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            mask_size=28,
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,              # NB: 0.05, not 0.3 (Stage A bug fix)
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5,
        ),
    ),
)
