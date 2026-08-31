# ==========================================================================
# maskrcnn_palm/_base_maskrcnn_palm.py
# --------------------------------------------------------------------------
# Mask R-CNN architecture base for the date palm benchmark.
# This file defines the detector ONLY. Dataset, schedule, and runtime are
# inherited by the per-run configs from _base_palm/.
#
# Backbone and neck in_channels are left to the per-run configs because
# they are the only parameters that vary between backbones.
#
# Protocol notes:
#   - mask_size = 28 (standard; 56 doubled mask-head VRAM with no quality gain)
#   - Anchor scales [4, 8, 16], ratios [0.75, 1.0, 1.25] — palms are roughly
#     circular in nadir view, no benefit from elongated anchors
#   - nms_pre = 1000 (sufficient for single-class detection at 1024x1024)
# ==========================================================================

_base_ = [
    '../../_base_/models/mask-rcnn_r50_fpn.py',
]

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

    # Backbone and neck are left as placeholders — every per-run config
    # must override them with _delete_=True.
    backbone=None,
    neck=None,

    # --- RPN head ----------------------------------------------------------
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8, 16],
            ratios=[0.75, 1.0, 1.25],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1., 1., 1., 1.],
        ),
        loss_cls=dict(
            _delete_=True,
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),

    # --- RoI head ----------------------------------------------------------
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            _delete_=True,
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
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
                type='CrossEntropyLoss',
                use_mask=True,
                loss_weight=1.0,
            ),
        ),
    ),

    # --- Training config ---------------------------------------------------
    train_cfg=dict(
        _delete_=True,
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
            nms_pre=1000,
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

    # --- Test config -------------------------------------------------------
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            # score_thr=0.05 follows mmdetection/COCO convention and
            # preserves the precision-recall tail required for unbiased
            # primary mAP. The earlier value of 0.3 truncated the PR
            # curve and depressed mAP / mAP_75 / mAP_s.
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            # Raised from 300 to 500 to match proposal_nums=(100,300,1000)
            # in the evaluator and to avoid a detection ceiling on dense
            # palm plantation tiles.
            max_per_img=500,
            mask_thr_binary=0.5,
        ),
    ),
)
