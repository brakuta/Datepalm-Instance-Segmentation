# ==========================================================================
# maskrcnn_palm/_base_maskrcnn_palm_ms15.py
# --------------------------------------------------------------------------
# Mask R-CNN architecture base for the Stage B (MS-15 cm) date palm
# benchmark.
#
# This file is a Stage-B-scoped fork of _base_maskrcnn_palm.py. The
# Stage A base (`_base_maskrcnn_palm.py`) is preserved unchanged so that
# new UAV-5 cm training runs remain reproducible and comparable to the
# existing Stage A results table. Stage B configs MUST inherit from this
# file, not the Stage A base.
#
# Difference vs Stage A base:
#   - Anchor scales: [2, 4, 8]    (Stage A: [4, 8, 16])
#   All other detector hyperparameters are identical to the Stage A base.
#
# Stage B anchor revision rationale:
#   The anchor grid was determined empirically from the geometric-mean
#   side-length distribution of ground-truth bounding boxes in the pooled
#   MS-15 cm training corpus (n = 830,830 instances). Pooled percentiles:
#   p5 = 23.9 px, p50 = 44.0 px, p95 = 79.0 px. The Aerial source
#   contributes a heavier small-object tail (p5 = 12.5 px, p50 = 30.0 px)
#   than the GE source (p5 = 28.5 px, p50 = 46.0 px). The Stage A grid
#   ([4, 8, 16] over strides [4, 8, 16, 32, 64]) spans 16-1024 px, with
#   approximately four of seven unique anchor sizes covering negligible
#   empirical mass at MS-15 cm GSD. Scales [2, 4, 8] produce a grid
#   spanning 8-512 px with the dense coverage band at 16-128 px aligned
#   to the pooled p5-p95 envelope. The same anchor configuration is
#   applied across all eight Stage B backbones to preserve architectural
#   comparability within the cohort.
#
# Protocol notes (inherited from Stage A; unchanged):
#   - mask_size = 28 (standard; 56 doubled mask-head VRAM with no quality
#     gain in Stage A ablations).
#   - Anchor ratios [0.75, 1.0, 1.25] reflect the approximately circular
#     nadir geometry of mature palm crowns; no benefit from elongated
#     anchors.
#   - nms_pre = 1000 (sufficient for single-class detection at 1024x1024).
#   - score_thr = 0.05 (mmdetection/COCO convention) and
#     max_per_img = 500 retained from the late-Stage A revision.
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

    # Backbone and neck are left as placeholders - every per-run config
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
            # Stage B anchor scales (revised from [4, 8, 16]).
            # Resulting grid (geometric-mean side, px) per FPN level:
            #   P2 (stride 4):  [ 8, 16, 32]
            #   P3 (stride 8):  [16, 32, 64]
            #   P4 (stride 16): [32, 64, 128]
            #   P5 (stride 32): [64, 128, 256]
            #   P6 (stride 64): [128, 256, 512]
            scales=[2, 4, 8],
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
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=500,
            mask_thr_binary=0.5,
        ),
    ),
)
