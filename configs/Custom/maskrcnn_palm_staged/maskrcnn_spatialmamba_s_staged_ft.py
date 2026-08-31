# ==========================================================================
# maskrcnn_palm_staged/maskrcnn_spatialmamba_s_staged_ft.py   (Stage D, v1)
# --------------------------------------------------------------------------
# WV-3 30 cm fine-tuning config. Backbone/neck VERBATIM from the Stage C
# counterpart (architecture untouched). Initialisation and annotation budget
# are injected at LAUNCH via --cfg-options (see tools_staged/
# run_staged_matrix.sh):
#   stagec arm : --cfg-options load_from=<Stage C checkpoint>
#   imagenet arm: omit load_from (backbone init_cfg supplies ImageNet)
#   budget     : --cfg-options train_dataloader.dataset.ann_file=...
# TITAN RTX only. Best Stage C Mamba so far (mean 0.914). SSM memory
# fix carried from Stage C (worst backbone by memory at batch 2).
# ==========================================================================

_base_ = [
    '../_base_palm/_base_maskrcnn_palm_stagec.py',
    '../_base_palm/dataset_sat_30cm_staged.py',
    '../_base_palm/schedule_staged_ft.py',
    '../_base_palm/runtime_palm_staged.py',
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
    dict(
        type='EarlyStoppingHook',
        monitor='coco/segm_mAP_50',
        rule='greater',
        min_delta=0.001,
        patience=8,
    ),
    dict(
        type='PalmBenchmarkLoggingHook',
        # 512, not 1024: Stage D tiles ARE 512 px. The hook profiles at
        # whatever shape it is given, so the inherited 1024 reported GFLOPs
        # about 4x the real cost (Swin-S: 248.3 at 1024). Stage C used 1024 px
        # tiles and its figures are correct as published; only Stage D's were
        # measured at a resolution it never runs.
        input_shape=(3, 512, 512),
        compute_flops=True,
        save_json=True,
    ),
]

model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4],
            ratios=[0.7, 1.0, 1.4],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
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
        frozen_stages=2,          # v3 locked recipe: partial unfreeze
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/maskrcnn_spatialmamba_s_staged_ft'


# Ignore regions for crowns cut by a tile edge.
# The tiler flags a crown below MIN_VISIBLE_FRACTION as partial and the COCO
# converter writes it as iscrowd=1. MaxIoUAssigner only HONOURS those when
# ignore_iof_thr > 0; the project base sets -1, which discards them and puts
# the crown's pixels back into the background -- exactly the false-negative
# supervision the flag exists to prevent. 0.5: an anchor whose intersection-
# over-foreground with an ignore region exceeds half is left unassigned rather
# than labelled negative. Applied identically to both Stage D arms.
#
# Written as an in-place mutation, NOT a second `model = dict(...)`: a config
# file is plain Python, so a second assignment would REBIND the name and throw
# away the backbone/neck defined above. MMEngine's merge semantics apply
# between a config and its bases, not between two statements in one file.
model.setdefault('train_cfg', {}).setdefault('rpn', {})['assigner'] = dict(
    ignore_iof_thr=0.5)
model['train_cfg'].setdefault('rcnn', {})['assigner'] = dict(
    ignore_iof_thr=0.5)

# Detection cap for the WV-3 density.
# The shared Stage C base caps rcnn test output at 300. WorldView-3 tiles are
# 512 px at 30 cm = 2.37 ha, and the reference data holds up to 622 crowns in a
# training tile, 333 in a test tile and 327 in a val tile -- 31 tiles exceed
# 300. On those the cap, not the model, would decide recall, and the reported
# WV-3 number would carry a measurement artefact that looks like a modelling
# result.
#
# The cap is not a threshold: every detection still clears the score threshold
# and NMS first, and the RPN already returns 1000 proposals at test time
# (base line 162), so raising the head to match cannot introduce false
# positives -- it only stops discarding survivors. 1000 clears the observed
# maximum with room to spare. Applied identically to both Stage D arms; the
# shared base is untouched, so the published Stage C benchmark still reports
# at 300.
model.setdefault('test_cfg', {}).setdefault('rcnn', {})['max_per_img'] = 1000

# Same uniform batch as arm B0 (batch 2 + accumulate 2 = effective batch 4),
# so the two arms differ only in initialisation and schedule, never in how
# much data reaches the model per step.
#
# Arm C keeps the shorter 40,000-iteration budget (80,000 samples, ~22 epochs
# at 3,636 tiles). It is a fine-tune from a converged Stage C prior, not a
# training run; the asymmetry is deliberate and stated in Methods, where each
# arm gets the budget its initialisation needs and both are governed by
# EarlyStopping.
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(accumulative_counts=2)

# ---------------------------------------------------------------------------
# Capacity for dense tiles. Identical in both arms and all four backbones.
# ---------------------------------------------------------------------------
# WV-3 tiles hold a median of 12 crowns but a 99th percentile of 292 and a
# maximum of 622. The inherited samplers were sized for the sparser Stage C
# sources and cap supervision far below that:
#
#   RPN  256 x 0.5  = 128 positive anchors per tile
#   RCNN 512 x 0.25 = 128 positive ROIs per tile
#
# On a 300-crown tile more than half the reference crowns then contribute
# nothing to the loss -- the model is not failing to learn them, it is never
# being asked to. Raising both to cover the 99th percentile costs ROI-head
# compute, not correctness, and cannot introduce false positives: these are
# training-time sampling budgets, invisible at inference.
#
# pos_fraction is left at the standard 0.5 / 0.25. Raising it instead of `num`
# would change the positive-to-negative balance the loss is calibrated on,
# which is a different intervention with its own false-positive risk.
model['train_cfg']['rpn']['sampler'] = dict(
    type='RandomSampler', num=512, pos_fraction=0.5,
    neg_pos_ub=-1, add_gt_as_proposals=False)          # 256 positive anchors
model['train_cfg']['rcnn']['sampler'] = dict(
    type='RandomSampler', num=1024, pos_fraction=0.25,
    neg_pos_ub=-1, add_gt_as_proposals=True)           # 256 positive ROIs

# MambaVision-S carried reduced proposal caps (nms_pre 500, max_per_img 512)
# as a Stage C memory accommodation for 1024 px tiles on a 24 GB TITAN. WV-3
# tiles are 512 px, and leaving one backbone with a third of the training
# proposals of the others would confound its row. Set uniformly; the memory
# probe in STAGE_D_README.md 5b covers it.
model['train_cfg']['rpn_proposal'] = dict(
    nms_pre=2000, max_per_img=1000,
    nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0)

# Test-time proposal budget, for dense plantations.
# Raising test_cfg.rcnn.max_per_img to 1000 only moves the bottleneck one
# stage back: the RPN proposed at most 1000 boxes (nms_pre 1000 / max_per_img
# 1000), so on a 622-crown tile nearly every surviving proposal would have to
# land on a distinct object, with no margin for the RPN's own duplicates and
# misses. Recall would then be limited by the proposal budget rather than by
# the detector.
#
# nms_pre is the pre-NMS cut PER FPN LEVEL, and at 30 cm essentially every
# crown is assigned to the finest level, so that level alone must supply the
# whole scene -- 1000 is the binding number, not a comfortable one.
#
# 2000/2000 costs RoI-head compute at inference only; it changes no training
# behaviour and cannot lower precision, since every proposal still has to
# clear the classifier, the score threshold and NMS. This is also what makes
# country-scale WorldView-3 mapping possible: the same limit would otherwise
# truncate every dense plantation tile at inference.
model['test_cfg']['rpn'] = dict(
    nms_pre=2000, max_per_img=2000,
    nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0)
