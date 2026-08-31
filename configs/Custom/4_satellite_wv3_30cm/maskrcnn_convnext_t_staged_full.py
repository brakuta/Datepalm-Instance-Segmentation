# ==========================================================================
# 4_satellite_wv3_30cm/maskrcnn_convnext_t_staged_full.py   (Stage D, v4)
# --------------------------------------------------------------------------
# ARM B0 -- full training on real WV-3 30 cm from ImageNet weights.
#
# Identical to maskrcnn_convnext_t_staged_ft.py in EVERY architectural
# respect; the two differ only in (a) the schedule base and (b) frozen_stages.
# Keep the model/neck blocks below byte-identical to the _ft sibling -- the
# B0-vs-C comparison is only interpretable if the network is the same.
#
#   schedule : schedule_staged_full.py (lr 1e-4, backbone lr_mult 0.1, 60k)
#              instead of schedule_staged_ft.py (2e-5 / 0.01 / 40k)
#   freezing : none. ImageNet features are far from 30 cm nadir satellite
#              imagery, so freezing the stem and first two stages -- correct
#              for a near-domain prior -- would cripple this arm and make the
#              baseline a strawman.
#
# NO load_from. The backbone init_cfg supplies ImageNet; passing a load_from
# here would silently turn arm B0 into arm C.
# ==========================================================================

_base_ = [
    '../_base_palm/_base_maskrcnn_palm_stagec.py',
    '../_base_palm/dataset_sat_30cm_staged.py',
    '../_base_palm/schedule_staged_full.py',
    '../_base_palm/runtime_palm_staged.py',
]

custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.nms_fp32_guard',
        'mmpretrain.models.backbones.convnext',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/segm_mAP_50',
        rule='greater',
        min_delta=0.001,
        # 12, not the fine-tune arm's 8: a from-scratch-ish run improves in
        # longer, flatter steps and patience 8 (9,600 iters) would cut it off
        # while it is still climbing.
        patience=12,
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
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        gap_before_final_norm=False,
        frozen_stages=0,          # arm B0: nothing frozen (mmpretrain: 0=none)
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/convnext_tiny_backbone_only.pth',
        ),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        # add_extra_convs intentionally absent (verbatim Stage A/B).
    ),
)

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/maskrcnn_convnext_t_staged_full'

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

# Uniform batch and iteration budget across backbones.
# Comparability requires every backbone to see the SAME number of samples at
# the SAME effective batch. SpatialMamba-S previously ran at per-GPU batch 1
# -- its Stage C accommodation -- while the others ran at batch 2, so at equal
# max_iters it saw half the samples and any deficit in the SSM row would have
# been a budget artefact rather than a property of the architecture.
#
# All four now use batch 2 + accumulative_counts 2 = effective batch 4. The
# batch-1 accommodation was sized for 1024 px Stage C tiles on a 24 GB TITAN;
# WV-3 tiles are 512 px, a quarter of the pixels, so the constraint that
# forced it does not apply here. VERIFY ONCE per backbone with a short run
# before committing to the full schedule -- see STAGE_D_README.md step 6.
#
# 60,000 iterations x batch 2 = 120,000 samples, exactly the Stage C exposure
# (120,000 iterations x batch 1), which is why the two stages are directly
# comparable despite the different iteration counts. At 3,636 WV-3 train tiles
# that is ~33 epochs. EarlyStopping on coco/segm_mAP_50 decides the real
# stopping point; the selected iteration is reported per run.
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(accumulative_counts=2)

max_iters = 60_000
train_cfg = dict(max_iters=max_iters)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False,
         begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=max_iters - 1000,
         eta_min=1e-6, by_epoch=False, begin=1000, end=max_iters),
]

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
