# ==========================================================================
# 4_satellite_wv3_30cm/maskrcnn_pvtv2_b2_ge30sim_stage1.py   (Stage D, arm P1)
# --------------------------------------------------------------------------
# P1 (Stage 1 of the 30 cm curriculum): train PVT-v2-B2 on the simulated-
# 30 cm Google Earth corpus (GE-30sim, 19,472 tiles). This is the single
# LONG run of the Stage D v3 matrix and the one missing piece for full
# family coverage (the Transformer seat moved from Swin-S to PVTv2-B2, and
# only PVTv2-B2 lacked a GE-30sim pretraining).
#
# INITIALISATION: ImageNet, supplied by the backbone init_cfg below
# (checkpoints/pvtv2_b2_backbone_only.pth). Per the v3 redesign, P1 starts
# from ImageNet -- do NOT pass --cfg-options load_from. This differs from the
# ConvNeXt-T stage1 config, which warm-starts from a Stage B 15 cm checkpoint;
# PVTv2-B2 has no designated 15 cm warm-start in the curriculum, so ImageNet
# init keeps the simulation prior clean. The detection heads train from
# scratch on GE-30sim; the resulting checkpoint feeds arm S (GE-30sim prior)
# of the WV-3 fine-tune.
#
# LAUNCH (WS1 / A5000; ~1 day, early stop):
#   python tools/train.py \
#     configs/Custom/4_satellite_wv3_30cm/maskrcnn_pvtv2_b2_ge30sim_stage1.py \
#     --work-dir /workspace/mmdetection/work_dirs/Stage_D/pvtv2_b2_ge30sim_stage1 \
#     --cfg-options \
#       train_dataloader.num_workers=2 train_dataloader.prefetch_factor=2
#
# Backbone/neck block VERBATIM from maskrcnn_pvtv2_b2_stagec.py (architecture
# untouched) so the P1 -> WV-3 (arm S) transfer is exact. Backbone type is
# 'PyramidVisionTransformerV2'; the import 'mmdet.models.backbones.pvt' MUST
# be present in custom_imports or the backbone will not be found in the
# registry. Anchors scales=[2,4] MATCH the WV-3 staged_ft config so the RPN
# heads transfer cleanly Stage 1 -> Stage 2 (no reinit between stages).
# min-box filter and fp32 OptimWrapper inherited from dataset_ge30sim.py.
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
        'mmdet.models.backbones.pvt',        # registers PyramidVisionTransformerV2
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='coco/segm_mAP_50', rule='greater',
         min_delta=0.001, patience=4),
    dict(type='PalmBenchmarkLoggingHook', input_shape=(3, 512, 512),
         compute_flops=True, save_json=True),
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/pvtv2_b2_backbone_only.pth',
        ),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5,
        # add_extra_convs / relu_before_extra_convs intentionally absent
        # (verbatim from Stage A/B/C). P6 via max-pool, not conv.
    ),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4],                 # MATCH WV-3 staged_ft -> clean transfer
            ratios=[0.7, 1.0, 1.4],
            strides=[4, 8, 16, 32, 64],
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
        '.bias': dict(decay_mult=0.0), '.norm': dict(decay_mult=0.0)}),
    accumulative_counts=2,
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/pvtv2_b2_ge30sim_stage1'
