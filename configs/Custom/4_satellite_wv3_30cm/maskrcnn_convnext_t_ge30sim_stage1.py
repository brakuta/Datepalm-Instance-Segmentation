# ==========================================================================
# 4_satellite_wv3_30cm/maskrcnn_convnext_t_ge30sim_stage1.py   (Stage D, A1)
# --------------------------------------------------------------------------
# A1 (Stage 1 of the 30 cm curriculum): train ConvNeXt-T on the simulated-
# 30 cm Google Earth corpus (GE-30sim, 19,472 tiles), INITIALISED AT LAUNCH
# from the ConvNeXt-T Stage B (15 cm) checkpoint via --cfg-options load_from.
#
# LAUNCH (WS2 / A5000):
#   python tools/train.py \
#     configs/Custom/4_satellite_wv3_30cm/maskrcnn_convnext_t_ge30sim_stage1.py \
#     --work-dir /workspace/mmdetection/work_dirs/Stage_D/convnext_t_ge30sim_stage1 \
#     --cfg-options \
#       load_from=work_dirs/Stage_B/maskrcnn_convnext_t_ms15/best_coco_segm_mAP_50_iter_65000.pth \
#       train_dataloader.num_workers=2 train_dataloader.prefetch_factor=2
#
# Backbone block VERBATIM from the Stage B config (maskrcnn_convnext_t_ms15.py)
# so the Stage B -> A1 load_from is exact. Type is 'mmpretrain.ConvNeXt'
# (bridged via MMPretrain); the corresponding import MUST be present in
# custom_imports or the backbone will not be found in the registry.
# Anchors scales=[2,4]; fp32 OptimWrapper; FilterAnnotations inherited.
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
        'mmpretrain.models',                 # registers mmpretrain.ConvNeXt
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
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=None,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
    ),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4],                 # MATCH Stage B / A2 -> clean transfer
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

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/convnext_t_ge30sim_stage1'
