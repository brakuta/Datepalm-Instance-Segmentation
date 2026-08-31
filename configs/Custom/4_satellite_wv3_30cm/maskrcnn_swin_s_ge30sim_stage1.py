# ==========================================================================
# 4_satellite_wv3_30cm/maskrcnn_swin_s_ge30sim_stage1.py   (Stage D, A1)
# --------------------------------------------------------------------------
# A1 (Stage 1 of the 30 cm curriculum): train Swin-S on the simulated-30 cm
# Google Earth corpus (GE-30sim, 19,472 tiles), INITIALISED AT LAUNCH from the
# Swin-S Stage B (15 cm) checkpoint via --cfg-options load_from=... . The
# resulting checkpoint initialises A2 (fine-tune on refined WV-3).
#
# LAUNCH (WS2 / A5000):
#   python tools/train.py \
#     configs/Custom/4_satellite_wv3_30cm/maskrcnn_swin_s_ge30sim_stage1.py \
#     --work-dir /workspace/mmdetection/work_dirs/Stage_D/swin_s_ge30sim_stage1 \
#     --cfg-options \
#       load_from=work_dirs/Stage_B/maskrcnn_swin_s_ms15/best_coco_segm_mAP_50_iter_50000.pth \
#       train_dataloader.num_workers=2 train_dataloader.prefetch_factor=2
#
# Backbone/neck VERBATIM from maskrcnn_swin_s_staged_ft.py (architecture
# identity across the curriculum -> clean Stage B -> A1 -> A2 transfer).
# Anchors scales=[2,4] (6 ch) MATCH Stage B and A2 so RPN heads transfer
# without reinitialisation. fp32 OptimWrapper (removes the fp16 loss_bbox
# overflow risk); FilterAnnotations guard inherited from dataset_ge30sim.py.
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
    ],
    allow_failed_imports=False,
)

# Reproduce runtime custom_hooks with the reduced-cost patience (see below).
custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='coco/segm_mAP_50', rule='greater',
         min_delta=0.001, patience=4),
    dict(type='PalmBenchmarkLoggingHook', input_shape=(3, 512, 512),
         compute_flops=True, save_json=True),
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        init_cfg=None,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        # add_extra_convs intentionally absent (verbatim Stage A/B; P6 max-pool)
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

# Reduced pretraining cost: transfer from a competent 15 cm Stage B model
# converges well before 60k. Cap at 30k; early stopping (patience 4 x 2000 =
# 8000-iter plateau) governs the actual stop. Cosine T_max tracks max_iters.
max_iters = 30_000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters,
                 val_interval=2_000)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=max_iters - 1000, eta_min=1e-6,
         by_epoch=False, begin=1000, end=max_iters),
]

# fp32 OptimWrapper (override the inherited AmpOptimWrapper fp16).
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999),
                   weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        '.bias': dict(decay_mult=0.0), '.norm': dict(decay_mult=0.0),
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0)}),
    accumulative_counts=2,
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/swin_s_ge30sim_stage1'
