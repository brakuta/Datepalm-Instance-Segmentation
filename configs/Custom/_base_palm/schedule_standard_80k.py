# ==========================================================================
# _base_palm/schedule_standard_80k.py
# --------------------------------------------------------------------------
# Stage A (UAV 5 cm) standard single-source training schedule.
#
# Reconstructed verbatim from the dumped run config of
#   work_dirs/Stage_A/maskrcnn_r50_uav5cm/20260424_142705/vis_data/config.py
# so the restored file is the true training record for the six Stage A
# configs that inherit it (r50, r101, swin_t, swin_s, convnext_t, pvtv2_b2).
#
# Protocol: 80k iterations, AdamW (lr 1e-4, wd 0.05), backbone lr_mult 0.1,
# AMP fp16, effective batch 4 (batch 2 x accumulative_counts 2),
# 1k linear warmup -> cosine anneal to eta_min 1e-6, grad-norm clip 1.0.
#
# NOTE: This schedule is consumed only by the training loop. Runner.test()
# does not read optim_wrapper, param_scheduler, or train_cfg, so restoring
# this file changes no evaluation result; it only lets the six configs
# parse and keeps the training record accurate.
# ==========================================================================

# --- Optimizer (AMP fp16, AdamW, backbone 0.1x LR) ------------------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    accumulative_counts=2,
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            '.bias': dict(decay_mult=0.0),
            '.norm': dict(decay_mult=0.0),
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
        }),
)

# --- LR schedule: 1k linear warmup -> cosine to 1e-6 over 80k -------------
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR',
         T_max=79000, eta_min=1e-06,
         by_epoch=False, begin=1000, end=80000),
]

# --- Loops ----------------------------------------------------------------
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=80000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# --- LR auto-scaling (disabled; effective batch fixed at 4) ---------------
auto_scale_lr = dict(base_batch_size=4, enable=False)
