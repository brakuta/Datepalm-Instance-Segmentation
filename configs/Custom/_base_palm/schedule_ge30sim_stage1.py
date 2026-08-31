# ==========================================================================
# _base_palm/schedule_ge30sim_stage1.py   (Stage-1 full training schedule)
# --------------------------------------------------------------------------
# 19k-tile corpus -> a real training run, NOT a fine-tune. Long schedule,
# Stage-C-style LR (1e-4), validation/early-stopping on GE-30sim val. The
# model learns palm-at-30cm features from abundance here; Stage-2 then
# fine-tunes on the 267 real WV-3 tiles at low LR.
#
# Initialised from Stage B GE weights (set load_from per run) so it starts
# from satellite features and adapts to the coarser 30cm scale.
# ==========================================================================

max_iters    = 60_000        # ~scaled to corpus size; early-stop will govern
val_interval = 2_000

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters,
                 val_interval=val_interval)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=max_iters - 1000, eta_min=1e-6,
         by_epoch=False, begin=1000, end=max_iters),
]

optim_wrapper = dict(
    type='AmpOptimWrapper', dtype='float16', accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999),
                   weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        '.bias': dict(decay_mult=0.0), '.norm': dict(decay_mult=0.0),
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'A_log': dict(decay_mult=0.0), 'D': dict(decay_mult=0.0),
        'dt_proj': dict(decay_mult=0.0)}),
    clip_grad=dict(max_norm=1.0, norm_type=2))

auto_scale_lr = dict(enable=False, base_batch_size=4)
