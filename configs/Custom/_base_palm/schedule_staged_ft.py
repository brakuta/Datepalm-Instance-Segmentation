# ==========================================================================
# _base_palm/schedule_staged_ft.py   (Stage D, v3)
# --------------------------------------------------------------------------
# v3: rescaled for the enlarged WV-3 set (2,413 train tiles, ~1,207 iter/epoch
# at batch 2). max_iters 8000 -> 40000 (~33 epochs); EarlyStoppingHook decides
# the real cutoff. val_interval 1000 -> 1200 (~1 val/epoch). Warmup 250 -> 1000.
# Cosine T_max tracks max_iters. fp16 AmpOptimWrapper -> fp32 OptimWrapper to
# remove the NaN-overflow risk (matches the ge30sim_stage1 fix); rely on
# FilterAnnotations in the pipeline as the real degenerate-box guard.
# Set EarlyStoppingHook patience=10 in the per-backbone custom_hooks.
#
# LOCKED RECIPE (Stage D v3, from the completed freeze-regime sweep; selected
# on a preliminary annotation version and held fixed for all reported arms
# B0/S/C/BU -- never re-tuned):
#   * partial unfreeze -- frozen_stages=2 set in each per-backbone config
#     (freezes patch/stem + the first two stages; stages 2-3 trainable);
#   * head LR 2e-5;
#   * backbone lr_mult 0.01 (applies to the *unfrozen* stages 2-3).
# ==========================================================================
max_iters    = 40_000
val_interval = 1_200
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters,
                 val_interval=val_interval)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=max_iters - 1000, eta_min=1e-7,
         by_epoch=False, begin=1000, end=max_iters),
]
optim_wrapper = dict(
    type='OptimWrapper', accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=2e-5, betas=(0.9, 0.999),
                   weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.01),
        '.bias': dict(decay_mult=0.0), '.norm': dict(decay_mult=0.0),
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'A_log': dict(decay_mult=0.0), 'D': dict(decay_mult=0.0),
        'dt_proj': dict(decay_mult=0.0)}),
    clip_grad=dict(max_norm=1.0, norm_type=2))
auto_scale_lr = dict(enable=False, base_batch_size=4)
