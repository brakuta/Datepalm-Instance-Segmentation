# ==========================================================================
# _base_palm/schedule_mamba_120k.py
# --------------------------------------------------------------------------
# Stage A (UAV 5 cm) SSM/Mamba training schedule — 120k iterations.
#
# Reconstructed verbatim from the dumped run configs of
#   work_dirs/Stage_A/maskrcnn_vmamba_s_uav5cm/20260427_032646/vis_data/config.py
#   work_dirs/Stage_A/maskrcnn_efficientvmamba_b_uav5cm/20260429_105056/vis_data/config.py
# (both identical), so the restored file is the true training record for the
# 12 Stage A SSM configs that inherit it (vmamba t/s, spatialmamba t/s,
# groupmamba t/s, efficientvmamba s/b, mambaout t/s, mambavision t/s).
#
# Difference from the 80k CNN/Transformer schedule (schedule_standard_80k.py):
#   * 120k iterations (vs 80k) — the longer SSM schedule from the locked
#     protocol;
#   * 1500-iter linear warmup -> cosine to 1e-6 over the remaining 118.5k
#     (vs 1000-iter warmup / 79k cosine);
#   * weight-decay disabled additionally on the SSM selective-scan
#     parameters A_log, D, dt_proj (these have no analogue in the CNN/
#     Transformer schedule).
# All other terms are shared: AdamW lr 1e-4, wd 0.05, backbone lr_mult 0.1,
# AMP fp16, accumulative_counts 2 (effective batch 4), grad-norm clip 1.0.
#
# NOTE: consumed only by the training loop. Runner.test() does not read
# optim_wrapper, param_scheduler, or train_cfg, so restoring this file
# changes no evaluation result; it only lets the 12 SSM configs parse.
# ==========================================================================

# --- Optimizer (AMP fp16, AdamW, backbone 0.1x LR, SSM wd-exclusions) -----
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
            # SSM / selective-scan parameters: no weight decay
            'A_log': dict(decay_mult=0.0),
            'D': dict(decay_mult=0.0),
            'dt_proj': dict(decay_mult=0.0),
        }),
)

# --- LR schedule: 1.5k linear warmup -> cosine to 1e-6 over 120k ----------
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.001, by_epoch=False, begin=0, end=1500),
    dict(type='CosineAnnealingLR',
         T_max=118500, eta_min=1e-06,
         by_epoch=False, begin=1500, end=120000),
]

# --- Loops ----------------------------------------------------------------
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=120000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# --- LR auto-scaling (disabled; effective batch fixed at 4) ---------------
auto_scale_lr = dict(base_batch_size=4, enable=False)
