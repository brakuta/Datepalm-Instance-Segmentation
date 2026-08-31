# ==========================================================================
# _base_palm/schedule_staged_full.py   (Stage D, v4 -- arm B0)
# --------------------------------------------------------------------------
# FULL TRAINING on the real WorldView-3 30 cm set, from ImageNet weights.
# This is the ImageNet baseline for Stage D, and it is deliberately NOT the
# locked fine-tune recipe.
#
# WHY A SEPARATE SCHEDULE
#   schedule_staged_ft.py is built for a near-domain prior: stem + stages 0-1
#   frozen, head LR 2e-5, backbone lr_mult 0.01. Those settings assume the
#   backbone already speaks the target domain and only needs nudging. Applied
#   to ImageNet weights they underfit badly, and beating an underfit baseline
#   proves nothing about the value of the Stage C prior. Arm B0 therefore gets
#   a genuine training run: nothing frozen, Stage-C-style LR, long schedule,
#   EarlyStopping to find the real cutoff.
#
#   The consequence is that arms B0 and C differ in BOTH initialisation and
#   recipe. That is intentional and must be stated in Methods: each arm uses
#   the recipe appropriate to its initialisation, and both are trained to
#   convergence. The claim the design supports is therefore
#       "initialising from the Stage C cross-resolution model reaches higher
#        WV-3 accuracy than the best result obtainable from ImageNet at this
#        data size, at a fraction of the training cost"
#   -- an operational statement -- and NOT "prior X beats prior Y under a
#   matched recipe". Do not write the second sentence from these runs.
#
# LR / SCHEDULE
#   Mirrors schedule_ge30sim_stage1.py (lr 1e-4, backbone lr_mult 0.1), which
#   is the project's established full-training setting, rather than inventing
#   a new one for this arm.
#
# PRECISION
#   fp32 OptimWrapper, not AmpOptimWrapper. The ge30sim stage-1 runs hit
#   NaN overflow under fp16 and the v3 fine-tune schedule moved to fp32 for
#   that reason; arm B0 inherits the same decision so a multi-hour run cannot
#   die at iteration 30k.
#
# ITERATION BUDGET -- VERIFY BEFORE LAUNCH
#   Sized for the refined WV-3 train split quoted in schedule_staged_ft.py
#   (2,413 tiles -> ~1,207 iter/epoch at batch 2, so 60k ~ 50 epochs).
#   The header of dataset_sat_30cm_staged.py still quotes the older 267/142/124
#   split. Confirm which is live before launching:
#       python -c "import json;print(len(json.load(open(
#         '/workspace/datasets/COCO/Sat_30cm/Annotations/train_sat.json'
#         ))['images']))"
#   If the split is the small one, 60k iterations is several hundred epochs
#   on a few hundred tiles and will overfit long before EarlyStopping earns
#   its keep -- drop max_iters to ~15k and re-check.
# ==========================================================================

max_iters    = 60_000
val_interval = 1_200          # same cadence as the fine-tune arm, so the two
                              # validation curves are directly comparable

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters,
                 val_interval=val_interval)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False,
         begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=max_iters - 1000, eta_min=1e-6,
         by_epoch=False, begin=1000, end=max_iters),
]

optim_wrapper = dict(
    type='OptimWrapper', accumulative_counts=2,      # effective batch 4
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
