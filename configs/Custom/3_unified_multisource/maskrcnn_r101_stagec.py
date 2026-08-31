# Copyright (c) Stage C Date Palm Benchmark.
# maskrcnn_r101_stagec.py  (v4)
#
# Backbone: ResNet-101 with FPN.
#
# WHAT CHANGED RELATIVE TO v2
# --------------------------------------------------------------------------
#  - The trailing `optim_wrapper = dict(optimizer=dict(lr=0.02))` override has
#    been REMOVED. Under the v2 runtime (SGD lr=0.02) it was a redundant
#    no-op; under the revised runtime (AdamW lr=1e-4) it would have merged
#    into the AdamW optimizer and wrongly set its learning rate to 0.02.
#    ResNet-101 now uses the same unified AdamW + paramwise + schedule as
#    every other Stage C backbone, matching Stage B (which trained all
#    families, including ResNet, under the unified AdamW schedule).
#  - This config does NOT redeclare custom_hooks, so it inherits the runtime
#    hook stack. That stack was previously unusable because the v2 runtime
#    passed an invalid save_last kwarg to PerSensorBestCheckpointHook; the
#    revised runtime removes that kwarg, so this config now builds correctly.
#  - work_dir set explicitly to native overlay storage (/root/work_dirs/...).
#    v3 set no work_dir, so it fell back to ./work_dirs/... relative to the
#    launch cwd; if launched without --work-dir that lands on the 9p mount
#    and checkpoints are lost on reboot (Root Cause 4.3). The in-file path
#    now matches the rest of the Stage C cohort as a backstop to the flag.
# ==========================================================================

_base_ = [
    '../_base_palm/_base_maskrcnn_palm_stagec.py',
    '../_base_palm/dataset_UAV_GE_Aerial_pooled_C.py',
    '../_base_palm/runtime_palm_stagec.py',
]

model = dict(
    backbone=dict(
        # _delete_=True is REQUIRED, not decorative. The shared base sets
        # backbone=None and neck=None so each Stage C config supplies its own;
        # without _delete_, MMEngine tries to MERGE a dict into None and the
        # config cannot be built at all. Every other Stage C config has it --
        # this one was missing both, and the omission was invisible until a
        # model was actually constructed from it.
        _delete_=True,
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet101',
        ),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
)

work_dir = '/root/work_dirs/Stage_C/maskrcnn_r101_stagec'

# optim_wrapper intentionally NOT overridden -- inherited from runtime.
