# ==========================================================================
# maskrcnn_palm_staged/maskrcnn_efficientvmamba_b_staged_ft.py   (Stage D, v1)
# --------------------------------------------------------------------------
# WV-3 30 cm fine-tuning config. Backbone/neck VERBATIM from the Stage C
# counterpart (architecture untouched). Initialisation and annotation budget
# are injected at LAUNCH via --cfg-options (see tools_staged/
# run_staged_matrix.sh):
#   stagec arm : --cfg-options load_from=<Stage C checkpoint>
#   imagenet arm: omit load_from (backbone init_cfg supplies ImageNet)
#   budget     : --cfg-options train_dataloader.dataset.ann_file=...
# TITAN RTX only. SSM memory fix carried from Stage C.
# Checkpoint path corrected to the TITAN mount '/workspace/mmdetection/...'
# (the obsolete A5000-style '/workspace/project/...' prefix was removed):
#   ls -lh /workspace/mmdetection/checkpoints/efficientvmamba/
# ==========================================================================

_base_ = [
    '../_base_palm/_base_maskrcnn_palm_stagec.py',
    '../_base_palm/dataset_sat_30cm_staged.py',
    '../_base_palm/schedule_staged_ft.py',
    '../_base_palm/runtime_palm_staged.py',
]

custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.nms_fp32_guard',
        'mmdet.models.backbones.efficientvmamba_backbone',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/segm_mAP_50',
        rule='greater',
        min_delta=0.001,
        patience=8,
    ),
    dict(
        type='PalmBenchmarkLoggingHook',
        input_shape=(3, 1024, 1024),
        compute_flops=True,
        save_json=True,
    ),
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_EFFVSSM',
        out_indices=(0, 1, 2, 3),
        pretrained='/workspace/mmdetection/checkpoints/efficientvmamba/efficient_vmamba_base.ckpt',
        depths=(2, 2, 9, 2),
        dims=96,
        ssm_d_state=16,
        ssm_dt_rank='auto',
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version='v1',
        patchembed_version='v1',
        window_size=2,
        drop_path_rate=0.2,
        frozen_stages=0,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/maskrcnn_efficientvmamba_b_staged_ft'

# SSM memory fix (Stage C lineage): per-GPU batch 1 + accumulate 4
# -> effective batch 4. Launch with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
train_dataloader = dict(batch_size=1)
optim_wrapper = dict(accumulative_counts=4)
