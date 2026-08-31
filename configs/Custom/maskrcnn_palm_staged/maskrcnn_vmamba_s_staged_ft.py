# ==========================================================================
# maskrcnn_palm_staged/maskrcnn_vmamba_s_staged_ft.py   (Stage D, v1)
# --------------------------------------------------------------------------
# WV-3 30 cm fine-tuning config. Backbone/neck VERBATIM from the Stage C
# counterpart (architecture untouched). Initialisation and annotation budget
# are injected at LAUNCH via --cfg-options (see tools_staged/
# run_staged_matrix.sh):
#   stagec arm : --cfg-options load_from=<Stage C checkpoint>
#   imagenet arm: omit load_from (backbone init_cfg supplies ImageNet)
#   budget     : --cfg-options train_dataloader.dataset.ann_file=...
# TITAN RTX only. SSM memory fix carried from Stage C.
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
        'mmdet.models.backbones.vmamba_backbone',
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
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained='/workspace/mmdetection/checkpoints/vmamba/vmamba_small_imagenet1k.pth',
        dims=96,
        depths=(2, 2, 15, 2),
        ssm_d_state=1,
        ssm_dt_rank='auto',
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type='v05_noz',
        mlp_ratio=4.0,
        downsample_version='v3',
        patchembed_version='v2',
        drop_path_rate=0.3,
        norm_layer='ln2d',
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

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/maskrcnn_vmamba_s_staged_ft'

# SSM memory fix (Stage C lineage): per-GPU batch 1 + accumulate 4
# -> effective batch 4. Launch with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
train_dataloader = dict(batch_size=1)
optim_wrapper = dict(accumulative_counts=4)
