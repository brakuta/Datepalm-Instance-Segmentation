# ==========================================================================
# maskrcnn_palm_staged/maskrcnn_mambaout_s_staged_ft.py   (Stage D, v1)
# --------------------------------------------------------------------------
# WV-3 30 cm fine-tuning config. Backbone/neck VERBATIM from the Stage C
# counterpart (architecture untouched). Initialisation and annotation budget
# are injected at LAUNCH via --cfg-options (see tools_staged/
# run_staged_matrix.sh):
#   stagec arm : --cfg-options load_from=<Stage C checkpoint>
#   imagenet arm: omit load_from (backbone init_cfg supplies ImageNet)
#   budget     : --cfg-options train_dataloader.dataset.ann_file=...
# Portable (gated-CNN SSM-ablation arm; no batch fix).
# Pretrained via timm cache (populated in Stage A; offline-safe).
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
        'mmdet.models.backbones.mambaout_backbone',
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
        type='MambaOutBackbone',
        variant='small',
        pretrained=True,
        init_cfg=None,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 576],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/maskrcnn_mambaout_s_staged_ft'
