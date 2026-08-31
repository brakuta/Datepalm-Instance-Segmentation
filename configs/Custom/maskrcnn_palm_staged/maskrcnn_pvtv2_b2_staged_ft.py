# ==========================================================================
# maskrcnn_palm_staged/maskrcnn_pvtv2_b2_staged_ft.py   (Stage D, v1)
# --------------------------------------------------------------------------
# WV-3 30 cm fine-tuning config. Backbone/neck VERBATIM from the Stage C
# counterpart (architecture untouched). Initialisation and annotation budget
# are injected at LAUNCH via --cfg-options (see tools_staged/
# run_staged_matrix.sh):
#   stagec arm : --cfg-options load_from=<Stage C checkpoint>
#   imagenet arm: omit load_from (backbone init_cfg supplies ImageNet)
#   budget     : --cfg-options train_dataloader.dataset.ann_file=...
# Portable. Relative checkpoint path:
#   ls checkpoints/pvtv2_b2_backbone_only.pth
# FPN neck verbatim (no add_extra_convs -- P6 via max-pool).
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
        'mmdet.models.backbones.pvt',
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
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/pvtv2_b2_backbone_only.pth',
        ),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5,
        # add_extra_convs intentionally absent (verbatim Stage A/B).
    ),
)

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/maskrcnn_pvtv2_b2_staged_ft'
