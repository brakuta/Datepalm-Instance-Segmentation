# ==========================================================================
# _base_palm/dataset_sat_30cm_staged.py   (Stage D, v2)
# --------------------------------------------------------------------------
# WorldView-3 30 cm single-source dataset. 3,636 train / 407 val / 413 test
# tiles, 512x512, class DatePalm; 63,946 distinct reference crowns.
# Rebuilt 4 Aug 2026 -- see configs/Custom/maskrcnn_palm_staged/STAGE_D_README.md
# section 5 for provenance and the per-split annotation counts.
#
# v2 (training-only changes; val/test pipelines unchanged so the
# budget/initialisation comparison stays unconfounded):
#   * Multi-scale RandomResize (0.8-1.6x of 512, keep_ratio) + RandomCrop to
#     a fixed 512 canvas -> presents crowns at varied pixel size (small-object
#     recall) and regularises overfitting.
#   * Wider PhotoMetricDistortion (illumination/colour invariance -> fewer FPs
#     on shadowed look-alikes).
# Budget override unchanged:
#   --cfg-options train_dataloader.dataset.ann_file=Annotations/train_sat_b25.json
# NOTE: configs that wrap the dataset (e.g. CopyPaste via MultiImageMixDataset)
# re-declare train_dataloader and supersede this train pipeline; they still
# inherit the val/test loaders + evaluators below.
# ==========================================================================

MACHINE = 'TITAN'
if MACHINE == 'TITAN':
    num_workers, pin_memory, prefetch_factor = 2, True, 4
elif MACHINE == 'A5000':
    num_workers, pin_memory, prefetch_factor = 1, False, 2
else:
    raise ValueError(f'Unknown MACHINE={MACHINE!r}')

data_root = '/workspace/datasets/COCO/Sat_30cm/'
dataset_type = 'CocoDataset'
metainfo = dict(classes=('DatePalm',), palette=[(220, 20, 60)])
backend_args = None
batch_size = 2          # heavy SSM configs override to 1 (+ accumulate 4)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),
    # Scale jitter, deliberately ASYMMETRIC about 1.0.
    # A 6 m crown at 30 cm is ~20 px; the WV-3 reference spans roughly 12-28 px.
    # Symmetric jitter spends half its range pushing crowns DOWN, and below
    # ~10 px a 28x28 mask target carries almost no shape -- the model is asked
    # to learn from a smear. Biasing upward gives the small-object regime more
    # resolved instances to learn from, which is the binding constraint at
    # 30 cm, while 0.8 still supplies enough scale-down for invariance.
    # 0.8-1.6 x 512 -> 410-819 px, cropped back to a 512 canvas.
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.8, 1.6),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), crop_type='absolute',
         recompute_bbox=True, allow_negative_crop=False),
    dict(type='Pad', size=(512, 512), pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion', brightness_delta=48,
         contrast_range=(0.4, 1.6), saturation_range=(0.4, 1.6), hue_delta=24),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0), keep_empty=True),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='Pad', size=(512, 512), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor')),
]

train_dataloader = dict(
    batch_size=batch_size, num_workers=num_workers, persistent_workers=True,
    pin_memory=pin_memory, pin_memory_device='cuda' if pin_memory else '',
    prefetch_factor=prefetch_factor,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type=dataset_type, metainfo=metainfo, data_root=data_root,
                 ann_file='Annotations/train_sat.json',
                 data_prefix=dict(img='train_sat/'),
                 # filter_empty_gt=False: the tiler deliberately writes
                 # palm-free tiles into the TRAIN split as background
                 # supervision (KEEP_EMPTY_TILES/MAX_EMPTY_FRACTION). The
                 # default True discards every one of them here, silently, and
                 # the only symptom is a lower image count in the training log.
                 filter_cfg=dict(filter_empty_gt=False, min_size=32),
                 pipeline=train_pipeline, serialize_data=True,
                 backend_args=backend_args))

val_dataloader = dict(
    batch_size=1, num_workers=num_workers, persistent_workers=False,
    pin_memory=pin_memory, pin_memory_device='cuda' if pin_memory else '',
    prefetch_factor=2, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, metainfo=metainfo, data_root=data_root,
                 ann_file='Annotations/val_sat.json',
                 data_prefix=dict(img='val_sat/'), test_mode=True,
                 pipeline=test_pipeline, serialize_data=True,
                 backend_args=backend_args))

val_evaluator = dict(type='CocoMetric',
                     ann_file=data_root + 'Annotations/val_sat.json',
                     metric=['bbox', 'segm'], format_only=False,
                     proposal_nums=(100, 300, 1000), backend_args=backend_args)

test_dataloader = dict(
    batch_size=1, num_workers=num_workers, persistent_workers=False,
    pin_memory=pin_memory, pin_memory_device='cuda' if pin_memory else '',
    prefetch_factor=2, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, metainfo=metainfo, data_root=data_root,
                 ann_file='Annotations/test_sat.json',
                 data_prefix=dict(img='test_sat/'), test_mode=True,
                 pipeline=test_pipeline, serialize_data=True,
                 backend_args=backend_args))

test_evaluator = dict(type='CocoMetric',
                      ann_file=data_root + 'Annotations/test_sat.json',
                      metric=['bbox', 'segm'], format_only=False,
                      proposal_nums=(100, 300, 1000), backend_args=backend_args)
