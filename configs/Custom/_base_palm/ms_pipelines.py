# ==========================================================================
# _base_palm/ms_pipelines.py
# --------------------------------------------------------------------------
# WorldView-3 30 cm MULTISPECTRAL pipelines and dataloaders, for the MS arm.
#
# A PLAIN MODULE, NOT A CONFIG BASE. The MS model configs inherit their
# architecture from maskrcnn_<bb>_staged_full.py, which already carries the
# RGB dataset; listing an MS dataset alongside it in _base_ collides on
# train_pipeline, train_dataloader and both evaluators. Importing instead
# keeps these definitions in ONE place while the four configs override only
# the dataloaders -- the alternative was duplicating the pipelines four times
# and letting them drift.
#
# Identical to dataset_sat_30cm_staged.py in tiling, splits, geometry and
# augmentation strength. The ONLY differences are the ones multispectral
# input forces:
#
#   * LoadMultispectralImageFromFile instead of LoadImageFromFile.
#     mmcv.imread returns 3-channel BGR whatever the file holds, so the stock
#     loader would hand the model three bands and the "multispectral"
#     experiment would quietly be RGB, with nothing in the metrics to show it.
#
#   * MultispectralPhotoMetricDistortion instead of PhotoMetricDistortion.
#     The stock transform converts BGR to HSV with cv2, which refuses more
#     than three channels. Dropping it instead would give the MS arm weaker
#     augmentation than the RGB arm and confound spectral content with
#     regularisation.
#
#   * data_preprocessor mean/std of length N_BANDS, and bgr_to_rgb=False.
#     With N-band input the channel-reversal would scramble the first three
#     channels away from the statistics that describe them.
#
# BAND SELECTION -- set when tiling, recorded here
#   WorldView-3 8-band native order is
#       1 Coastal  2 Blue  3 Green  4 Yellow  5 Red  6 RedEdge  7 NIR1  8 NIR2
#   so the first three bands of the raw product are NOT R, G, B.
#
#   ORDER MATTERS FOR THE INFLATED STEM. Inflation copies the pretrained
#   ImageNet filters into the first three input channels and fills the rest
#   from their mean. Tiling in native order would therefore apply the
#   pretrained RED filter to Coastal, GREEN to Blue and BLUE to Green -- the
#   weights would be present but misassigned, which is worse than useless and
#   invisible once training starts. The job file must request
#
#       "bands": [5, 3, 2, 1, 4, 6, 7, 8]
#
#   giving channels R, G, B, Coastal, Yellow, RedEdge, NIR1, NIR2. The first
#   three then match both the inflated stem and the ImageNet entries of
#   _MEAN/_STD below; the remaining five are new channels the model learns.
#
# RUN compute_band_stats.py ON THE TRAIN SPLIT AND PASTE THE RESULT BELOW.
# The placeholders for channels 4-8 are the ImageNet red statistic, which is a
# stand-in, not a measurement -- none of those bands has an ImageNet
# counterpart, and a badly centred channel is one the network must spend
# capacity correcting.
# ==========================================================================

N_BANDS = 8

MACHINE = 'TITAN'
if MACHINE == 'TITAN':
    num_workers, pin_memory, prefetch_factor = 2, True, 4
elif MACHINE == 'A5000':
    num_workers, pin_memory, prefetch_factor = 1, False, 2
else:
    raise ValueError(f'Unknown MACHINE={MACHINE!r}')

data_root = '/workspace/datasets/COCO/Sat_30cm_MS/'
dataset_type = 'CocoDataset'
metainfo = dict(classes=('DatePalm',), palette=[(220, 20, 60)])
backend_args = None
batch_size = 2

# ---- normalisation -------------------------------------------------------
# First three entries are the ImageNet values, matching the inflated stem.
# REPLACE the trailing entries with measured values:
#   python configs/Custom/tools_staged/compute_band_stats.py \
#       --images /workspace/datasets/COCO/Sat_30cm_MS/train_ms/JPEGImages \
#       --ignore-zero
# MEASURED 4 Aug 2026 on the train_ms split, 400 tiles, 103,606,740 valid
# pixels, nodata excluded (compute_band_stats.py --ignore-zero).
#
# Channels 1-3 keep the IMAGENET values, not the measured ones (which were
# 138.6/129.5/121.9, std 69.2/68.7/74.2). Two reasons, and they point the same
# way. The inflated stem carries filters fitted under the ImageNet statistics,
# so those are the statistics they expect. And the RGB arm normalises with
# them too -- using measured values here would mean the two arms treat the
# same three channels differently, so an MS-vs-RGB difference could no longer
# be attributed to the extra bands.
#
# Channels 4-8 are measured. None of Coastal, Yellow, RedEdge, NIR1 or NIR2
# has an ImageNet counterpart, and a badly centred channel is one the network
# must spend capacity correcting.
#
# Sanity: NIR1 and NIR2 are the brightest (157.8, 159.8) -- both vegetation
# and sand are bright in the near infrared -- and Coastal is the darkest and
# narrowest (114.0, std 50.5), which is what atmospheric scattering does to
# that band. Consistent with the [5,3,2,1,4,6,7,8] ordering.
_MEAN = [123.675, 116.28, 103.53,
         113.973, 133.572, 148.779, 157.831, 159.764]
_STD = [58.395, 57.12, 57.375,
        50.482, 56.969, 51.882, 52.161, 51.726]
assert len(_MEAN) == len(_STD) == N_BANDS, (
    f'mean/std must have N_BANDS={N_BANDS} entries; re-run '
    f'compute_band_stats.py if the band selection changed')

train_pipeline = [
    dict(type='LoadMultispectralImageFromFile', expected_channels=N_BANDS),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.8, 1.6),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), crop_type='absolute',
         recompute_bbox=True, allow_negative_crop=False),
    dict(type='MultispectralPad', size=(512, 512),
         pad_val=dict(img=(114,) * N_BANDS)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultispectralPhotoMetricDistortion', brightness_delta=48,
         contrast_range=(0.4, 1.6), saturation_range=(0.4, 1.6), hue_delta=24),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0), keep_empty=True),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadMultispectralImageFromFile', expected_channels=N_BANDS),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='MultispectralPad', size=(512, 512),
         pad_val=dict(img=(114,) * N_BANDS)),
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
                 ann_file='Annotations/train_ms.json',
                 data_prefix=dict(img='train_ms/'),
                 # Background tiles are deliberate supervision -- see
                 # dataset_sat_30cm_staged.py for the full reasoning.
                 filter_cfg=dict(filter_empty_gt=False, min_size=32),
                 pipeline=train_pipeline, serialize_data=True,
                 backend_args=backend_args))

val_dataloader = dict(
    batch_size=1, num_workers=num_workers, persistent_workers=False,
    pin_memory=pin_memory, pin_memory_device='cuda' if pin_memory else '',
    prefetch_factor=2, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, metainfo=metainfo, data_root=data_root,
                 ann_file='Annotations/val_ms.json',
                 data_prefix=dict(img='val_ms/'), test_mode=True,
                 pipeline=test_pipeline, serialize_data=True,
                 backend_args=backend_args))

val_evaluator = dict(type='CocoMetric',
                     ann_file=data_root + 'Annotations/val_ms.json',
                     metric=['bbox', 'segm'], format_only=False,
                     proposal_nums=(100, 300, 1000), backend_args=backend_args)

test_dataloader = dict(
    batch_size=1, num_workers=num_workers, persistent_workers=False,
    pin_memory=pin_memory, pin_memory_device='cuda' if pin_memory else '',
    prefetch_factor=2, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, metainfo=metainfo, data_root=data_root,
                 ann_file='Annotations/test_ms.json',
                 data_prefix=dict(img='test_ms/'), test_mode=True,
                 pipeline=test_pipeline, serialize_data=True,
                 backend_args=backend_args))

test_evaluator = dict(type='CocoMetric',
                      ann_file=data_root + 'Annotations/test_ms.json',
                      metric=['bbox', 'segm'], format_only=False,
                      proposal_nums=(100, 300, 1000), backend_args=backend_args)


# --------------------------------------------------------------------------
# Exported names. The MS configs do:
#     from configs.Custom._base_palm.ms_pipelines import (
#         N_BANDS, MS_MEAN, MS_STD, ms_train_dataloader, ms_val_dataloader,
#         ms_test_dataloader, ms_val_evaluator, ms_test_evaluator)
# --------------------------------------------------------------------------
MS_MEAN = _MEAN
MS_STD = _STD
# The pipelines are exported too, not just the dataloaders. Evaluation
# (configs/Custom/Evaluation/evaluate_model.py) rebuilds the test dataset
# from cfg.test_pipeline, and an MS config that does not set it inherits
# the 3-channel RGB pipeline from dataset_sat_30cm_staged.py -- feeding
# LoadImageFromFile output to an 8-channel stem.
ms_train_pipeline = train_pipeline
ms_test_pipeline = test_pipeline
ms_train_dataloader = train_dataloader
ms_val_dataloader = val_dataloader
ms_test_dataloader = test_dataloader
ms_val_evaluator = val_evaluator
ms_test_evaluator = test_evaluator
