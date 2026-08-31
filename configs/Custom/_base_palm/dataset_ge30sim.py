# ==========================================================================
# _base_palm/dataset_ge30sim.py   (Stage-1 of the 30cm curriculum)
# --------------------------------------------------------------------------
# Simulated-30cm GE corpus: GE 15cm tiles downsampled 2x (INTER_AREA + mild
# PSF blur + sensor noise) to ~30cm via resample_gsd.py, then labelme2coco.
# 19,472 train / 761 val / 881 test tiles. Crowns ~17 px == real WV-3 scale.
#
# This is the LARGE, target-scale corpus that supplies the data quantity the
# 267-tile real WV-3 set cannot. Stage-1 trains SpatialMamba here (full run);
# Stage-2 fine-tunes the result on real WV-3.
#
# Multi-scale + photometric augmentation as in the WV-3 v2 dataset, so the
# Stage-1 model is robust before the Stage-2 domain transfer. Class DatePalm.
# Adjust data_root / ann_file names to your labelme2coco output.
# ==========================================================================

MACHINE = 'TITAN'
if MACHINE == 'TITAN':
    num_workers, pin_memory, prefetch_factor = 4, True, 4
elif MACHINE == 'A5000':
    num_workers, pin_memory, prefetch_factor = 2, False, 2
else:
    raise ValueError(f'Unknown MACHINE={MACHINE!r}')

data_root = '/workspace/datasets/COCO/GE_30sim/'      # <-- set to labelme2coco output
dataset_type = 'CocoDataset'
metainfo = dict(classes=('DatePalm',), palette=[(220, 20, 60)])
backend_args = None
batch_size = 2          # SSM config overrides to 1 (+ accumulate)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 1.5),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), crop_type='absolute',
         recompute_bbox=True, allow_negative_crop=False),
    dict(type='Pad', size=(512, 512), pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0), keep_empty=False),
    dict(type='PhotoMetricDistortion', brightness_delta=48,
         contrast_range=(0.4, 1.6), saturation_range=(0.4, 1.6), hue_delta=24),
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
                 ann_file='Annotations/GE_30sim_train.json',
                 data_prefix=dict(img='GE_30sim_train/'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=train_pipeline, serialize_data=True,
                 backend_args=backend_args))

val_dataloader = dict(
    batch_size=1, num_workers=num_workers, persistent_workers=False,
    pin_memory=pin_memory, pin_memory_device='cuda' if pin_memory else '',
    prefetch_factor=2, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, metainfo=metainfo, data_root=data_root,
                 ann_file='Annotations/GE_30sim_val.json',
                 data_prefix=dict(img='GE_30sim_val/'), test_mode=True,
                 pipeline=test_pipeline, serialize_data=True,
                 backend_args=backend_args))

val_evaluator = dict(type='CocoMetric',
                     ann_file=data_root + 'Annotations/GE_30sim_val.json',
                     metric=['bbox', 'segm'], format_only=False,
                     proposal_nums=(100, 300, 1000), backend_args=backend_args)

test_dataloader = dict(
    batch_size=1, num_workers=num_workers, persistent_workers=False,
    pin_memory=pin_memory, pin_memory_device='cuda' if pin_memory else '',
    prefetch_factor=2, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, metainfo=metainfo, data_root=data_root,
                 ann_file='Annotations/GE_30sim_test.json',
                 data_prefix=dict(img='GE_30sim_test/'), test_mode=True,
                 pipeline=test_pipeline, serialize_data=True,
                 backend_args=backend_args))

test_evaluator = dict(type='CocoMetric',
                      ann_file=data_root + 'Annotations/GE_30sim_test.json',
                      metric=['bbox', 'segm'], format_only=False,
                      proposal_nums=(100, 300, 1000), backend_args=backend_args)
