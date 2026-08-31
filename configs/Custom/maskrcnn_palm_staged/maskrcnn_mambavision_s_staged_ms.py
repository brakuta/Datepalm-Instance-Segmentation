# ==========================================================================
# maskrcnn_mambavision_s_staged_ms.py   (Stage D, MULTISPECTRAL arm)
# --------------------------------------------------------------------------
# WorldView-3 30 cm, 8 bands (R, G, B, Coastal, Yellow, RedEdge, NIR1, NIR2), full training from ImageNet.
#
# Architecture, schedule, samplers, ignore regions and detection caps are
# inherited VERBATIM from maskrcnn_mambavision_s_staged_full.py, so the MS and RGB
# arms differ in spectral content and nothing else that could be helped.
# Only what multispectral input forces is overridden here.
#
# BEFORE RUNNING -- both steps, or the arm is not what it claims to be:
#
# 1. Inflate the stem. A 3-channel ImageNet conv cannot accept 8 channels.
#    MMEngine reports a size mismatch, SKIPS the tensor, and leaves the stem
#    randomly initialised while training proceeds and the log looks normal.
#    The MS arm would then be handicapped against RGB for a reason invisible
#    in the metrics.
#      MambaVision loads through timm, not a file this repo owns. Run
#      inflate_stem_to_nband.py --list against the cached timm checkpoint to
#      find the stem key, inflate it, and point the backbone at the result --
#      or accept a randomly initialised stem and SAY SO in Methods. Do not
#      leave it ambiguous: this is the one backbone where the MS arm could
#      lose for a reason that has nothing to do with spectral content.
#
# 2. Measure the normalisation. NIR has no ImageNet counterpart, so the
#    placeholder in ms_pipelines.py is a stand-in, not a measurement:
#      python configs/Custom/tools_staged/compute_band_stats.py \
#          --images /workspace/datasets/COCO/Sat_30cm_MS/train_ms/JPEGImages \
#          --ignore-zero
#
# THE BACKBONE ARGUMENT NAME DIFFERS PER FAMILY (in_chans). Verify the built
# config reports the right input width before launching -- an ignored kwarg
# would leave a 3-channel stem silently consuming the first three bands.
# ==========================================================================

_base_ = ['./maskrcnn_mambavision_s_staged_full.py']

from configs.Custom._base_palm.ms_pipelines import (  # noqa: E402
    N_BANDS, MS_MEAN, MS_STD,
    ms_train_dataloader, ms_val_dataloader, ms_test_dataloader,
    ms_val_evaluator, ms_test_evaluator, ms_test_pipeline)

custom_imports = dict(
    imports=[
        'configs.Custom._base_palm.benchmark_logging_hook',
        'configs.Custom._base_palm.nms_fp32_guard',
        'configs.Custom._base_palm.loading_multispectral',
        'configs.Custom._base_palm.ms_data_preprocessor',
        'mmdet.models.backbones.mambavision_backbone',
    ],
    allow_failed_imports=False,
)

# Dataloaders REPLACE the RGB ones inherited from the base.
# Evaluation rebuilds the test dataset from cfg.test_pipeline; without
# this the 3-channel RGB pipeline from the base config would be used.
test_pipeline = ms_test_pipeline

train_dataloader = ms_train_dataloader
val_dataloader = ms_val_dataloader
test_dataloader = ms_test_dataloader
val_evaluator = ms_val_evaluator
test_evaluator = ms_test_evaluator

model = dict(backbone=dict(in_chans=N_BANDS))

# bgr_to_rgb MUST be False: with N-band input the channel reversal would
# scramble the first three channels away from the statistics describing them.
model['data_preprocessor'] = dict(
    type='MultispectralDetDataPreprocessor',
    mean=MS_MEAN, std=MS_STD, bgr_to_rgb=False,
    pad_mask=True, pad_size_divisor=32)

work_dir = '/workspace/mmdetection/work_dirs/Stage_D/maskrcnn_mambavision_s_staged_ms'
