# ==========================================================================
# configs/Custom/Evaluation/sensor_registry.py
# --------------------------------------------------------------------------
# CENTRAL SENSOR REGISTRY for the date palm detection benchmark.
#
# Single source of truth for every evaluation test set, across every
# stage and sensor. Every evaluation script in this folder reads its
# test-set paths from here and from nowhere else.
#
# WHY THIS FILE EXISTS
# --------------------
# The evaluation pipeline is source-agnostic: the runner and the
# compiler contain NO hardcoded dataset paths. All such paths live here.
# Consequences:
#   * adding a new sensor (any GSD, any stage) is a single edit here;
#   * correcting a path propagates to the runner and compiler at once;
#   * the same scripts evaluate UAV, GE, Aerial, Satellite, or any
#     future multiscale source with no code change.
#
# --------------------------------------------------------------------------
# HOW TO ADD OR EDIT A SENSOR
# --------------------------------------------------------------------------
# Each entry in SENSORS maps a short sensor key (used on the command
# line, e.g. --sensors GE Aerial) to a dictionary with four required
# fields and one optional field:
#
#   stage          : free-text stage label, for grouping/reporting only.
#   ann_file       : ABSOLUTE path to the COCO-format test annotation JSON.
#   img_prefix     : ABSOLUTE path to the directory holding the test images.
#   label          : the partition tag used in output filenames, e.g.
#                    'test_GE' -> <config_stem>__test_GE.csv
#   test_pipeline  : (OPTIONAL) an MMDetection test pipeline (list of dict).
#                    When present, the cross-transfer runner injects it in
#                    place of the config's default test pipeline for this
#                    sensor. Used only by the resampled UAV sets, which must
#                    be evaluated at their TRUE coarse pixel size (no resize).
#                    evaluate_model.py reads only the four required fields and
#                    ignores this key, so it is unaffected.
#
# To add a future source, copy any block below, change the fields, and
# pass its key via --sensors. No other file needs to change.
#
# IMPORTANT -- paths are NOT validated here. Each script checks at run
# time that ann_file and img_prefix exist, and skips a sensor with a
# clear message if they do not. This file only declares them.
#
# --------------------------------------------------------------------------
# DATASET ROOTS (confirmed present under /workspace/datasets/COCO/):
#   UAV_5cm      train_UAV    / val_UAV    / test_UAV
#                test_UAV_15cm / test_UAV_30cm      (controlled-GSD resamples)
#                Annotations/  test_UAV.json, test_UAV_15cm.json,
#                              test_UAV_30cm.json, ...
#   GE_15cm      train_GE     / val_GE     / test_GE
#   Aerial_15cm  train_aerial / val_aerial / test_aerial
#   Sat_30cm     train_sat    / val_sat    / test_sat
# ==========================================================================

from typing import Dict, List

# --------------------------------------------------------------------------
# Common dataset root. Edit once if the datasets move (e.g. to native
# ext4); every entry below is derived from it.
# --------------------------------------------------------------------------
_COCO_ROOT = '/workspace/datasets/COCO'

# --------------------------------------------------------------------------
# No-resize test pipeline for the resampled sets
# (UAV_15sim, UAV_30sim, GE_30sim).
# --------------------------------------------------------------------------
# The resampled tiles are written at their TRUE coarse pixel size (~347 px at
# 15 cm, ~174 px at 30 cm). The standard test pipeline resizes to 1024 px,
# which would ENLARGE these tiles, restore the original apparent crown size,
# and erase the resolution effect being measured. This pipeline omits Resize
# and fixed Pad: the model's DetDataPreprocessor pads only to a /32 multiple
# (e.g. 347 -> 352), so crowns reach the network at their true coarse scale.
# Native sensors (UAV, GE, Aerial, Sat) do NOT use this; they keep the
# config's standard 1024 px pipeline.
TEST_PIPELINE_NORESIZE = [
    dict(type='LoadImageFromFile', backend_args=None),
    # Unit-scale Resize: geometrically a no-op (scale_factor=1.0 keeps the tile
    # at its true coarse size), but it POPULATES the 'scale_factor' meta key the
    # Mask R-CNN mask head requires at predict_by_feat. Without it,
    # FCNMaskHead._predict_by_feat_single raises KeyError('scale_factor').
    # It does NOT enlarge the tile; the coarse pixel scale is preserved.
    dict(type='Resize', scale_factor=1.0, keep_ratio=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor')),
]

# --------------------------------------------------------------------------
# Sensor table.
# --------------------------------------------------------------------------
SENSORS: Dict[str, dict] = {

    # ---- UAV 5 cm (Stage A) ----------------------------------------------
    'UAV': dict(
        stage='Stage_A',
        ann_file=f'{_COCO_ROOT}/UAV_5cm/Annotations/test_UAV.json',
        img_prefix=f'{_COCO_ROOT}/UAV_5cm/test_UAV/',
        label='test_UAV',
    ),

    # ---- Google Earth 15 cm (Stage B) ------------------------------------
    # NOTE: the GE_15cm/Annotations directory must contain test_GE.json.
    # If only a misnamed 'test_GE' entry exists, rename it to test_GE.json
    # before running GE evaluation.
    'GE': dict(
        stage='Stage_B',
        ann_file=f'{_COCO_ROOT}/GE_15cm/Annotations/test_GE.json',
        img_prefix=f'{_COCO_ROOT}/GE_15cm/test_GE/',
        label='test_GE',
    ),

    # ---- Aerial 15 cm (Stage B) ------------------------------------------
    'Aerial': dict(
        stage='Stage_B',
        ann_file=f'{_COCO_ROOT}/Aerial_15cm/Annotations/test_aerial.json',
        img_prefix=f'{_COCO_ROOT}/Aerial_15cm/test_aerial/',
        label='test_aerial',
    ),

    # ---- WorldView-3 satellite 30 cm (Stage D) ---------------------------
    'Sat': dict(
        stage='Stage_D',
        ann_file=f'{_COCO_ROOT}/Sat_30cm/Annotations/test_sat.json',
        img_prefix=f'{_COCO_ROOT}/Sat_30cm/test_sat/',
        label='test_sat',
    ),

    # ======================================================================
    # Controlled-GSD resampled UAV sets (cross-resolution transfer; §4.6/§5.2)
    # ----------------------------------------------------------------------
    # Source models are the Stage A UAV-trained models, evaluated zero-shot.
    # These two entries carry the optional 'test_pipeline' key so the
    # cross-transfer runner evaluates them at their true coarse pixel size.
    # They live under UAV_5cm because they are resamples of the UAV test set.
    # ======================================================================

    # ---- UAV resampled to 15 cm (no-resize) ------------------------------
    'UAV_15sim': dict(
        stage='Stage_A',
        ann_file=f'{_COCO_ROOT}/UAV_5cm/Annotations/test_UAV_15cm.json',
        img_prefix=f'{_COCO_ROOT}/UAV_5cm/test_UAV_15cm/',
        label='test_UAV_15cm',
        test_pipeline=TEST_PIPELINE_NORESIZE,
    ),

    # ---- UAV resampled to 30 cm (no-resize) ------------------------------
    'UAV_30sim': dict(
        stage='Stage_A',
        ann_file=f'{_COCO_ROOT}/UAV_5cm/Annotations/test_UAV_30cm.json',
        img_prefix=f'{_COCO_ROOT}/UAV_5cm/test_UAV_30cm/',
        label='test_UAV_30cm',
        test_pipeline=TEST_PIPELINE_NORESIZE,
    ),

    # ---- GE resampled to 30 cm (no-resize; cross-resolution transfer) ----
    # Resample of the GE 15 cm TEST split (resample_gsd.py, 0.15 -> 0.30 m,
    # --psf-blur 0.6 --sensor-noise 3). Tiles are 474 x 474 at their TRUE
    # 30 cm pixel size (GE test tiles are 948 px at 15 cm; crowns ~17 px,
    # the WorldView-3 scale). Zero-shot only: this is a resample of the TEST
    # split, so no training data is touched.
    #
    # The no-resize pipeline is MANDATORY. The config's standard 1024 px
    # pipeline would enlarge these tiles ~2.2x, restore the 15 cm apparent
    # crown size, and erase the resolution effect being measured.
    #
    # 474 is not a multiple of 32, so DetDataPreprocessor pads to 480 --
    # identical treatment to UAV_15sim (347 -> 352) and UAV_30sim.
    # Variable tile size implies batch_size_noresize=1 and cudnn_benchmark
    # off; the cross-transfer runner already selects both from the presence
    # of the 'test_pipeline' key.
    #
    # NOTE: file_name in GE_30sim_test.json already carries the 'JPEGImages/'
    # prefix, so img_prefix is the PARENT directory (verified 10 Jul 2026:
    # 881 images, 73,589 annotations).
    'GE_30sim': dict(
        stage='Stage_B',
        ann_file=f'{_COCO_ROOT}/GE_30sim/Annotations/GE_30sim_test.json',
        img_prefix=f'{_COCO_ROOT}/GE_30sim/GE_30sim_test/',
        label='test_GE_30sim',
        test_pipeline=TEST_PIPELINE_NORESIZE,
    ),
}


# --------------------------------------------------------------------------
# Accessors -- used by the evaluation scripts; do not edit.
# --------------------------------------------------------------------------
def available_sensors() -> List[str]:
    """Returns the list of sensor keys currently declared in SENSORS."""
    return list(SENSORS.keys())


def get_sensor(key: str) -> dict:
    """Returns the registry entry for one sensor key.

    Raises KeyError with an informative message listing the valid keys
    if the requested sensor is not declared.
    """
    if key not in SENSORS:
        raise KeyError(
            f'sensor "{key}" is not declared in sensor_registry.py. '
            f'Declared sensors: {available_sensors()}. '
            f'Add it to the SENSORS table to use it.')
    return SENSORS[key]


def sensors_for_stage(stage: str) -> List[str]:
    """Returns all sensor keys belonging to a given stage label."""
    return [k for k, v in SENSORS.items() if v.get('stage') == stage]
