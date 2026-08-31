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
# No-resize test pipeline for the resampled UAV sets (UAV_15sim, UAV_30sim).
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

    # ---- WorldView-3 30 cm, 8-band multispectral (Stage D, MS arm) -------
    # Same scenes and splits as 'Sat', written as 8-band GeoTIFF instead of
    # 3-band RGB. Evaluating an MS model against 'Sat' would silently feed
    # it 3-channel imagery, so the MS arm needs its own entry. The reading
    # is done by the config's test_pipeline (LoadMultispectralImageFromFile),
    # which the MS configs set explicitly for exactly this reason.
    'SatMS': dict(
        stage='Stage_D',
        ann_file=f'{_COCO_ROOT}/Sat_30cm_MS/Annotations/test_ms.json',
        img_prefix=f'{_COCO_ROOT}/Sat_30cm_MS/test_ms/',
        label='test_ms',
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
    # ---- GE resampled to 30 cm (no-resize; §4.6/§5.2) ---------------------
    # Resample of the GE TEST split (resample_gsd.py, target 0.30 m, PSF blur
    # sigma 0.6 native px, sensor noise 3 DN). The no-resize pipeline is
    # MANDATORY: the standard 1024 px pipeline would enlarge the tiles,
    # restore the source apparent crown size, and erase the resolution effect.
    #
    # TILE SIZE IS NOT UNIFORM. The resampler derives each tile's output size
    # from its own georeferenced GSD, and the GE source is a per-scene mixture:
    #   512 px (n=362)  source 0.1500 m   Mercator metres, zoom 20
    #   510 px (n=269)  source 0.1494 m   Mercator metres, zoom 20
    #   474 px (n= 55)  source 0.1389 m   UTM 40N, true ground metres
    #   461 px (n=105)  source 0.1351 m   UTM 40N, true ground metres
    #   255 px (n= 90)  source 0.0746 m   RAK_1 only -- zoom 21, twice as fine
    # The 0.150 and 0.135 groups are the SAME ground resolution (~0.1355 m)
    # expressed in different projections: a Mercator metre at 25 N is
    # 1/cos(lat) ~ 1.10x a ground metre. Only RAK_1 differs physically.
    #
    # Consequences: none of these are /32 multiples, so the preprocessor DOES
    # pad; and because sizes differ between tiles, evaluation must run with
    # batch size 1 (the resampler logs this at generation time) so tiles are
    # not padded to an arbitrary batch maximum.
    # NOTE: file_name in the JSON already carries the 'JPEGImages/' prefix,
    # so img_prefix is the PARENT directory (verified 10 Jul 2026).
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
