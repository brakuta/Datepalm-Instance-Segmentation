#!/usr/bin/env python3
# ==========================================================================
# add_ge30sim.py  --  one-shot, idempotent patch
# --------------------------------------------------------------------------
# Adds the GE_30sim sensor (GE 15 cm TEST split resampled to 30 cm) to:
#   1. sensor_registry.py   -- new SENSORS entry, no-resize pipeline
#   2. cross_transfer.json  -- new column in BOTH rows of the transfer matrix
#
# Both files are backed up (.bak) before modification. Re-running is safe:
# each edit is skipped if already present.
#
# Usage (from configs/Custom/Evaluation/):
#   python add_ge30sim.py                      # apply
#   python add_ge30sim.py --manifest my.json   # non-default manifest name
#   python add_ge30sim.py --dry-run            # show what would change
# ==========================================================================

import argparse
import json
import os.path as osp
import shutil
import sys

REGISTRY = 'sensor_registry.py'

# --------------------------------------------------------------------------
# 1. Registry entry, inserted immediately before the closing brace of SENSORS.
# --------------------------------------------------------------------------
ANCHOR = """        label='test_UAV_30cm',
        test_pipeline=TEST_PIPELINE_NORESIZE,
    ),
}"""

ENTRY = """        label='test_UAV_30cm',
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
}"""

# Header comment fix: the no-resize pipeline now serves GE_30sim too.
HDR_OLD = ("# No-resize test pipeline for the resampled UAV sets "
           "(UAV_15sim, UAV_30sim).")
HDR_NEW = ("# No-resize test pipeline for the resampled sets\n"
           "# (UAV_15sim, UAV_30sim, GE_30sim).")

# --------------------------------------------------------------------------
# 2. Manifest matrix: GE_30sim added to both training sources.
#    - GE_Aerial: controlled degradation of the IN-DOMAIN source (the key cell)
#    - UAV:       completes the matrix (combined cross-sensor + cross-resolution)
#    Inserted after 'GE' so the column order matches the figure's DOMAIN_ORDER.
# --------------------------------------------------------------------------
NEW_SENSOR = 'GE_30sim'


def patch_registry(dry: bool) -> bool:
    if not osp.isfile(REGISTRY):
        sys.exit(f'[ERROR] {REGISTRY} not found. Run from configs/Custom/Evaluation/.')
    src = open(REGISTRY).read()

    if "'GE_30sim'" in src:
        print(f'[skip] {REGISTRY}: GE_30sim already declared.')
        return False
    if ANCHOR not in src:
        sys.exit(f'[ERROR] {REGISTRY}: could not find the UAV_30sim block to '
                 f'anchor the insertion. Add the entry manually.')

    out = src.replace(ANCHOR, ENTRY, 1)
    if HDR_OLD in out:
        out = out.replace(HDR_OLD, HDR_NEW, 1)

    if dry:
        print(f'[dry-run] {REGISTRY}: would insert GE_30sim entry '
              f'(+{len(out.splitlines()) - len(src.splitlines())} lines).')
        return True
    shutil.copy(REGISTRY, REGISTRY + '.bak')
    open(REGISTRY, 'w').write(out)
    print(f'[ok] {REGISTRY}: GE_30sim entry inserted (backup -> {REGISTRY}.bak).')
    return True


def patch_manifest(path: str, dry: bool) -> bool:
    if not osp.isfile(path):
        sys.exit(f'[ERROR] manifest {path} not found. Pass --manifest PATH.')
    m = json.load(open(path))
    matrix = m.get('matrix')
    if not matrix:
        sys.exit(f'[ERROR] {path}: no "matrix" key. This script expects an '
                 f'explicit matrix; add one, or edit DEFAULT_MATRIX in '
                 f'run_cross_transfer.py instead.')

    changed = False
    for src_key, sensors in matrix.items():
        if NEW_SENSOR in sensors:
            print(f'[skip] {path}: matrix["{src_key}"] already has {NEW_SENSOR}.')
            continue
        if 'GE' in sensors:                       # keep figure column order
            sensors.insert(sensors.index('GE') + 1, NEW_SENSOR)
        else:
            sensors.append(NEW_SENSOR)
        changed = True
        print(f'[{"dry-run" if dry else "ok"}] {path}: matrix["{src_key}"] -> {sensors}')

    if changed and not dry:
        shutil.copy(path, path + '.bak')
        with open(path, 'w') as f:
            json.dump(m, f, indent=2)
        print(f'[ok] {path}: matrix updated (backup -> {path}.bak).')
    return changed


def verify() -> None:
    """Re-imports the registry and checks the entry resolves on disk."""
    sys.path.insert(0, osp.dirname(osp.abspath(REGISTRY)))
    for mod in ('sensor_registry',):
        sys.modules.pop(mod, None)
    from sensor_registry import get_sensor          # noqa: E402
    e = get_sensor('GE_30sim')
    ok_ann = osp.isfile(e['ann_file'])
    print(f'  ann_file exists            : {ok_ann}')
    if not ok_ann:
        return
    d = json.load(open(e['ann_file']))
    img0 = d['images'][0]
    p = osp.join(e['img_prefix'], img0['file_name'])
    print(f'  first image resolves       : {osp.isfile(p)}')
    print(f'  no-resize pipeline attached: {"test_pipeline" in e}')
    print(f'  images / annotations       : {len(d["images"])} / {len(d["annotations"])}')
    print(f'  tile size                  : {img0["width"]} x {img0["height"]} '
          f'(pads to {-(-img0["width"] // 32) * 32})')


def main() -> None:
    ap = argparse.ArgumentParser(description='Add GE_30sim to the transfer matrix.')
    ap.add_argument('--manifest', default='cross_transfer.json')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-manifest', action='store_true',
                    help='patch only sensor_registry.py')
    a = ap.parse_args()

    patch_registry(a.dry_run)
    if not a.skip_manifest:
        patch_manifest(a.manifest, a.dry_run)

    if not a.dry_run:
        print('\n[verify] resolving GE_30sim against disk:')
        verify()
        print('\n[next]\n'
              f'  python run_cross_transfer.py --manifest {a.manifest}\n'
              f'  python compile_cross_transfer.py --manifest {a.manifest}')


if __name__ == '__main__':
    main()
