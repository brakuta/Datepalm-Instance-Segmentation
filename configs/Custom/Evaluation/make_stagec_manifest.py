#!/usr/bin/env python3
# ==========================================================================
# make_stagec_manifest.py  --  Stage C cross-transfer manifest (diagonal)
# --------------------------------------------------------------------------
# Adds the multi-source (Stage C) model to the cross-transfer experiment
# under the per-sensor DIAGONAL checkpoint protocol, evaluated on all six
# domains INCLUDING the simulated sets (which need the no-resize pipeline
# that only the cross-transfer runner injects).
#
# THE CHECKPOINT SPLIT
# --------------------
# Stage C saves two checkpoints per backbone (best_UAV, best_GE). The
# runner's inner loop evaluates every row on ALL sensors of matrix[src],
# with ONE checkpoint per row, so the diagonal is expressed by splitting
# each backbone into TWO rows keyed by two train_source values:
#
#   Stage_C_U  (best_UAV) -> matrix: [UAV, UAV_15sim, UAV_30sim]
#   Stage_C_G  (best_GE)  -> matrix: [GE, GE_30sim, Aerial]
#
# UAV-derived domains take best_UAV; GE-derived + aerial take best_GE.
# The six output CSVs per backbone never collide (distinct sensor labels),
# though both rows share config_stem = maskrcnn_<bb>_stagec. The two half-
# rows are merged back into one 'Stage_C' row at load time in the notebook.
#
# results_dir is written ABSOLUTE so the new CSVs land beside the existing
# 28x6 matrix regardless of the launch directory.
#
# Usage (from configs/Custom/Evaluation/):
#   python make_stagec_manifest.py \
#       --results-dir /workspace/mmdetection/results/cross_transfer \
#       --src cross_transfer.json
#   PYTHONPATH=/workspace/mmdetection:$PYTHONPATH \
#       python run_cross_transfer.py --manifest cross_transfer_stagec.json
#   PYTHONPATH=/workspace/mmdetection:$PYTHONPATH \
#       python compile_cross_transfer.py --manifest cross_transfer.json  # see note
# ==========================================================================

import argparse
import glob
import json
import os.path as osp
import sys

CFG_DIR = '/workspace/mmdetection/configs/Custom/3_unified_multisource'
WORK    = '/workspace/mmdetection/work_dirs/Stage_C'

# backbone stem -> (label, family)
MODELS = {
    'maskrcnn_r50_stagec':             ('ResNet-50',        'CNN'),
    'maskrcnn_convnext_t_stagec':      ('ConvNeXt-T',       'CNN'),
    'maskrcnn_swin_s_stagec':          ('Swin-S',           'Transformer'),
    'maskrcnn_pvtv2_b2_stagec':        ('PVTv2-B2',         'Transformer'),
    'maskrcnn_spatialmamba_s_stagec':  ('SpatialMamba-S',   'Mamba'),
    'maskrcnn_vmamba_s_stagec':        ('VMamba-S',         'Mamba'),
    'maskrcnn_mambavision_s_stagec':   ('MambaVision-S',    'Mamba'),
    'maskrcnn_mambaout_s_stagec':      ('MambaOut-S',       'Mamba'),
    'maskrcnn_groupmamba_s_stagec':    ('GroupMamba-S',     'Mamba'),
    'maskrcnn_efficientvmamba_b_stagec': ('EfficientVMamba-B', 'Mamba'),
}

# the diagonal: which train_source (checkpoint) covers which sensors
SPLIT = {
    'Stage_C_U': dict(glob='best_UAV_*.pth',
                      sensors=['UAV', 'UAV_15sim', 'UAV_30sim']),
    'Stage_C_G': dict(glob='best_GE_*.pth',
                      sensors=['GE', 'GE_30sim', 'Aerial']),
}


def one(pattern, what):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        return None, f'{what}: expected 1 match, got {len(hits)} for {pattern}'
    return hits[0], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='cross_transfer.json',
                    help='full manifest, for protocol block (read-only)')
    ap.add_argument('--out', default='cross_transfer_stagec.json')
    ap.add_argument('--results-dir', required=True,
                    help='ABSOLUTE dir holding the existing matrix CSVs')
    a = ap.parse_args()

    if not osp.isabs(a.results_dir):
        sys.exit('[ERROR] --results-dir must be absolute.')
    if not osp.isfile(a.src):
        sys.exit(f'[ERROR] {a.src} not found (needed for the protocol block).')
    protocol = json.load(open(a.src))['protocol']

    models, problems = [], []
    for stem, (label, family) in MODELS.items():
        cfg = osp.join(CFG_DIR, f'{stem}.py')
        if not osp.isfile(cfg):
            cfg = osp.join(WORK, stem, f'{stem}.py')      # fallback to work_dir copy
        if not osp.isfile(cfg):
            problems.append(f'{label}: config missing'); continue
        for src_key, spec in SPLIT.items():
            ckpt, err = one(osp.join(WORK, stem, spec['glob']), f'{label}/{src_key}')
            if err:
                problems.append(err); continue
            models.append({
                'label': label, 'family': family, 'train_source': src_key,
                'config': cfg, 'checkpoint': ckpt, 'applied_score_thr': None,
            })

    if problems:
        print('[PROBLEM] not added:')
        for p in problems:
            print('   ', p)
    if not models:
        sys.exit('[ERROR] no Stage C rows built.')

    matrix = {k: v['sensors'] for k, v in SPLIT.items()}
    out = {
        '_comment': (
            'Stage C (multi-source) diagonal manifest. Two rows per backbone: '
            'Stage_C_U carries best_UAV over UAV-derived domains, Stage_C_G '
            'carries best_GE over GE-derived + aerial. Merge Stage_C_U + '
            'Stage_C_G into one "Stage_C" row when reading. Compile with the '
            'FULL manifest after adding these two sources to its matrix (see '
            'the runner note), or compile this file alone for a Stage-C-only '
            'matrix.'),
        'results_dir': a.results_dir,
        'protocol': protocol,
        'matrix': matrix,
        'models': models,
    }
    json.dump(out, open(a.out, 'w'), indent=2)

    ncell = sum(len(SPLIT[m['train_source']]['sensors']) for m in models)
    print(f'[ok] wrote {a.out}')
    print(f'     backbones : {len(MODELS)}')
    print(f'     rows      : {len(models)}  (2 checkpoint-split rows per backbone)')
    print(f'     matrix    : {json.dumps(matrix)}')
    print(f'     cells     : {ncell}  (should be {len(MODELS)*6})')
    print('\n[next]')
    print(f'  PYTHONPATH=/workspace/mmdetection:$PYTHONPATH \\')
    print(f'      python run_cross_transfer.py --manifest {a.out}')
    print('  # optional split across machines:')
    print(f'  #   --train-source Stage_C_U   (UAV-derived, best_UAV)')
    print(f'  #   --train-source Stage_C_G   (GE-derived + aerial, best_GE)')


if __name__ == '__main__':
    main()
