#!/usr/bin/env python3
# ==========================================================================
# splice_uav8.py  --  restore the 8 UAV backbones missing from the manifest
# --------------------------------------------------------------------------
# The cross-transfer manifest lists only 10 of the 18 UAV-trained models, so
# the compiled matrix has been a 10-UAV-row artifact and GE_30sim produced
# only 10 UAV cells. These 8 backbones were evaluated on the five original
# sensors (present on disk) but dropped from the manifest before both backups.
#
# This script reconstructs their entries from the on-disk work_dir layout,
# using the SAME schema as the existing 10 UAV entries (config + checkpoint
# under work_dirs/Stage_A/<stem>/). Every path is verified before writing;
# a model with a missing config or checkpoint is reported and skipped rather
# than written broken. cross_transfer.json is backed up to .bak3.
#
# Usage (from configs/Custom/Evaluation/):
#   python splice_uav8.py --dry-run
#   python splice_uav8.py
# ==========================================================================

import argparse
import glob
import json
import os.path as osp
import shutil
import sys

WORKDIR = '/workspace/mmdetection/work_dirs/Stage_A'

# stem -> (manuscript label, family)
MODELS = {
    'maskrcnn_r101_uav5cm':            ('ResNet-101',       'CNN'),
    'maskrcnn_swin_t_uav5cm':          ('Swin-T',           'Transformer'),
    'maskrcnn_vmamba_t_uav5cm':        ('VMamba-T',         'Mamba'),
    'maskrcnn_spatialmamba_t_uav5cm':  ('SpatialMamba-T',   'Mamba'),
    'maskrcnn_groupmamba_t_uav5cm':    ('GroupMamba-T',     'Mamba'),
    'maskrcnn_mambaout_t_uav5cm':      ('MambaOut-T',       'Mamba'),
    'maskrcnn_mambavision_t_uav5cm':   ('MambaVision-T',    'Mamba'),
    'maskrcnn_efficientvmamba_s_uav5cm': ('EfficientVMamba-S', 'Mamba'),
}


def find_checkpoint(stem: str) -> str:
    """Best segm checkpoint, tolerant of the mAP vs mAP_50 filename variants."""
    d = osp.join(WORKDIR, stem)
    for pat in ('best_coco_segm_mAP_50_iter_*.pth',   # 50-variant (most models)
                'best_coco_segm_mAP_iter_*.pth',       # non-50 variant (e.g. r50)
                'best_*segm_mAP*_*.pth'):               # last-resort catch-all
        hits = sorted(glob.glob(osp.join(d, pat)))
        if hits:
            return hits[-1]                             # latest iter if several
    return ''


def main() -> None:
    ap = argparse.ArgumentParser(description='Splice 8 UAV models into manifest.')
    ap.add_argument('--manifest', default='cross_transfer.json')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    if not osp.isfile(a.manifest):
        sys.exit(f'[ERROR] {a.manifest} not found. Run from Evaluation/.')
    m = json.load(open(a.manifest))
    have = {x['label'] for x in m['models']}

    new, problems = [], []
    for stem, (label, family) in MODELS.items():
        if label in have:
            print(f'[skip] {label}: already in manifest.')
            continue
        cfg  = osp.join(WORKDIR, stem, f'{stem}.py')
        ckpt = find_checkpoint(stem)
        if not osp.isfile(cfg):
            problems.append(f'{label}: config missing ({cfg})'); continue
        if not ckpt:
            problems.append(f'{label}: checkpoint missing under {WORKDIR}/{stem}/')
            continue
        entry = {'label': label, 'family': family, 'train_source': 'UAV',
                 'config': cfg, 'checkpoint': ckpt, 'applied_score_thr': None}
        new.append(entry)
        print(f'[add ] {label:18s} family={family:11s} '
              f'ckpt={osp.basename(ckpt)}')

    if problems:
        print('\n[PROBLEM] the following were NOT added:')
        for p in problems:
            print('   ', p)
        print('    Resolve these (locate the checkpoint) and re-run; the '
              'others are safe to add now.')

    if not new:
        print('\n[done] nothing to add.')
        return

    uav = sum(1 for x in m['models'] if x['train_source'] == 'UAV') + len(new)
    tot = len(m['models']) + len(new)
    print(f'\n[summary] adding {len(new)} model(s); '
          f'UAV -> {uav}, total -> {tot}')

    if a.dry_run:
        print('[dry-run] manifest not modified.')
        return
    shutil.copy(a.manifest, a.manifest + '.bak3')
    m['models'].extend(new)
    json.dump(m, open(a.manifest, 'w'), indent=2)
    print(f'[ok] {a.manifest} updated (backup -> {a.manifest}.bak3).')
    print('\n[next]\n'
          '  python make_ge30sim_manifest.py --results-dir '
          '/workspace/mmdetection/results/cross_transfer --src cross_transfer.json\n'
          '  PYTHONPATH=/workspace/mmdetection:$PYTHONPATH \\\n'
          '      python run_cross_transfer.py --manifest cross_transfer_ge30sim.json')


if __name__ == '__main__':
    main()
