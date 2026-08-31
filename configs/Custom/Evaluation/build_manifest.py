#!/usr/bin/env python3
# ==========================================================================
# build_manifest.py
# --------------------------------------------------------------------------
# Auto-generates the cross-transfer manifest for ALL single-source models by
# scanning the training work_dirs. Each Stage A run dir becomes a UAV-trained
# model; each Stage B run dir a 15 cm-trained (GE_Aerial) model. For each run
# it resolves the best checkpoint and the matching config, and infers a clean
# backbone label and family from the run-dir name.
#
# It does NOT guess accuracy thresholds: applied_score_thr is left null. To
# reproduce the Stage A/B/C F1 operating point exactly, fill each model's
# applied_score_thr with the validation-selected threshold you used in-domain
# (mAP@50 — hence every transfer gap — is unaffected by this field).
#
# Any run whose config or checkpoint cannot be resolved is written with an
# "UNRESOLVED__..." placeholder and listed at the end, so nothing is silently
# dropped; fix those lines by hand before running.
#
# Usage
# -----
#   python build_manifest.py \
#       --stage-a-dir work_dirs/Stage_A \
#       --stage-b-dir work_dirs/Stage_B \
#       --config-roots configs/Custom \
#       --max-dets 500 \
#       --results-dir results/cross_transfer \
#       --out cross_transfer.json
# ==========================================================================

import argparse
import glob
import json
import os
import os.path as osp
import re
import sys
from typing import List, Optional, Tuple

# Backbone label/family inference. Ordered specific -> general; first match
# wins. EfficientVMamba precedes VMamba (its name contains 'vmamba').
CANON: List[Tuple[str, str, str]] = [
    (r'(r101|resnet[_-]?101)',        'ResNet-101',       'CNN'),
    (r'(r50|resnet[_-]?50)',          'ResNet-50',        'CNN'),
    (r'convnext.*_s',                 'ConvNeXt-S',       'CNN'),
    (r'convnext',                     'ConvNeXt-T',       'CNN'),
    (r'swin.*_s',                     'Swin-S',           'Transformer'),
    (r'swin.*_t',                     'Swin-T',           'Transformer'),
    (r'(pvt.*b2|pvtv2[_-]?b2)',       'PVTv2-B2',         'Transformer'),
    (r'uniformer',                    'UniFormer-XS',     'Transformer'),
    (r'spatialmamba.*_s',             'SpatialMamba-S',   'Mamba'),
    (r'spatialmamba.*_t',             'SpatialMamba-T',   'Mamba'),
    (r'groupmamba.*_s',               'GroupMamba-S',     'Mamba'),
    (r'groupmamba.*_t',               'GroupMamba-T',     'Mamba'),
    (r'efficientvmamba.*_b',          'EfficientVMamba-B', 'Mamba'),
    (r'efficientvmamba.*_s',          'EfficientVMamba-S', 'Mamba'),
    (r'mambaout.*_s',                 'MambaOut-S',       'Mamba'),
    (r'mambaout.*_t',                 'MambaOut-T',       'Mamba'),
    (r'mambavision.*_s',              'MambaVision-S',    'Mamba'),
    (r'mambavision.*_t',              'MambaVision-T',    'Mamba'),
    (r'vmamba.*_s',                   'VMamba-S',         'Mamba'),
    (r'vmamba.*_t',                   'VMamba-T',         'Mamba'),
]


def infer_label_family(stem: str) -> Tuple[str, str]:
    s = stem.lower()
    for pat, label, fam in CANON:
        if re.search(pat, s):
            return label, fam
    return stem, ''            # fall back to the run-dir name


def resolve_checkpoint(run_dir: str, ckpt_glob: str) -> Optional[str]:
    """Returns the best checkpoint in run_dir (highest iter if several)."""
    hits = sorted(glob.glob(osp.join(run_dir, ckpt_glob)))
    if not hits:
        # fall back to whatever last_checkpoint points at, if present
        lc = osp.join(run_dir, 'last_checkpoint')
        if osp.isfile(lc):
            with open(lc) as f:
                p = f.read().strip()
            return p if osp.isfile(p) else None
        return None
    if len(hits) == 1:
        return hits[0]

    def it(path):
        m = re.search(r'iter_(\d+)', osp.basename(path))
        return int(m.group(1)) if m else -1
    return max(hits, key=it)


def resolve_config(stem: str, run_dir: str,
                   config_roots: List[str]) -> Tuple[Optional[str], bool]:
    """Finds the config .py for a run. The dumped config inside the run dir is
    authoritative: it is the exact config that produced the checkpoint and
    cannot have drifted from the weights (this is what prevents the d_state-style
    config/checkpoint mismatch). Only when no dump exists does it fall back to
    the authored config under the config roots, which MAY differ from the
    trained weights. Returns (config_path_or_None, fell_back_to_config_root)."""
    # 1) Exact-name dump in the run dir.
    dumped = osp.join(run_dir, stem + '.py')
    if osp.isfile(dumped):
        return dumped, False
    # 2) Any .py dumped directly in the run dir (mmengine names it after the
    #    config; prefer the stem match, else take the sole .py if unambiguous).
    run_pys = sorted(glob.glob(osp.join(run_dir, '*.py')))
    stem_match = [p for p in run_pys if osp.splitext(osp.basename(p))[0] == stem]
    if stem_match:
        return stem_match[0], False
    if len(run_pys) == 1:
        return run_pys[0], False
    # 3) Fall back to the authored config under the config roots. Flagged,
    #    because this file is not guaranteed consistent with the checkpoint.
    for root in config_roots:
        for cand in glob.glob(osp.join(root, '**', stem + '.py'), recursive=True):
            return cand, True
    return None, False


def collect_stage(stage_dir: str, train_source: str, ckpt_glob: str,
                  config_roots: List[str]):
    models, unresolved, fell_back = [], [], []
    if not osp.isdir(stage_dir):
        print(f'[warn] stage dir not found: {stage_dir}')
        return models, unresolved, fell_back
    for run_dir in sorted(glob.glob(osp.join(stage_dir, '*'))):
        if not osp.isdir(run_dir):
            continue
        stem = osp.basename(run_dir)
        label, family = infer_label_family(stem)
        ckpt = resolve_checkpoint(run_dir, ckpt_glob)
        config, used_root = resolve_config(stem, run_dir, config_roots)
        entry = {
            'label': label, 'family': family, 'train_source': train_source,
            'config': config or f'UNRESOLVED__config__{stem}.py',
            'checkpoint': ckpt or f'UNRESOLVED__checkpoint__{run_dir}/{ckpt_glob}',
            'applied_score_thr': None,
        }
        models.append(entry)
        if config is None or ckpt is None:
            unresolved.append((stem, config is None, ckpt is None))
        elif used_root:
            fell_back.append((label, config))
    return models, unresolved, fell_back


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Build the cross-transfer manifest from training work_dirs.')
    ap.add_argument('--stage-a-dir', default='work_dirs/Stage_A',
                    help='UAV-trained runs (train_source=UAV)')
    ap.add_argument('--stage-b-dir', default='work_dirs/Stage_B',
                    help='pooled 15 cm runs (train_source=GE_Aerial)')
    ap.add_argument('--config-roots', nargs='+', default=['configs/Custom'],
                    help='roots searched recursively for <run_stem>.py')
    ap.add_argument('--ckpt-glob', default='best_coco_segm_mAP_50_*.pth',
                    help='checkpoint glob within each run dir')
    ap.add_argument('--results-dir', default='results/cross_transfer')
    ap.add_argument('--max-dets', type=int, default=500)
    ap.add_argument('--batch-native', type=int, default=2)
    ap.add_argument('--batch-noresize', type=int, default=1)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--out', default='cross_transfer.json')
    args = ap.parse_args()

    a_models, a_un, a_fb = collect_stage(args.stage_a_dir, 'UAV',
                                         args.ckpt_glob, args.config_roots)
    b_models, b_un, b_fb = collect_stage(args.stage_b_dir, 'GE_Aerial',
                                         args.ckpt_glob, args.config_roots)
    models = a_models + b_models
    if not models:
        sys.exit('[ERROR] no runs discovered. Check --stage-a-dir/--stage-b-dir.')

    manifest = {
        'results_dir': args.results_dir,
        'protocol': {
            'score_thr': 0.05, 'iou_thr': 0.50, 'max_dets': args.max_dets,
            'batch_size_native': args.batch_native,
            'batch_size_noresize': args.batch_noresize,
            'num_workers': args.num_workers,
        },
        'matrix': {
            'UAV':       ['UAV', 'UAV_15sim', 'UAV_30sim', 'GE', 'Aerial'],
            'GE_Aerial': ['GE', 'Aerial', 'UAV', 'UAV_15sim', 'UAV_30sim'],
        },
        'models': models,
    }
    with open(args.out, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\n[manifest] {len(models)} models '
          f'({len(a_models)} UAV-trained, {len(b_models)} 15 cm-trained) '
          f'-> {args.out}')
    print(f'[protocol] max_dets={args.max_dets}, batch native/noresize='
          f'{args.batch_native}/{args.batch_noresize}')
    by_fam = {}
    for m in models:
        by_fam.setdefault(m['family'] or '?', []).append(m['label'])
    for fam, labels in sorted(by_fam.items()):
        print(f'    {fam:12s}: {", ".join(labels)}')

    un = a_un + b_un
    if un:
        print(f'\n[ACTION REQUIRED] {len(un)} run(s) need manual fixing in {args.out}:')
        for stem, no_cfg, no_ckpt in un:
            miss = []
            if no_cfg:
                miss.append('config')
            if no_ckpt:
                miss.append('checkpoint')
            print(f'    {stem}: unresolved {" & ".join(miss)}')
    else:
        print('\n[ok] every run resolved a config and a checkpoint.')

    fb = a_fb + b_fb
    if fb:
        print(f'\n[WARNING] {len(fb)} model(s) used an AUTHORED config from the '
              f'config roots (no dump found in the run dir). These are NOT '
              f'guaranteed consistent with the trained checkpoint — verify each '
              f'backbone arg (e.g. d_state) against the checkpoint before trusting:')
        for label, cfg in fb:
            print(f'    {label}: {cfg}')

    print(f'\n[next]\n  python run_cross_transfer.py --manifest {args.out}\n'
          f'  python compile_cross_transfer.py --manifest {args.out}')


if __name__ == '__main__':
    main()
