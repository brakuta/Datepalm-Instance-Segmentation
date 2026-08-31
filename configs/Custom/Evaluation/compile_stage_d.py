#!/usr/bin/env python3
# =============================================================================
# compile_stage_d.py
# -----------------------------------------------------------------------------
# Compiles the Stage D test-set results (b0 vs cf, 4 backbones) into one wide
# table.
#
# WHY NOT compile_results.py
#   That compiler derives its backbone label from the CONFIG filename
#   (tidy_backbone(config_stem)). In Stage D, arm b0 and arm cf are trained
#   from the same config file -- maskrcnn_<bb>_staged_full.py -- with only
#   load_from differing at train time, so both arms' accuracy CSVs carry the
#   identical config_stem and would collapse to one indistinguishable backbone
#   label. The arm identity here comes from which results/stage_d/<arm>/
#   directory a CSV was written into (set via --results-dir at eval time),
#   not from the config name, so this script keys on that instead.
#
# INPUT LAYOUT (produced by the Stage D evaluate_model.py loop)
#   results/stage_d/b0/maskrcnn_<bb>_staged_full__test_sat.csv
#   results/stage_d/b0/maskrcnn_<bb>_staged_full__efficiency.csv
#   results/stage_d/cf/maskrcnn_<bb>_staged_full__test_sat.csv
#   results/stage_d/zeroshot/maskrcnn_<bb>_stagec__test_sat.csv
#   results/stage_d/ms/maskrcnn_<bb>_staged_ms__test_ms.csv
#   (the MS arm is evaluated against the 8-band SatMS test set, so its
#   CSV is named __test_ms.csv rather than __test_sat.csv)
#   (cf and zeroshot carry no efficiency CSV of their own -- cf is
#   architecturally identical to b0, and zeroshot re-evaluates the
#   untouched Stage C checkpoint with no WV-3 adaptation at all, so both
#   were run with --no-efficiency to avoid a redundant profiling pass)
#
# USAGE
#   python configs/Custom/Evaluation/compile_stage_d.py \
#       --results-dir results/stage_d
# =============================================================================

from __future__ import annotations

import argparse
import csv
import glob
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from compile_results import (          # noqa: E402
    EFF_COLUMNS, EFF_SUFFIX, parse_eval_csv,
)

ARMS = ('zeroshot', 'b0', 'cf', 'ms')
ARM_LABEL = {'zeroshot': 'Stage C zero-shot', 'b0': 'ImageNet',
            'cf': 'Stage C unified', 'ms': 'ImageNet (8-band)'}
# Each arm's accuracy CSV is named after the CONFIG it was evaluated with,
# and that config's suffix differs by arm: b0/cf share staged_full.py
# (only the checkpoint differs, see the module docstring), while zeroshot
# evaluates the untouched Stage C config directly, whose filename ends
# '_stagec' instead. Longest match first so '_stagec' does not accidentally
# swallow part of a backbone name that happens to contain it.
BACKBONE_SUFFIXES = ('_staged_full', '_staged_ms', '_stagec')


def backbone_of(config_stem: str) -> str:
    stem = config_stem
    if stem.startswith('maskrcnn_'):
        stem = stem[len('maskrcnn_'):]
    for suffix in sorted(BACKBONE_SUFFIXES, key=len, reverse=True):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def load_arm(results_dir: str, arm: str) -> dict:
    """{backbone: record} for one arm's accuracy CSVs."""
    out = {}
    patt = '*__test_ms.csv' if arm == 'ms' else '*__test_sat.csv'
    for csv_path in sorted(glob.glob(osp.join(results_dir, arm, patt))):
        rec = parse_eval_csv(csv_path)
        if rec is None:
            continue
        out[backbone_of(rec['model'])] = rec
    return out


def load_efficiency(results_dir: str, arm: str) -> dict:
    """{backbone: {col: value}} from one arm's efficiency CSVs, if any."""
    out = {}
    for csv_path in sorted(glob.glob(osp.join(results_dir, arm, f'*{EFF_SUFFIX}'))):
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        rec = rows[0]
        bb = backbone_of(rec.get('model', ''))
        out[bb] = {c: rec.get(c, '') for c in EFF_COLUMNS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', default='results/stage_d')
    ap.add_argument('--out', default=None,
                    help='Output CSV path. Default: <results-dir>/compiled/'
                         'stage_d_test_results.csv')
    args = ap.parse_args()

    by_arm = {a: load_arm(args.results_dir, a) for a in ARMS}
    eff = {}
    for a in ARMS:
        eff.update({bb: v for bb, v in load_efficiency(args.results_dir, a).items()
                   if bb not in eff})   # b0's efficiency wins; cf has none

    backbones = sorted(set(by_arm['b0']) | set(by_arm['cf']))
    if not backbones:
        sys.exit(f'[ERROR] no accuracy CSVs found under {args.results_dir}/{{zeroshot,b0,cf,ms}}/')

    out_path = args.out or osp.join(args.results_dir, 'compiled',
                                    'stage_d_test_results.csv')
    import os
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    fields = ['backbone', 'arm', 'init',
             'bbox_mAP50', 'segm_mAP50',
             'bbox_F1_opt', 'segm_F1_opt',
             'bbox_P_opt', 'segm_P_opt', 'bbox_R_opt', 'segm_R_opt',
             'bbox_mAP', 'segm_mAP',
             'gflops', 'params_total_M', 'fps']
    rows_out = []
    for bb in backbones:
        for arm in ARMS:
            rec = by_arm[arm].get(bb)
            if rec is None:
                sys.stderr.write(f'[WARN] no {arm} test-set result for {bb}\n')
                continue
            e = eff.get(bb, {})
            rows_out.append({
                'backbone': bb, 'arm': arm, 'init': ARM_LABEL[arm],
                'bbox_mAP50': rec['bbox'].get('mAP50'),
                'segm_mAP50': rec['segm'].get('mAP50'),
                'bbox_F1_opt': rec['bbox'].get('F1_opt'),
                'segm_F1_opt': rec['segm'].get('F1_opt'),
                'bbox_P_opt': rec['bbox'].get('P_opt'),
                'segm_P_opt': rec['segm'].get('P_opt'),
                'bbox_R_opt': rec['bbox'].get('R_opt'),
                'segm_R_opt': rec['segm'].get('R_opt'),
                'bbox_mAP': rec['bbox'].get('mAP'),
                'segm_mAP': rec['segm'].get('mAP'),
                'gflops': e.get('gflops', ''),
                'params_total_M': e.get('params_total_M', ''),
                'fps': e.get('fps', ''),
            })

    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    # Readable console table, one line per backbone, both arms side by side.
    w = max(len(b) for b in backbones)
    def mAP50(arm, bb):
        r = by_arm[arm].get(bb)
        return r['segm']['mAP50'] if r and 'mAP50' in r['segm'] else None

    print(f"\n{'backbone'.ljust(w)}  zero-shot   ImageNet   StageC   8-band   "
          f"MS gain")
    print('-' * (w + 56))
    for bb in backbones:
        vz = mAP50('zeroshot', bb)
        v0 = mAP50('b0', bb)
        v1 = mAP50('cf', bb)
        vm = mAP50('ms', bb)
        # The MS arm's comparator is the RGB ImageNet arm: same
        # initialisation and recipe, differing only in spectral input.
        gm = f"{vm - v0:+.4f}" if (vm is not None and v0 is not None) else '-'
        print(f"{bb.ljust(w)}  {str(vz):>9}  {str(v0):>8}  {str(v1):>7}  "
              f"{str(vm):>7}  {gm:>8}")

    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
