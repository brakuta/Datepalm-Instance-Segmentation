#!/usr/bin/env python3
# ==========================================================================
# make_ge30sim_manifest.py
# --------------------------------------------------------------------------
# Derives a GE_30sim-ONLY manifest from the existing cross_transfer.json.
#
# WHY
#   run_cross_transfer.py resumes by testing whether a cell's CSV exists in
#   results_dir. results_dir is RELATIVE in the shipped manifest, so it
#   resolves against the current working directory; launching the driver
#   from a different directory than the original stages makes every cell
#   look absent and triggers a full 168-cell recompute.
#
#   This script sidesteps the issue entirely: the derived manifest asks for
#   exactly one test sensor (GE_30sim) on both training sources, so only the
#   28 missing cells can ever be evaluated, regardless of resume state.
#   results_dir is rewritten to an ABSOLUTE path so the new CSVs land beside
#   the existing ones no matter where the driver is launched from.
#
#   Everything else -- protocol, model list, configs, checkpoints,
#   applied_score_thr -- is copied verbatim, so the new cells are produced
#   under byte-identical evaluation settings to the existing matrix.
#
# Usage (from configs/Custom/Evaluation/):
#   python make_ge30sim_manifest.py --results-dir /abs/path/to/results/cross_transfer
#   # then:
#   python run_cross_transfer.py --manifest cross_transfer_ge30sim.json
#   python compile_cross_transfer.py --manifest cross_transfer.json   # FULL manifest
# ==========================================================================

import argparse
import json
import os.path as osp
import sys

SENSOR = 'GE_30sim'


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Write a GE_30sim-only manifest derived from the full one.')
    ap.add_argument('--src', default='cross_transfer.json',
                    help='existing full manifest (read-only)')
    ap.add_argument('--out', default='cross_transfer_ge30sim.json')
    ap.add_argument('--results-dir', default=None,
                    help='ABSOLUTE results dir holding the existing CSVs. '
                         'Default: absolutise the value found in --src.')
    a = ap.parse_args()

    if not osp.isfile(a.src):
        sys.exit(f'[ERROR] {a.src} not found. Run from configs/Custom/Evaluation/.')
    m = json.load(open(a.src))

    rd = a.results_dir or osp.abspath(m.get('results_dir', 'results/cross_transfer'))
    if not osp.isabs(rd):
        sys.exit(f'[ERROR] --results-dir must be absolute; got {rd}')
    if not osp.isdir(rd):
        print(f'[warn] {rd} does not exist yet; it will be created by the driver.')
    else:
        n = len([f for f in __import__("os").listdir(rd) if f.endswith('.csv')])
        print(f'[info] results_dir holds {n} existing CSV(s): {rd}')

    models = m.get('models', [])
    if not models:
        sys.exit(f'[ERROR] {a.src} has no "models".')
    srcs = sorted({x['train_source'] for x in models})

    out = {
        '_comment': (
            f'GE_30sim-only manifest, derived from {osp.basename(a.src)}. '
            f'Adds the missing column of the transfer matrix: the GE 15 cm TEST '
            f'split resampled to 30 cm, evaluated ZERO-SHOT by every '
            f'single-source model. Protocol, models, configs and checkpoints are '
            f'copied verbatim from the source manifest, so these cells are '
            f'directly comparable with the existing ones. Compile with the FULL '
            f'manifest, not this one.'),
        'results_dir': rd,
        'protocol': m['protocol'],
        'matrix': {s: [SENSOR] for s in srcs},
        'models': models,
    }
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=2)

    n_cells = len(models)
    per_src = {s: sum(1 for x in models if x['train_source'] == s) for s in srcs}
    print(f'[ok] wrote {a.out}')
    print(f'     matrix      : {json.dumps(out["matrix"])}')
    print(f'     models      : {n_cells} ' +
          '(' + ', '.join(f'{s}={n}' for s, n in per_src.items()) + ')')
    print(f'     cells to run: {n_cells}  (one GE_30sim cell per model)')
    print(f'\n[next]\n'
          f'  python run_cross_transfer.py --manifest {a.out}\n'
          f'  # optionally split across machines:\n'
          f'  #   python run_cross_transfer.py --manifest {a.out} --train-source GE_Aerial\n'
          f'  #   python run_cross_transfer.py --manifest {a.out} --train-source UAV\n'
          f'  python compile_cross_transfer.py --manifest {a.src}')


if __name__ == '__main__':
    main()
