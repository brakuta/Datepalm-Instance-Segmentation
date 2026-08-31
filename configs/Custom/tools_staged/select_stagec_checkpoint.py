#!/usr/bin/env python3
# ==========================================================================
# tools_staged/select_stagec_checkpoint.py   (Stage D, v1)
# --------------------------------------------------------------------------
# For each Stage C work_dir: reads every validation record from
# vis_data/*scalars.json, reports the iteration with the best MEAN of the
# two sensor segm_mAP_50 values, and recommends which SURVIVING .pth to use
# as the "Stage C weights" initialisation for Stage D.
#
# Recommendation rule: among surviving checkpoints (best_UAV_*, best_GE_*,
# iter_*), pick the one whose iteration is closest to the best-mean
# iteration; ties broken by the higher mean at that checkpoint's iteration.
#
# Usage (run on EACH machine):
#   python tools_staged/select_stagec_checkpoint.py \
#       --root /workspace/mmdetection/work_dirs/Stage_C
# ==========================================================================
import argparse
import glob
import json
import os
import re


def val_records(work_dir):
    recs = {}
    for sj in glob.glob(os.path.join(work_dir, '*', 'vis_data',
                                     '*scalars.json')):
        for line in open(sj):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = r.get('UAV/coco/segm_mAP_50')
            g = r.get('GE/coco/segm_mAP_50')
            if u is None or g is None:
                continue
            step = r.get('step', r.get('iter'))
            if step is None:
                continue
            recs[int(step)] = (u + g) / 2.0
    return recs


def ckpt_iter(path):
    m = re.search(r'iter_(\d+)\.pth$', path)
    return int(m.group(1)) if m else None


def mean_at(recs, it, tol=2):
    if not recs:
        return None
    near = min(recs, key=lambda s: abs(s - it))
    return recs[near] if abs(near - it) <= max(tol, 2) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/workspace/mmdetection/work_dirs/Stage_C')
    args = ap.parse_args()

    for wd in sorted(glob.glob(os.path.join(args.root, 'maskrcnn_*'))):
        name = os.path.basename(wd)
        recs = val_records(wd)
        pths = glob.glob(os.path.join(wd, '**', '*.pth'), recursive=True)
        if not recs or not pths:
            print(f'{name}: missing scalars or checkpoints -- skipped')
            continue

        best_step = max(recs, key=recs.get)
        cands = []
        for p in pths:
            it = ckpt_iter(p)
            if it is None:
                continue
            m = mean_at(recs, it)
            cands.append((abs(it - best_step), -(m if m is not None else -1),
                          it, m, p))
        cands.sort()
        _, _, it, m, pick = cands[0]

        print(f'\n{name}')
        print(f'  best mean segm_mAP_50 = {recs[best_step]:.4f} '
              f'at iter {best_step}')
        for _, _, it_, m_, p_ in cands[:4]:
            mark = '  -> USE' if p_ == pick else '        '
            ms = f'{m_:.4f}' if m_ is not None else '  n/a '
            print(f'{mark}  iter {it_:>6}  mean={ms}  '
                  f'{os.path.basename(p_)}')
        print(f'  load_from={pick}')


if __name__ == '__main__':
    main()
