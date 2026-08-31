#!/usr/bin/env python3
"""
Measure what the post-processing gates cost in recall, and sweep them.
=============================================================================
WHY THIS EXISTS
  calibrate reports precision and recall of the RAW model output: it matches
  pred_instances straight against COCO ground truth and sweeps the score
  threshold. The deployed country-scale pipeline does considerably more to
  every detection before it becomes a record -- morphological cleaning, a
  minimum mask area, a shape gate, a geometry build, and polygon NMS. Every
  one of those is a filter and none of them can add a detection, so pipeline
  recall is bounded above by the model recall calibrate reports. Quoting the
  calibration figure as though it described the pipeline overstates
  completeness, which matters directly when the detection count is the
  published product.

  This replays the gates on cached calibration predictions and re-runs the
  same greedy matcher on the raw and the gated stream, so the cost is a
  measured number. --sweep then varies the area floor and whichever shape
  thresholds the active SHAPE_GATE mode uses, so they can be chosen from data
  instead of inherited.

  The gates are imported from the pipeline (analyse_mask), not reimplemented
  here. A measurement that reimplements the thing it measures measures the
  reimplementation.

SCOPE
  Covers the per-detection mask gates, which act on a single tile and are
  reproducible from a per-tile prediction cache. It does NOT cover
  global_polygon_nms or tile-ownership rejection: both act across a whole
  mosaicked unit and cannot be replayed from per-tile predictions. Treat the
  recall reported here as an UPPER bound on the deployed pipeline's recall --
  tighter than calibrate's, still optimistic.

  It also measures on GE test tiles, which are less dense than the worst
  production plantations. That is the same blind spot that made the old
  max_per_img cap look safe on validation.

USAGE
  python measure_postproc_recall.py \
      --pkl /workspace/datasets/GEE_Geotiff/output/calibration_preds.pkl \
      --gt  /workspace/datasets/COCO/GE_15cm/Annotations/test_GE.json

  python measure_postproc_recall.py --pkl ... --gt ... --sweep
"""

from __future__ import annotations

import argparse
import copy
import os.path as osp
import pickle
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))


def as_dense(m):
    """Binary uint8 mask, from a dense array or a COCO RLE dict."""
    from pycocotools import mask as maskUtils
    if isinstance(m, dict):
        return maskUtils.decode(m).astype(np.uint8)
    return np.asarray(m, dtype=np.uint8)


def as_rle(m):
    """COCO RLE for a mask that may already be one -- the cache stores RLE,
    so re-encoding would be a decode/encode round trip for nothing."""
    from pycocotools import mask as maskUtils
    if isinstance(m, dict):
        return m
    return maskUtils.encode(np.asfortranarray(np.asarray(m, dtype=np.uint8)))


class _Gates:
    """Duck-typed stand-in for CONFIG, so one gate setting can be swept
    without mutating the module-level CONFIG the pipeline reads."""

    def __init__(self, cfg, **over):
        for k in ('MORPH_KERNEL_PX', 'MIN_MASK_AREA_PX', 'CIRCULARITY_MIN',
                  'CIRCULARITY_SMOOTH_PX', 'SOLIDITY_MIN', 'AXIS_RATIO_MIN',
                  'SHAPE_GATE'):
            setattr(self, k, getattr(cfg, k, None))
        for k, v in over.items():
            if v is not None:
                setattr(self, k, v)

    def __repr__(self):
        return (f'<gates area>={self.MIN_MASK_AREA_PX} '
                f'mode={self.SHAPE_GATE} circ>={self.CIRCULARITY_MIN} '
                f'sol>={self.SOLIDITY_MIN} axis>={self.AXIS_RATIO_MIN}>')


def evaluate(results, coco, gates, iou_thr, want_raw=False):
    """Run one gate setting over the whole cache.

    Returns a dict with the matched/score streams for the gated output, the
    per-gate drop tally, and (when want_raw) the same streams for the
    ungated model output.
    """
    from palm_inference_pipeline import GATE_NAMES, analyse_mask, _greedy_match
    from pycocotools import mask as maskUtils

    out = dict(total_gt=0, n_raw=0, n_kept=0, n_nomask=0,
               drops={g: 0 for g in GATE_NAMES},
               sc_gate=[], mt_gate=[], sc_raw=[], mt_raw=[],
               circ=[], sol=[], ar=[])

    for r in results:
        img_id = int(r['img_id'])
        if img_id not in coco.imgs:
            continue
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))
        out['total_gt'] += len(anns)
        gts = [coco.annToRLE(a) for a in anns]

        pi = r['pred_instances']
        scores = np.asarray(pi['scores'], dtype=np.float64)
        if len(scores) == 0:
            continue
        # The gates all operate on masks. A record without them cannot be
        # replayed at all, so it is counted and skipped rather than silently
        # treated as if every detection survived.
        if pi.get('masks') is None:
            out['n_nomask'] += 1
            continue

        order = np.argsort(-scores)          # _greedy_match needs descending
        raw_rle, raw_sc, keep_rle, keep_sc = [], [], [], []
        for i in order:
            src = pi['masks'][int(i)]
            if want_raw:
                raw_rle.append(as_rle(src))
                raw_sc.append(scores[i])
            out['n_raw'] += 1

            info, reason = analyse_mask(as_dense(src), gates)
            if reason is not None:
                out['drops'][reason] += 1
                continue
            out['circ'].append(info['circularity'])
            out['sol'].append(info['solidity'])
            out['ar'].append(info['axis_ratio'])
            # The mask the pipeline keeps is the CLEANED one, so that is what
            # gets matched -- morphological cleaning moves the boundary and
            # therefore moves the IoU.
            keep_rle.append(maskUtils.encode(
                np.asfortranarray(info['mask'])))
            keep_sc.append(scores[i])
            out['n_kept'] += 1

        if want_raw and raw_rle:
            out['sc_raw'].extend(raw_sc)
            out['mt_raw'].extend(_greedy_match(raw_rle, gts, iou_thr))
        if keep_rle:
            out['sc_gate'].extend(keep_sc)
            out['mt_gate'].extend(_greedy_match(keep_rle, gts, iou_thr))
    return out


def pr_at(scores, matched, total_gt, thr):
    """Precision, recall and F1 at one score threshold."""
    if not scores:
        return 0.0, 0.0, 0.0
    sc = np.asarray(scores)
    mt = np.asarray(matched)
    o = np.argsort(-sc)
    sc, mt = sc[o], mt[o]
    tp = np.cumsum(mt)
    fp = np.cumsum(~mt)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(total_gt, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = max(int(np.searchsorted(-sc, -thr, side='right')) - 1, 0)
    return float(prec[i]), float(rec[i]), float(f1[i])


def main() -> None:
    from palm_inference_pipeline import CONFIG, SHAPE_GATE_MODES

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pkl', required=True,
                    help='calibration predictions cache written by calibrate')
    ap.add_argument('--gt', required=True, help='COCO annotation json')
    ap.add_argument('--score-thr', type=float, default=None,
                    help='operating threshold to report at '
                         '(default: CONFIG.SCORE_THR)')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--morph-kernel', type=int, default=None)
    ap.add_argument('--min-mask-area', type=int, default=None)
    ap.add_argument('--circularity-min', type=float, default=None)
    ap.add_argument('--solidity-min', type=float, default=None)
    ap.add_argument('--axis-ratio-min', type=float, default=None)
    ap.add_argument('--shape-gate', default=None,
                    choices=sorted(SHAPE_GATE_MODES))
    ap.add_argument('--sweep', action='store_true',
                    help='vary the shape and area thresholds and report '
                         'P/R/F1 for each, so they can be chosen from data')
    ap.add_argument('--sweep-shape', type=float, nargs='+',
                    default=[0.0, 0.30, 0.40, 0.50, 0.60, 0.70])
    ap.add_argument('--sweep-area', type=int, nargs='+',
                    default=[0, 10, 25, 50])
    args = ap.parse_args()

    from pycocotools.coco import COCO

    thr = args.score_thr if args.score_thr is not None else CONFIG.SCORE_THR
    base = _Gates(CONFIG,
                  MORPH_KERNEL_PX=args.morph_kernel,
                  MIN_MASK_AREA_PX=args.min_mask_area,
                  CIRCULARITY_MIN=args.circularity_min,
                  SOLIDITY_MIN=args.solidity_min,
                  AXIS_RATIO_MIN=args.axis_ratio_min,
                  SHAPE_GATE=args.shape_gate)

    coco = COCO(args.gt)
    with open(args.pkl, 'rb') as fh:
        results = pickle.load(fh)
    print(f'{len(results)} image prediction set(s) loaded')
    print(f'gates: morph={base.MORPH_KERNEL_PX} '
          f'area>={base.MIN_MASK_AREA_PX} shape_gate={base.SHAPE_GATE} '
          f'circ>={base.CIRCULARITY_MIN} sol>={base.SOLIDITY_MIN} '
          f'axis>={base.AXIS_RATIO_MIN} '
          f'smooth={base.CIRCULARITY_SMOOTH_PX}')
    print(f'score threshold {thr}, mask IoU {args.iou}')

    res = evaluate(results, coco, base, args.iou, want_raw=True)
    gt = res['total_gt']

    p_raw, r_raw, f_raw = pr_at(res['sc_raw'], res['mt_raw'], gt, thr)
    p_gat, r_gat, f_gat = pr_at(res['sc_gate'], res['mt_gate'], gt, thr)

    print()
    print('=' * 72)
    print(f'GT instances: {gt:,}')
    print('-' * 72)
    print(f'{"raw model output":24s} P={p_raw:.3f}  R={r_raw:.3f}  '
          f'F1={f_raw:.3f}   (n={len(res["sc_raw"]):,})')
    print(f'{"after mask gates":24s} P={p_gat:.3f}  R={r_gat:.3f}  '
          f'F1={f_gat:.3f}   (n={len(res["sc_gate"]):,})')
    print('-' * 72)
    print(f'detections entering gates : {res["n_raw"]:,}')
    for g, v in res['drops'].items():
        print(f'  dropped, {g:14s}: {v:,} '
              f'({100 * v / max(res["n_raw"], 1):.2f}%)')
    print(f'surviving                 : {res["n_kept"]:,}')
    if res['n_nomask']:
        print(f'[warn] {res["n_nomask"]} record(s) carried no masks and were '
              f'excluded from both streams')
    for name, vals in (('circularity', res['circ']), ('solidity', res['sol']),
                       ('axis_ratio', res['ar'])):
        if vals:
            v = np.asarray(vals)
            print(f'  {name:11s} of survivors: p1={np.percentile(v, 1):.3f} '
                  f'p50={np.median(v):.3f}  p99={np.percentile(v, 99):.3f}')
    dr = r_gat - r_raw
    print('-' * 72)
    print(f'RECALL COST OF THE GATES: {dr:+.4f} '
          f'({100 * dr / max(r_raw, 1e-9):+.2f}% relative)')
    print(f'PRECISION GAINED        : {p_gat - p_raw:+.4f}')
    print('Global polygon NMS and tile-ownership rejection are NOT included; '
          'true pipeline recall is at or below this.')
    print('=' * 72)

    if not args.sweep:
        return

    print()
    print('SWEEP -- pick the threshold, do not inherit it')
    print(f'{"area":>6s}{"shape":>8s}{"P":>9s}{"R":>9s}{"F1":>9s}'
          f'{"kept":>10s}')
    best = None
    for a in args.sweep_area:
        for s in args.sweep_shape:
            g = copy.copy(base)
            g.MIN_MASK_AREA_PX = a
            # Move whichever thresholds this mode actually gates on, so the
            # swept column means something in every mode.
            active = SHAPE_GATE_MODES[g.SHAPE_GATE]
            if 'circularity' in active:
                g.CIRCULARITY_MIN = s
            if 'solidity' in active:
                g.SOLIDITY_MIN = s
            if 'axis_ratio' in active:
                g.AXIS_RATIO_MIN = s
            e = evaluate(results, coco, g, args.iou)
            p, r, f = pr_at(e['sc_gate'], e['mt_gate'], gt, thr)
            print(f'{a:6d}{s:8.2f}{p:9.3f}{r:9.3f}{f:9.3f}'
                  f'{e["n_kept"]:10,d}')
            if best is None or f > best[0]:
                best = (f, a, s, p, r)
    if best:
        print(f'\nbest F1 {best[0]:.3f} at area>={best[1]} shape>={best[2]:.2f}'
              f'  (P={best[3]:.3f} R={best[4]:.3f})')
        print('F1 weights a missed palm and a false palm equally. If the '
              'deliverable is a COUNT, recall is worth more than that, so '
              'read the whole table rather than only this line.')


if __name__ == '__main__':
    main()
