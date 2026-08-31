#!/usr/bin/env python3
"""
Compare two runs of the same unit and characterise what changed.
=============================================================================
WHY THIS EXISTS
  Re-running a unit with the filters opened up answers "how many more?" but
  not the two questions that decide whether the extras matter:

      WHICH filter excluded each one -- the score threshold, the shape gate,
      the area floor? A detection kept out by score at 0.07 is a different
      claim from one kept out by circularity at 0.99 confidence.

      ARE THEY PALMS? Recall measured against annotations cannot answer this,
      because the annotations are the thing in doubt. Only looking can. So
      this writes the added detections out as their own layer, plus a random
      sample sized for hand-scoring, ready to drop on the imagery in QGIS.

  It also reports what the open run LOST relative to the baseline. Opening
  filters should only add, so anything lost is a signal that something other
  than the filters differs between the runs -- worth knowing before drawing
  conclusions from the count.

CONTEXT FOR THE ADDED DETECTIONS
  Distance to the nearest BASELINE detection separates the two populations
  that get conflated in a single count: an extra crown inside a plantation,
  metres from confirmed palms, is very likely a real palm the threshold cut;
  an isolated extra in open desert is the classic false positive from a
  palm-like shrub. The split is reported, and carried as an attribute so the
  visual check can be stratified by it.

USAGE
  python compare_runs.py \
      --baseline /workspace/results/uae_palms_hn/UAE_373_palms.gpkg \
      --open     /workspace/results/openfilter/UAE_373_palms.gpkg \
      --out      /workspace/results/openfilter/UAE_373_added.gpkg \
      --sample   100
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def write_layer(gdf, path, layer):
    """Write via the pipeline's normalize_geometry so the result opens in
    ArcGIS Pro. Declaring geometry_type on a populated frame is rejected by
    pyogrio 0.13; homogenising the column first is what the pipeline settled
    on and it is the behaviour worth matching exactly."""
    from palm_inference_pipeline import normalize_geometry
    normalize_geometry(gdf).to_file(path, driver='GPKG', layer=layer)


def load(path, label):
    import geopandas as gpd
    g = gpd.read_file(path)
    if g.empty:
        sys.exit(f'[ERROR] {label} layer is empty: {path}')
    return g


def pct(v, qs=(1, 25, 50, 75, 99)):
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return 'n/a'
    return '  '.join(f'p{q}={np.percentile(a, q):.3f}' for q in qs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline', required=True,
                    help='the delivered run for this unit')
    ap.add_argument('--open', dest='open_', required=True,
                    help='the same unit re-run with filters opened up')
    ap.add_argument('--out', default=None,
                    help='write the ADDED detections here (GPKG)')
    ap.add_argument('--sample', type=int, default=0,
                    help='also write a random sample of N added detections, '
                         'stratified by plantation/isolated, for hand-scoring')
    ap.add_argument('--match-dist', type=float, default=2.0,
                    help='centroid distance (m) at which a detection in the '
                         'open run is considered the same crown as one in '
                         'the baseline')
    ap.add_argument('--plantation-dist', type=float, default=12.0,
                    help='an added detection within this distance of a '
                         'baseline detection counts as inside a plantation')
    # The deployed operating point, used to attribute each addition to the
    # filter that excluded it. Defaults come from CONFIG so they cannot drift.
    ap.add_argument('--score-thr', type=float, default=None)
    ap.add_argument('--circularity-min', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import geopandas as gpd
    sys.path.insert(0, __file__.rsplit('/', 1)[0])
    try:
        from palm_inference_pipeline import CONFIG
        d_score = CONFIG.SCORE_THR
        d_circ = CONFIG.CIRCULARITY_MIN
    except Exception:
        d_score, d_circ = 0.30, 0.60
    score_thr = args.score_thr if args.score_thr is not None else d_score
    circ_min = (args.circularity_min if args.circularity_min is not None
                else d_circ)

    base = load(args.baseline, 'baseline')
    opn = load(args.open_, 'open')
    if opn.crs != base.crs:
        opn = opn.to_crs(base.crs)
    if base.crs is None or base.crs.is_geographic:
        sys.exit('[ERROR] layers must be in a projected CRS for metric '
                 'distances; reproject to the unit UTM zone first.')

    bxy = np.column_stack([base.geometry.centroid.x.values,
                           base.geometry.centroid.y.values])
    oxy = np.column_stack([opn.geometry.centroid.x.values,
                           opn.geometry.centroid.y.values])

    from scipy.spatial import cKDTree
    tb = cKDTree(bxy)

    # Nearest baseline crown for every open-run detection. Matching on the
    # centroid rather than polygon IoU on purpose: the geometry is an
    # area-preserving circle, so a lower score threshold can change a crown's
    # RADIUS without moving it, and an IoU test would then call the same palm
    # a different one.
    d_ob, _ = tb.query(oxy, k=1)
    is_new = d_ob > args.match_dist

    # And the reverse, to catch anything the open run lost.
    to = cKDTree(oxy)
    d_bo, _ = to.query(bxy, k=1)
    lost = int((d_bo > args.match_dist).sum())

    added = opn[is_new].copy()
    n_b, n_o, n_a = len(base), len(opn), len(added)

    print('=' * 72)
    print(f'baseline   : {n_b:,}')
    print(f'open run   : {n_o:,}')
    print(f'added      : {n_a:,}  ({100 * n_a / max(n_b, 1):+.1f}% of baseline)')
    print(f'lost       : {lost:,}'
          + ('   <- opening filters should only ADD; investigate'
             if lost else ''))
    print('=' * 72)

    if not n_a:
        print('nothing added; the filters were not what limited this unit.')
        return

    # ---- Which filter kept each addition out? -----------------------------
    # Checked in the order the pipeline applies them, so each detection is
    # attributed to the FIRST gate that would have rejected it -- the same
    # logic as the pipeline, which stops at the first failure.
    sc = added['score'].values.astype(float)
    cr = (added['circularity'].values.astype(float)
          if 'circularity' in added else np.full(n_a, np.nan))
    by_score = sc < score_thr
    by_circ = ~by_score & np.isfinite(cr) & (cr < circ_min)
    by_other = ~by_score & ~by_circ
    added['excluded_by'] = np.where(by_score, 'score',
                                    np.where(by_circ, 'circularity', 'other'))

    print('\nWHY EACH ADDITION WAS EXCLUDED FROM THE DELIVERED RUN')
    for name, m in (('score threshold', by_score),
                    ('circularity gate', by_circ),
                    ('area / NMS / other', by_other)):
        print(f'  {name:20s}: {int(m.sum()):>8,}  '
              f'({100 * m.sum() / n_a:5.1f}% of added)')

    print('\nSCORE OF ADDED   ', pct(sc))
    if 'circularity' in added:
        print('CIRCULARITY      ', pct(cr))
    if 'diam_m' in added:
        print('DIAMETER (m)     ', pct(added['diam_m'].values))

    # ---- Plantation context -----------------------------------------------
    dist_added = d_ob[is_new]
    in_plant = dist_added <= args.plantation_dist
    added['dist_to_baseline_m'] = np.round(dist_added, 2)
    added['context'] = np.where(in_plant, 'plantation', 'isolated')
    print(f'\nCONTEXT (distance to nearest confirmed detection)')
    print(f'  inside plantation (<={args.plantation_dist:g} m): '
          f'{int(in_plant.sum()):,} ({100*in_plant.mean():.1f}%)'
          f'   <- likely real palms below threshold')
    print(f'  isolated                    : '
          f'{int((~in_plant).sum()):,} ({100*(~in_plant).mean():.1f}%)'
          f'   <- likely shrubs / false positives')

    # ---- Outputs -----------------------------------------------------------
    if args.out:
        # Written through the pipeline's own homogeniser: a mixed
        # Polygon/MultiPolygon column makes GDAL declare the generic
        # 'GEOMETRY' type, which ArcGIS Pro then refuses to open -- the exact
        # failure that made the first batch of delivered files unopenable.
        write_layer(added, args.out, 'added')
        print(f'\nadded detections -> {args.out}')
    if args.sample and args.out:
        rng = np.random.default_rng(args.seed)
        # Stratified so the smaller class is never lost to chance: hand-scoring
        # 100 detections that are 95% plantation says little about the
        # isolated ones, which are where the false positives live.
        take = []
        for ctx in ('plantation', 'isolated'):
            idx = np.where(added['context'].values == ctx)[0]
            if len(idx):
                k = min(len(idx), max(1, args.sample // 2))
                take.append(rng.choice(idx, size=k, replace=False))
        samp = added.iloc[np.concatenate(take)].copy()
        samp['verdict'] = ''          # fill in by eye: palm / not_palm / unsure
        sp = args.out.replace('.gpkg', '_sample.gpkg')
        write_layer(samp, sp, 'sample')
        print(f'hand-scoring sample ({len(samp)}) -> {sp}')
        print('  Open it on the imagery, fill the empty "verdict" column with '
              'palm / not_palm, and the share of "palm" in each context is '
              'the precision of the additions.')


if __name__ == '__main__':
    main()
