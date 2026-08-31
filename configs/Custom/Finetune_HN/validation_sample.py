#!/usr/bin/env python3
# =============================================================================
# validation_sample.py
# -----------------------------------------------------------------------------
# Turn a national detection map into a CITABLE national estimate.
#
# WHY THIS EXISTS
#   A model count is not a national total, however good the model is. Detection
#   metrics (precision, recall, mAP) describe the model on the data it was
#   tested on; they do not carry an uncertainty for the number that gets
#   published, quoted in news, and built on by other studies. "We counted
#   N palms" invites the question "how do you know it is not 20% out", and
#   without a probability sample there is no answer.
#
#   The standard remedy is a design-based estimate from a stratified random
#   sample with human verification (Olofsson et al. 2014, Good practices for
#   estimating area and assessing accuracy of land change, RSE 148:42-57). It
#   yields a corrected total WITH a confidence interval, and the interval is
#   what makes the number defensible.
#
# THE TWO ESTIMATORS THIS REPORTS
#   Stratified mean. Yhat = sum_h N_h * ybar_h, where N_h is the number of
#   tiles in stratum h and ybar_h the mean verified count in the sample. Simple
#   and assumption-free, but its interval is wide because it ignores what the
#   model already tells you about every tile.
#
#   Stratified RATIO estimator. Yhat = sum_h (ybar_h / xbar_h) * X_h, where x
#   is the MODEL count, known for every tile in the population, and X_h its
#   stratum total. Because verified and model counts correlate strongly, this
#   is far more precise for the same labelling effort. It is the one to report,
#   with the stratified mean alongside as a check: if the two disagree beyond
#   their intervals, the correlation assumption is failing and the strata need
#   revisiting.
#
#   Both are reported. Neither is a model metric; both are properties of the
#   sample design, which is why they survive review.
#
# WHAT YOU MUST DO BETWEEN THE TWO STEPS
#   `sample` writes tiles and a manifest. A human then counts EVERY palm in
#   each sampled tile, honestly, including ones the model missed. That last
#   part is the whole point: counting only what the model found measures
#   precision and cannot see false negatives, so the correction would be
#   one-sided and the total biased low.
#
# USAGE
#   # 1. draw the sample (before or during the country run)
#   python configs/Custom/Finetune_HN/validation_sample.py sample \
#       --images /workspace/datasets/GE15cm \
#       --detections /workspace/results/uae_palms \
#       --n-tiles 400 --tile 1024 \
#       --out /workspace/work_dirs/validation_sample
#
#   # 2. verify by hand in QGIS, fill the `verified_count` column, then
#   python configs/Custom/Finetune_HN/validation_sample.py estimate \
#       --manifest /workspace/work_dirs/validation_sample/sample_manifest.csv \
#       --out /workspace/work_dirs/validation_sample
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RASTER_EXTS = ('.tif', '.tiff', '.img', '.jp2', '.vrt')
DET_EXTS = ('.gpkg', '.shp', '.geojson')

# Cut points on MODEL detections per tile. They define the strata, and the
# estimator is unbiased whatever they are -- a bad choice costs precision, not
# validity. These follow the density classes the inventory reports in, so the
# sample doubles as a per-class accuracy statement.
STRATA = [
    ('empty',   0,      0),        # no detections: where false positives live
    ('sparse',  1,     10),        # scattered trees, desert margin, roadside
    ('medium',  11,   100),        # small farms
    ('dense',   101, 10 ** 9),     # plantations, where truncation risk lives
]


def find_rasters(path: Path):
    if not path.exists():
        sys.exit(f'[FATAL] --images does not exist: {path}')
    out = ([path] if path.suffix.lower() in RASTER_EXTS
           else sorted(p for p in path.rglob('*')
                       if p.suffix.lower() in RASTER_EXTS))
    if not out:
        sys.exit(f'[FATAL] no rasters under {path}')
    return out


def stratum_of(n):
    for name, lo, hi in STRATA:
        if lo <= n <= hi:
            return name
    return STRATA[-1][0]


def cmd_sample(args):
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window
    from shapely.geometry import box
    from shapely.strtree import STRtree

    out = Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tile = args.tile

    det_root = Path(args.detections)
    det_files = ([det_root] if det_root.suffix.lower() in DET_EXTS
                 else sorted(p for p in det_root.rglob('*')
                             if p.suffix.lower() in DET_EXTS))
    if not det_files:
        sys.exit(f'[FATAL] no detection files under {det_root}')

    # ---- census EVERY tile: the population must be enumerated, not sampled --
    # A stratified estimator needs N_h, the true number of tiles in each
    # stratum, and X_h, the model total in each. Both come from a full pass
    # over the grid; only the VERIFICATION is sampled.
    rows = []
    for rp in find_rasters(Path(args.images)):
        with rasterio.open(rp) as src:
            hit = [d for d in det_files if rp.stem in d.stem or d.stem in rp.stem]
            if not hit:
                print(f'  [warn] no detections matched {rp.stem}; '
                      f'its tiles are counted as empty')
                pts, tree = None, None
            else:
                g = gpd.read_file(hit[0])
                if g.crs != src.crs:
                    g = g.to_crs(src.crs)
                pts = np.array([(p.x, p.y) for p in g.geometry.representative_point()])
                tree = STRtree([box(x, y, x, y) for x, y in pts]) if len(pts) else None
                print(f'  [det] {hit[0].name}: {len(g)} detection(s)')

            n_x, n_y = src.width // tile, src.height // tile
            for iy in range(n_y):
                for ix in range(n_x):
                    x, y = ix * tile, iy * tile
                    w = Window(x, y, tile, tile)
                    b = rasterio.windows.bounds(w, src.transform)
                    n_det = (len(tree.query(box(*b))) if tree is not None else 0)
                    rows.append(dict(raster=str(rp), tile_x=x, tile_y=y,
                                     minx=b[0], miny=b[1], maxx=b[2], maxy=b[3],
                                     crs=str(src.crs), model_count=int(n_det)))
            print(f'  {rp.name}: {n_x * n_y} tiles')

    pop = pd.DataFrame(rows)
    pop['stratum'] = pop['model_count'].map(stratum_of)
    pop.to_parquet(out / 'population_tiles.parquet') if args.parquet else \
        pop.to_csv(out / 'population_tiles.csv', index=False)

    # ---- allocation ------------------------------------------------------
    # Neyman allocation puts effort where the variance is, which is the dense
    # strata. A floor per stratum keeps the rare ones estimable at all: a
    # stratum with two sampled tiles has no usable variance, so its interval
    # would be meaningless however small it is.
    summ = pop.groupby('stratum').agg(N_h=('model_count', 'size'),
                                      X_h=('model_count', 'sum'),
                                      sd_h=('model_count', 'std')).fillna(0.0)
    summ['sd_h'] = summ['sd_h'].replace(0.0, 1.0)
    wts = summ['N_h'] * summ['sd_h']
    alloc = (wts / wts.sum() * args.n_tiles).round().astype(int)
    alloc = alloc.clip(lower=args.min_per_stratum)
    alloc = alloc.clip(upper=summ['N_h'])
    summ['n_h'] = alloc

    print('\nstratum      N_h        X_h    n_h')
    for s, r in summ.iterrows():
        print(f'  {s:<10s} {int(r.N_h):>8d} {int(r.X_h):>10d} {int(r.n_h):>6d}')
    print(f'  total      {int(summ.N_h.sum()):>8d} {int(summ.X_h.sum()):>10d} '
          f'{int(summ.n_h.sum()):>6d}')

    # ---- draw and cut ----------------------------------------------------
    picks = []
    for s, r in summ.iterrows():
        sub = pop[pop['stratum'] == s]
        idx = rng.choice(len(sub), size=int(r.n_h), replace=False)
        picks.append(sub.iloc[idx])
    sample = pd.concat(picks, ignore_index=True)

    if not args.no_tiles:
        from PIL import Image
        for i, r in sample.iterrows():
            with rasterio.open(r['raster']) as src:
                bands = [1, 2, 3] if src.count >= 3 else [1]
                a = src.read(bands, window=Window(r['tile_x'], r['tile_y'],
                                                  tile, tile),
                             boundless=True, fill_value=0)
            arr = np.transpose(a, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            name = (f"{Path(r['raster']).stem}_x{int(r['tile_x']):06d}"
                    f"_y{int(r['tile_y']):06d}.jpg")
            Image.fromarray(arr.astype(np.uint8)).save(out / 'images' / name,
                                                       quality=95)
            sample.loc[i, 'file_name'] = name
            if (i + 1) % 50 == 0:
                print(f'    cut {i + 1}/{len(sample)}')

    # verified_count is left EMPTY on purpose. It is the human's job, and a
    # pre-filled column invites confirming the model instead of counting.
    sample['verified_count'] = ''
    sample['verifier'] = ''
    sample['notes'] = ''
    sample.to_csv(out / 'sample_manifest.csv', index=False)
    summ.reset_index().to_csv(out / 'strata_summary.csv', index=False)

    if args.footprints:
        gdf = gpd.GeoDataFrame(
            sample.drop(columns=['minx', 'miny', 'maxx', 'maxy']),
            geometry=[box(r.minx, r.miny, r.maxx, r.maxy)
                      for r in sample.itertuples()],
            crs=sample['crs'].iloc[0])
        gdf.to_file(out / 'sample_footprints.gpkg', driver='GPKG')

    json.dump(dict(created=datetime.now(timezone.utc).isoformat(),
                   tile=tile, seed=args.seed, n_tiles=args.n_tiles,
                   strata=[dict(name=n, lo=lo, hi=hi) for n, lo, hi in STRATA],
                   allocation='Neyman, floored at --min-per-stratum'),
              open(out / 'sample_provenance.json', 'w'), indent=2)

    print(f'\nsample -> {out / "sample_manifest.csv"}')
    print(f'tiles  -> {out / "images"}')
    print('\nNEXT: count EVERY palm in each tile by hand, including ones the '
          'model missed, and put the number in `verified_count`. Counting '
          'only what the model found measures precision and cannot see false '
          'negatives, which biases the corrected total low.')


def cmd_estimate(args):
    m = pd.read_csv(args.manifest)
    m = m[pd.to_numeric(m['verified_count'], errors='coerce').notna()].copy()
    if m.empty:
        sys.exit('[FATAL] no rows have a verified_count. Fill it in first.')
    m['verified_count'] = m['verified_count'].astype(float)
    strata = pd.read_csv(Path(args.manifest).parent / 'strata_summary.csv') \
        .set_index('stratum')

    rows = []
    for s, g in m.groupby('stratum'):
        N_h, X_h = float(strata.loc[s, 'N_h']), float(strata.loc[s, 'X_h'])
        n_h = len(g)
        y, x = g['verified_count'].values, g['model_count'].values.astype(float)
        ybar, xbar = y.mean(), x.mean()
        s2y = y.var(ddof=1) if n_h > 1 else 0.0
        fpc = max(0.0, 1.0 - n_h / N_h)

        # simple stratified
        tot_srs = N_h * ybar
        var_srs = N_h ** 2 * fpc * s2y / max(n_h, 1)

        # Ratio: uses the model count, known for every tile in the
        # population. It is DEGENERATE where the model found nothing: X_h = 0
        # forces the stratum total to zero, so every palm the model missed
        # entirely would be estimated as not existing. That is exactly the
        # error a national count must not make, and it hides in a column that
        # otherwise looks like the more precise estimator. Where the auxiliary
        # variable carries no information, fall back to the stratified mean
        # for that stratum only.
        if xbar > 0 and X_h > 0:
            R = ybar / xbar
            tot_rat = R * X_h
            resid = y - R * x
            s2r = resid.var(ddof=1) if n_h > 1 else 0.0
            var_rat = N_h ** 2 * fpc * s2r / max(n_h, 1)
            est_kind = 'ratio'
        else:
            R = float('nan')
            tot_rat, var_rat = tot_srs, var_srs
            est_kind = 'mean (model count is zero here)'

        rows.append(dict(stratum=s, N_h=N_h, n_h=n_h, X_h=X_h,
                         mean_verified=ybar, mean_model=xbar, ratio=R,
                         total_srs=tot_srs, var_srs=var_srs,
                         total_ratio=tot_rat, var_ratio=var_rat,
                         estimator=est_kind))
    t = pd.DataFrame(rows)
    t.to_csv(Path(args.out) / 'estimate_by_stratum.csv', index=False)

    def ci(tot, var):
        se = float(np.sqrt(var))
        return tot, se, tot - 1.96 * se, tot + 1.96 * se

    T_s, se_s, lo_s, hi_s = ci(t['total_srs'].sum(), t['var_srs'].sum())
    T_r, se_r, lo_r, hi_r = ci(t['total_ratio'].sum(), t['var_ratio'].sum())
    model_total = float(t['X_h'].sum())

    print(t.to_string(index=False, float_format=lambda v: f'{v:,.2f}'))
    print(f'\nmodel count (all tiles)      {model_total:>15,.0f}')
    print(f'stratified mean estimate     {T_s:>15,.0f}  '
          f'95% CI [{lo_s:,.0f}, {hi_s:,.0f}]  +/- {1.96*se_s/T_s*100:.1f}%')
    print(f'stratified RATIO estimate    {T_r:>15,.0f}  '
          f'95% CI [{lo_r:,.0f}, {hi_r:,.0f}]  +/- {1.96*se_r/T_r*100:.1f}%')
    print(f'\nmodel bias vs ratio estimate {model_total - T_r:>+15,.0f} '
          f'({(model_total / T_r - 1) * 100:+.1f}%)')

    # Disagreement between the two is diagnostic, not cosmetic: the ratio
    # estimator assumes verified and model counts are proportional within a
    # stratum. If the intervals do not overlap, that assumption is failing.
    if hi_r < lo_s or hi_s < lo_r:
        print('\n[warn] the two estimates DISAGREE beyond their intervals. The '
              'ratio estimator assumes verified and model counts are '
              'proportional within each stratum; that is failing somewhere. '
              'Inspect per-stratum ratios before reporting either number.')
    else:
        print('\nThe two estimates agree within their intervals. Report the '
              'ratio estimate and its CI; cite the stratified mean as a '
              'design check.')
    print(f'\ntable -> {Path(args.out) / "estimate_by_stratum.csv"}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('sample')
    s.add_argument('--images', required=True)
    s.add_argument('--detections', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--tile', type=int, default=1024)
    s.add_argument('--n-tiles', type=int, default=400)
    s.add_argument('--min-per-stratum', type=int, default=40,
                   help='floor per stratum. Below about 30 the within-stratum '
                        'variance is not estimable and the interval is '
                        'meaningless however small it looks.')
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--no-tiles', action='store_true',
                   help='write the manifest only (plan the sample first)')
    s.add_argument('--footprints', action='store_true', default=True)
    s.add_argument('--parquet', action='store_true')
    s.set_defaults(func=cmd_sample)

    e = sub.add_parser('estimate')
    e.add_argument('--manifest', required=True)
    e.add_argument('--out', required=True)
    e.set_defaults(func=cmd_estimate)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
