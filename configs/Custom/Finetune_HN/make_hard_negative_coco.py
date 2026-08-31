#!/usr/bin/env python3
# =============================================================================
# make_hard_negative_coco.py
# -----------------------------------------------------------------------------
# Build a valid COCO instance-segmentation dataset of HARD-NEGATIVE (unlabeled)
# tiles for fine-tuning a Mask R-CNN date-palm model against its own false
# positives — desert shrubs and native trees (ghaf, acacia) that the Stage C
# training corpus never contained as labelled background.
#
# WHY THIS IS NEEDED
# ------------------
# "You cannot train instance segmentation with background" is a misconception
# rooted in one config line, filter_empty_gt=True, which SKIPS empty tiles.
# Mask R-CNN trains on empty images natively: on a tile with zero annotations
# every RPN anchor and every RoI proposal is an unmatched NEGATIVE, so the tile
# is pure "nothing here is a palm" supervision for the two classification
# heads; box/mask losses receive no positives and are inert. An empty-
# annotation COCO file is fully valid — labelme2coco is unnecessary for
# negatives because there are no polygons.
#
# WHAT THIS SCRIPT DOES  (three input modes)
# ------------------------------------------
#   --from-images  DIR|TIF   (recommended for your workflow)
#       Take whole selected images (e.g. tens of 1x1 km GE tiffs over desert /
#       struggle areas) and tile each into non-overlapping TILE_SIZE tiles.
#       Skips tiles that are mostly nodata. Produces the negative tiles + the
#       empty-annotation COCO in one step. No shapefile, no labelme2coco.
#
#   --from-tiles   DIR
#       Register tiles you already cut (e.g. via your own tiler).
#
#   --from-detections  GPKG  --imagery ROOT
#       Cut tiles centred on detections you flagged is_fp=1 in QGIS.
#
# ─────────────────────────────────────────────────────────────────────────────
# RADIOMETRY — READ THIS (it decides whether the fine-tune works)
# ─────────────────────────────────────────────────────────────────────────────
# The negative tiles MUST match the radiometry of the POSITIVE tiles the model
# is fine-tuned against (GE_train), or the model learns a shortcut ("processed
# one way = palm domain, the other = negative domain") instead of shrub
# rejection.
#
# For the GE 15 cm workflow: GE_train tiles were NOT contrast-stretched (the
# tiling pipeline's p2/p98 stretch was applied to WorldView-3 only), and the
# country inference pipeline also does NOT stretch uint8 GE imagery. Training,
# negatives, and inference are all raw uint8 — consistent. So the DEFAULT here
# is --stretch none (raw passthrough). Do NOT stretch GE negatives.
#
# --stretch match-train (per-image p2/p98) exists only for a WorldView-3-style
# workflow whose positive tiles WERE stretched. Use it only if your positive
# training tiles were stretched the same way; otherwise it creates a mismatch.
# =============================================================================

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds, transform as win_transform


CATEGORIES = [{'id': 1, 'name': 'DatePalm', 'supercategory': 'tree'}]
TILE_SIZE = 1024               # match the model's training tile size


# ---------------------------------------------------------------------------
# Radiometry helpers — replicate the training tiler's per-image p2/p98 stretch
# ---------------------------------------------------------------------------
def image_stretch(src, bands, lower=2, upper=98, max_dim=2048):
    """Per-band (lo, hi) from a decimated overview of the whole image, matching
    the training tiler (compute once per image, apply to every tile). Nodata
    (value 0) excluded from the percentiles."""
    scale = max(1, int(max(src.width, src.height) / max_dim))
    arr = src.read(bands, out_shape=(len(bands),
                                     max(1, src.height // scale),
                                     max(1, src.width // scale)))
    out = []
    for c in range(arr.shape[0]):
        b = arr[c].astype(np.float32)
        valid = b[b > 0]
        if valid.size == 0:
            out.append((0.0, 255.0))
            continue
        lo, hi = np.percentile(valid, (lower, upper))
        if hi - lo < 1:
            hi = lo + 1
        out.append((float(lo), float(hi)))
    return out


def to_uint8(data, stretch):
    """uint8 passthrough, or apply the given per-band (lo, hi) stretch. Nodata
    (0) is preserved as 0 (same as the training tiler)."""
    if stretch is None:
        if data.dtype == np.uint8:
            return data
        stretch = [(float(np.percentile(data[c][data[c] > 0], 2))
                    if (data[c] > 0).any() else 0.0,
                    float(np.percentile(data[c][data[c] > 0], 98))
                    if (data[c] > 0).any() else 1.0)
                   for c in range(data.shape[0])]
    out = np.zeros(data.shape, dtype=np.uint8)
    for c, (lo, hi) in enumerate(stretch):
        b = data[c].astype(np.float32)
        zero = (b == 0)
        s = np.clip((b - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255)
        s[zero] = 0
        out[c] = s.astype(np.uint8)
    return out


def _write_u8_tile(u8, out_path, crs, transform):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver='GTiff', height=u8.shape[1], width=u8.shape[2],
                   count=u8.shape[0], dtype='uint8', crs=crs,
                   transform=transform, compress='lzw', tiled=True,
                   blockxsize=256, blockysize=256)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(u8)


# ---------------------------------------------------------------------------
# COCO emit (empty annotations)
# ---------------------------------------------------------------------------
def build_coco(image_files, out_json):
    images = [{'id': i, 'file_name': fn, 'width': w, 'height': h}
              for i, (fn, w, h) in enumerate(image_files, 1)]
    coco = {
        'info': {'description': 'Hard-negative tiles (empty annotations) for '
                                'date-palm false-positive suppression',
                 'date_created': datetime.now(timezone.utc).isoformat()},
        'licenses': [], 'images': images, 'annotations': [],
        'categories': CATEGORIES,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(coco, f)
    return len(images)


def _summary(out, n, note=''):
    print('=' * 64)
    print(f'Hard-negative COCO written: {n} tiles  {note}')
    print(f'  images : {out / "images"}')
    print(f'  coco   : {out / "annotations" / "hard_neg.json"}')
    print(f'  (annotations = [], categories = DatePalm id 1)')
    if n == 0:
        print('  WARNING: 0 tiles written — check inputs / coverage threshold.')
    else:
        print('Next: point HN_ROOT in the fine-tune config at this directory.')


# ---------------------------------------------------------------------------
# Mode A: --from-images  (tile whole images into negatives)
# ---------------------------------------------------------------------------
def from_images(args):
    out = Path(args.out)
    img_out = out / 'images'
    inp = Path(args.from_images)
    if inp.is_dir():
        tiffs = sorted(p for p in inp.rglob('*')
                       if p.suffix.lower() in ('.tif', '.tiff'))
    elif inp.suffix.lower() in ('.tif', '.tiff'):
        tiffs = [inp]
    else:
        raise SystemExit(f'--from-images must be a folder or a .tif: {inp}')
    if not tiffs:
        raise SystemExit(f'No .tif images under {inp}')
    print(f'{len(tiffs)} source image(s) to tile at {args.tile} px '
          f'(stretch={args.stretch}, min-coverage={args.min_coverage}).')

    ts = int(args.tile)
    written, idx = [], 0
    for tp in tiffs:
        with rasterio.open(tp) as src:
            bands = [1, 2, 3] if src.count >= 3 else \
                list(range(1, src.count + 1))
            stretch = image_stretch(src, bands) \
                if args.stretch == 'match-train' else None
            n_before = idx
            for y in range(0, src.height, ts):
                for x in range(0, src.width, ts):
                    data = src.read(bands, window=Window(x, y, ts, ts),
                                    boundless=True, fill_value=0)
                    cov = np.count_nonzero(data.any(axis=0)) / float(ts * ts)
                    if cov < args.min_coverage:
                        continue
                    u8 = to_uint8(data, stretch)
                    if u8.max() == 0:
                        continue
                    fname = f'hardneg_{idx:06d}.tif'
                    tfm = win_transform(Window(x, y, ts, ts), src.transform)
                    _write_u8_tile(u8, img_out / fname, src.crs, tfm)
                    written.append((fname, ts, ts))
                    idx += 1
            print(f'  {tp.name}: +{idx - n_before} tiles')
    n = build_coco(written, out / 'annotations' / 'hard_neg.json')
    _summary(out, n, note=f'from {len(tiffs)} image(s)')


# ---------------------------------------------------------------------------
# Mode B: --from-tiles  (register pre-cut tiles)
# ---------------------------------------------------------------------------
def from_tiles(args):
    out = Path(args.out)
    img_out = out / 'images'
    img_out.mkdir(parents=True, exist_ok=True)
    tifs = sorted(p for p in Path(args.from_tiles).rglob('*')
                  if p.suffix.lower() in ('.tif', '.tiff'))
    if not tifs:
        raise SystemExit(f'No .tif tiles under {args.from_tiles}')
    written = []
    for p in tifs:
        with rasterio.open(p) as src:
            w, h = src.width, src.height
        dst = img_out / p.name
        if not dst.exists():
            try:
                os.link(p, dst)
            except OSError:
                import shutil
                shutil.copy2(p, dst)
        written.append((p.name, w, h))
    n = build_coco(written, out / 'annotations' / 'hard_neg.json')
    _summary(out, n)


# ---------------------------------------------------------------------------
# Mode C: --from-detections  (cut tiles at confirmed false positives)
# ---------------------------------------------------------------------------
def from_detections(args):
    import geopandas as gpd
    out = Path(args.out)
    img_out = out / 'images'

    exts = ('.tif', '.tiff', '.img', '.jp2', '.vrt')
    entries = []
    for p in sorted(Path(args.imagery).rglob('*')):
        if p.suffix.lower() not in exts:
            continue
        try:
            with rasterio.open(p) as src:
                entries.append((p, src.crs, tuple(src.bounds)))
        except Exception:
            continue
    if not entries:
        raise SystemExit(f'No rasters under {args.imagery}')

    g = gpd.read_file(args.from_detections)
    if args.fp_column not in g.columns:
        raise SystemExit(
            f"Column '{args.fp_column}' not in the GeoPackage. In QGIS add an "
            f"integer field '{args.fp_column}', set 1 for detections you "
            f"confirm are NOT palms, save, re-run. Columns: {list(g.columns)}")
    fps = g[g[args.fp_column].fillna(0).astype(int) == 1].copy()
    if fps.empty:
        raise SystemExit(f'No rows with {args.fp_column}=1.')
    print(f'{len(fps)} confirmed false positive(s).')

    cen = fps.geometry.centroid
    written = []
    for i, (x, y) in enumerate(zip(cen.x.values, cen.y.values)):
        raster = next((p for p, rcrs, (mnx, mny, mxx, mxy) in entries
                       if rcrs == fps.crs and mnx <= x <= mxx and mny <= y <= mxy),
                      None)
        if raster is None:
            continue
        with rasterio.open(raster) as src:
            bands = [1, 2, 3] if src.count >= 3 else \
                list(range(1, src.count + 1))
            gsd = abs(src.transform.a)
            half = TILE_SIZE / 2.0 * gsd
            win = from_bounds(x - half, y - half, x + half, y + half,
                              src.transform)
            data = src.read(bands, window=win, boundless=True, fill_value=0,
                            out_shape=(len(bands), TILE_SIZE, TILE_SIZE))
            stretch = None if args.stretch == 'none' else \
                image_stretch(src, bands)
            u8 = to_uint8(data, stretch)
            if u8.max() == 0:
                continue
            fname = f'hardneg_{i:06d}.tif'
            tfm = rasterio.windows.transform(win, src.transform)
            _write_u8_tile(u8, img_out / fname, src.crs, tfm)
            written.append((fname, TILE_SIZE, TILE_SIZE))
    n = build_coco(written, out / 'annotations' / 'hard_neg.json')
    _summary(out, n)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--from-images', metavar='DIR|TIF',
                     help='whole images to tile into negatives (recommended)')
    src.add_argument('--from-tiles', metavar='DIR',
                     help='folder of already-cut negative tiles')
    src.add_argument('--from-detections', metavar='GPKG',
                     help='detections GeoPackage with confirmed FPs')
    ap.add_argument('--imagery', help='imagery root (with --from-detections)')
    ap.add_argument('--fp-column', default='is_fp',
                    help="integer column, 1 = confirmed not-a-palm")
    ap.add_argument('--tile', type=int, default=TILE_SIZE,
                    help=f'tile size for --from-images (default {TILE_SIZE})')
    ap.add_argument('--min-coverage', type=float, default=0.5,
                    help='drop tiles with less than this fraction of valid '
                         'pixels (default 0.5)')
    ap.add_argument('--stretch', choices=('match-train', 'none'),
                    default='none',
                    help="'none' = raw uint8 passthrough (matches GE_train and "
                         "GE inference; DEFAULT, use this for GE). "
                         "'match-train' = per-image p2/p98, only for a "
                         "WorldView-3-style workflow whose positives were "
                         "stretched.")
    ap.add_argument('--out', required=True, help='output dataset directory')
    args = ap.parse_args()

    if args.from_images:
        from_images(args)
    elif args.from_tiles:
        from_tiles(args)
    else:
        if not args.imagery:
            ap.error('--imagery is required with --from-detections')
        from_detections(args)


if __name__ == '__main__':
    main()
