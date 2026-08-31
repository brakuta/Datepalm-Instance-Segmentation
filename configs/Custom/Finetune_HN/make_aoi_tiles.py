#!/usr/bin/env python3
# =============================================================================
# make_aoi_tiles.py
# -----------------------------------------------------------------------------
# Cut 1024 px training tiles from full images, RESTRICTED to areas of interest
# (AOIs) given as shapefile extent polygons — the sample-generation step for
# FALSE-NEGATIVE (missed-palm) hard-example fine-tuning.
#
# WHY THIS EXISTS (and why it is not make_hard_negative_coco.py)
# -------------------------------------------------------------------------
#   make_hard_negative_coco.py fixes FALSE POSITIVES: it writes tiles with
#   ZERO annotations, so every RoI proposal becomes a negative and the
#   classifier learns "shrub is not palm". Nothing needs to be digitised.
#
#   FALSE NEGATIVES are the opposite problem. A missed palm cannot be taught
#   with an empty tile — an unlabelled palm actively teaches the model that
#   palms are background, i.e. it makes recall WORSE. Fixing recall requires
#   LABELLED POSITIVES from the exact regimes that fail. So this script only
#   produces tiles + (optionally) pre-seeded LabelMe sidecars; you then correct
#   the labels in LabelMe and run the project's usual labelme2coco.
#
# WHAT IT DOES
# -------------------------------------------------------------------------
#   1. Pairs each raster with its AOI polygons (per-image shapefile, or one
#      shapefile for everything), reprojecting the AOI into the raster CRS.
#   2. Lays a tile grid ANCHORED TO THE RASTER PIXEL GRID (integer windows,
#      no resampling => GSD, radiometry and crown scale are bit-identical to
#      the imagery the model was trained on) and keeps only tiles that fall
#      sufficiently inside the AOI and contain enough valid pixels.
#   3. Writes georeferenced uint8 GeoTIFF tiles, a tile-footprint GeoPackage
#      and a manifest CSV for QA in QGIS, and a provenance JSON.
#   4. OPTIONAL --seed-labels: clips your existing predicted crown polygons
#      (the .gpkg from palm_inference_pipeline.py) into per-tile LabelMe JSON
#      sidecars, so annotation becomes CORRECTION (add the missed crowns,
#      delete the wrong ones) instead of digitising from scratch. This is
#      typically a 5-10x speed-up and is the recommended route.
#   5. OPTIONAL --val-frac: holds out WHOLE AOIs (never individual tiles) as
#      a validation split, so adjacent-tile spatial leakage cannot inflate
#      the recall improvement you are trying to measure.
#
# RADIOMETRY — same rule as the hard-negative tiler
# -------------------------------------------------------------------------
#   GE_train tiles were NOT contrast-stretched, and the GE inference pipeline
#   does not stretch uint8 imagery either. So the DEFAULT is --stretch none
#   (raw uint8 passthrough): training tiles, new AOI tiles and inference all
#   agree. Only use --stretch match-train for a WorldView-3-style workflow
#   whose positive tiles were themselves stretched.
#
# REQUIREMENTS
#   rasterio, geopandas, shapely  (all already used by make_hard_negative_coco)
#
# USAGE
# -------------------------------------------------------------------------
#   # one shapefile per image, matched by filename stem, seeded from your
#   # existing predictions, 20% of AOIs held out for validation:
#   python configs/Custom/Finetune_HN/make_aoi_tiles.py \
#       --images   /workspace/datasets/GE15cm/struggle_areas \
#       --aoi      /workspace/datasets/GE15cm/struggle_areas/extents \
#       --out      /workspace/datasets/COCO/HardPos_GE \
#       --seed-labels /workspace/results/uae_palms/UAE_palms_master.gpkg \
#       --tile 1024 --overlap 0.25 --val-frac 0.2
#
#   # single shapefile covering all images, no seeding (digitise from scratch):
#   python configs/Custom/Finetune_HN/make_aoi_tiles.py \
#       --images /path/to/tiffs --aoi /path/to/all_extents.shp \
#       --out /workspace/datasets/COCO/HardPos_GE --tile 1024 --overlap 0.25
#
# NEXT STEPS after this script (see README_false_negative_finetune.md)
#   1. annotate  <out>/images_train  and  <out>/images_val  in LabelMe
#   2. labelme2coco  ->  <out>/annotations/train_hardpos.json , val_hardpos.json
#   3. train with maskrcnn_palm_finetune_hn/maskrcnn_spatialmamba_s_finetune_fn.py
# =============================================================================

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, transform as win_transform

# Reuse the VALIDATED radiometry helpers from the hard-negative tiler so AOI
# tiles and HN tiles are produced by identical code paths.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hard_negative_coco import (image_stretch, to_uint8,   # noqa: E402
                                     _write_u8_tile, build_coco)

RASTER_EXTS = ('.tif', '.tiff', '.img', '.jp2', '.vrt')
VECTOR_EXTS = ('.shp', '.gpkg', '.geojson', '.json', '.kml')
# Detection folders live beside sidecar .json/.log files, so discovery there is
# restricted to formats that are unambiguously vector data.
DET_EXTS = ('.gpkg', '.shp', '.geojson')
LABEL_NAME = 'DatePalm'


# ---------------------------------------------------------------------------
# CODEC MATCHING — read this before choosing --format
# ---------------------------------------------------------------------------
# The GE training corpus (COCO/GE_15cm/train_GE/JPEGImages) is JPEG. JPEG
# leaves characteristic 8x8 DCT blocking and chroma-subsampling artefacts in
# every tile. If new tiles are written as lossless GeoTIFF and then mixed with
# JPEG positives in the same fine-tune, the classifier can separate the two
# sources by CODEC ARTEFACT alone -- "clean image = not a palm" -- instead of
# learning the actual content distinction. That is the same shortcut failure
# the radiometry section guards against, arriving through the encoder rather
# than through a contrast stretch.
#
# So when the corpus you are mixing against is JPEG, write JPEG:
#     --format jpg --jpeg-ref <one training .jpg>
# --jpeg-ref copies the reference's exact quantisation tables and subsampling,
# so the new tiles carry the same artefact signature as the corpus rather than
# a guessed quality setting. Georeferencing is preserved alongside the .jpg as
# a world file (.jgw) + .prj, and in tile_footprints.gpkg / tile_manifest.csv.
# ---------------------------------------------------------------------------
def read_jpeg_profile(ref_path):
    """Extract quantisation tables + subsampling from a reference JPEG."""
    from PIL import Image, JpegImagePlugin
    with Image.open(ref_path) as im:
        if im.format != 'JPEG':
            raise SystemExit(f'--jpeg-ref must be a JPEG file, got '
                             f'{im.format}: {ref_path}')
        qtables = getattr(im, 'quantization', None)
        try:
            subsampling = JpegImagePlugin.get_sampling(im)
        except Exception:                                       # noqa: BLE001
            subsampling = -1
    if not qtables:
        raise SystemExit(f'no quantisation tables readable from {ref_path}')
    return qtables, subsampling


def _write_jpeg_tile(u8, out_path, crs, transform, qtables=None,
                     subsampling=-1, quality=95):
    """Write an RGB JPEG tile plus .jgw/.prj so it stays georeferenced."""
    from PIL import Image
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = u8[:3] if u8.shape[0] >= 3 else np.repeat(u8[:1], 3, axis=0)
    im = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode='RGB')
    if qtables is not None:
        im.save(out_path, format='JPEG', qtables=qtables,
                subsampling=subsampling)
    else:
        im.save(out_path, format='JPEG', quality=quality, subsampling=0)

    # world file: pixel size and centre of the upper-left pixel
    a, b, c, d, e, f = (transform.a, transform.b, transform.c,
                        transform.d, transform.e, transform.f)
    with open(out_path.with_suffix('.jgw'), 'w') as wf:
        wf.write(f'{a}\n{d}\n{b}\n{e}\n{c + a / 2.0}\n{f + e / 2.0}\n')
    if crs is not None:
        try:
            with open(out_path.with_suffix('.prj'), 'w') as pf:
                pf.write(crs.to_wkt())
        except Exception:                                       # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# discovery / pairing
# ---------------------------------------------------------------------------
def find_rasters(path: Path) -> list[Path]:
    if str(path) in ('', '.'):
        raise SystemExit(
            '--images is empty. If you used a shell variable, it is unset: '
            'check with `echo "$IMAGES"` and assign the real path.')
    if not path.exists():
        raise SystemExit(f'--images does not exist: {path}')
    if path.is_dir():
        out = sorted(p for p in path.rglob('*')
                     if p.suffix.lower() in RASTER_EXTS)
    elif path.suffix.lower() in RASTER_EXTS:
        out = [path]
    else:
        raise SystemExit(f'--images must be a folder or a raster: {path}')
    if not out:
        raise SystemExit(f'No rasters found under {path}')
    return out


def ext_of(args) -> str:
    """The output extension, needed by the exclusion filter before the write
    loop computes it. Kept as one function so the two cannot diverge."""
    return '.jpg' if args.format == 'jpg' else '.tif'


def find_vectors(path: Path) -> list[Path]:
    """AOI discovery.

    Two guards, both learned the hard way. An empty --aoi (an unset shell
    variable expands to '') becomes Path('') == Path('.'), which sent an
    rglob over the entire repository and reported 277 "AOI files", nearly all
    of them sidecar .json that pyogrio then failed to open one by one. And a
    bare .json is almost never AOI data next to a project tree full of config
    and log files, so directory scans use the unambiguous formats only. An
    explicitly named .json is still honoured, because naming a file is a
    statement of intent that a directory scan is not.
    """
    if str(path) in ('', '.'):
        raise SystemExit(
            '--aoi is empty. If you used a shell variable, it is unset: '
            'check with `echo "$AOI"` and assign the real path. Refusing to '
            'scan the current directory for vector data.')
    if not path.exists():
        raise SystemExit(f'--aoi does not exist: {path}')
    if path.is_dir():
        scan_exts = tuple(e for e in VECTOR_EXTS if e != '.json')
        out = sorted(p for p in path.rglob('*')
                     if p.suffix.lower() in scan_exts)
        if not out:
            raise SystemExit(
                f'No vector files ({", ".join(scan_exts)}) found under {path}. '
                f'Name the file directly if it is a .json.')
    elif path.suffix.lower() in VECTOR_EXTS:
        out = [path]
    else:
        raise SystemExit(f'--aoi must be a folder or a vector file: {path}')
    return out


def pair_aoi(raster: Path, vectors: list[Path]) -> list[Path]:
    """Match a raster to its AOI file(s).

    Rule: if exactly one vector file was supplied it applies to every raster.
    Otherwise prefer a stem match (image.tif <-> image.shp, or a vector whose
    stem is contained in the image stem or vice versa); if nothing matches,
    fall back to all vectors and let the spatial intersection decide.
    """
    if len(vectors) == 1:
        return vectors
    stem = raster.stem.lower()
    exact = [v for v in vectors if v.stem.lower() == stem]
    if exact:
        return exact
    fuzzy = [v for v in vectors
             if v.stem.lower() in stem or stem in v.stem.lower()]
    return fuzzy if fuzzy else vectors


# ---------------------------------------------------------------------------
# AOI loading
# ---------------------------------------------------------------------------
def load_aoi_features(vector_paths: list[Path], dst_crs, id_field=None):
    """Return [(aoi_id, shapely geometry in dst_crs), ...] for valid polygons."""
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    feats = []
    for vp in vector_paths:
        try:
            g = gpd.read_file(vp)
        except Exception as exc:                                # noqa: BLE001
            print(f'  [warn] cannot read {vp.name}: {exc}')
            continue
        if g.empty:
            continue
        if g.crs is None:
            print(f'  [warn] {vp.name} has no CRS -> assuming it already '
                  f'matches the raster CRS')
        elif dst_crs is not None:
            g = g.to_crs(dst_crs)
        for i, row in enumerate(g.itertuples(index=False)):
            geom = getattr(row, 'geometry', None)
            if geom is None or geom.is_empty:
                continue
            if not isinstance(geom, (Polygon, MultiPolygon)):
                # extents given as lines/points are unusable as areas
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
                if geom.is_empty:
                    continue
            # The AOI id ALWAYS carries the feature index. A label field is
            # only decoration on top of it. Deriving the id from the field
            # alone silently merges every polygon whose attribute is null or
            # duplicated into one AOI, which then shares a single --max-per-aoi
            # budget -- a partly-populated attribute table would quietly cost
            # you most of your tiles.
            label = None
            if id_field is not None:
                v = getattr(row, id_field, None)
                if v is not None and str(v).strip().lower() not in (
                        '', 'none', 'nan', '<na>', 'null'):
                    label = str(v).strip().replace('/', '_').replace(' ', '_')
            aid = f'{vp.stem}-{i:03d}' + (f'-{label}' if label else '')
            feats.append((aid, geom))
    return feats


# ---------------------------------------------------------------------------
# LabelMe sidecars seeded from existing predictions
# ---------------------------------------------------------------------------
def load_seed_index(seed_path: Path, dst_crs, drop_query=None):
    """Load seed crown polygons once per raster CRS, with an STRtree index."""
    import geopandas as gpd
    from shapely.strtree import STRtree

    g = gpd.read_file(seed_path)
    if g.empty:
        return None, None
    if drop_query:
        try:
            g = g.query(drop_query)
            print(f'  [seed] --seed-query kept {len(g)} feature(s)')
        except Exception as exc:                                # noqa: BLE001
            print(f'  [warn] --seed-query failed ({exc}); using all features')
    if g.crs is not None and dst_crs is not None:
        g = g.to_crs(dst_crs)
    geoms = [gm for gm in g.geometry.values if gm is not None and not gm.is_empty]
    if not geoms:
        return None, None
    return geoms, STRtree(geoms)


def tree_hits(tree, geoms, query_geom):
    """STRtree.query compat: shapely >=2 returns indices, 1.8 returns geoms."""
    res = tree.query(query_geom)
    out = []
    for r in res:
        if isinstance(r, (int, np.integer)):
            out.append(geoms[int(r)])
        else:
            out.append(r)
    return out


def polygons_to_labelme_shapes(geoms, tile_tfm, tile_px, min_area_px,
                               simplify_px, label=LABEL_NAME):
    """Map-space polygons -> LabelMe polygon shapes in tile pixel coords."""
    from shapely.geometry import MultiPolygon, Polygon

    inv = ~tile_tfm                      # (x, y) map -> (col, row) pixel
    shapes = []
    for geom in geoms:
        parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for part in parts:
            if not isinstance(part, Polygon) or part.is_empty:
                continue
            pix = [inv * (x, y) for x, y in part.exterior.coords]
            p = Polygon(pix)
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_empty or p.geom_type != 'Polygon':
                continue
            if p.area < min_area_px:
                continue
            if simplify_px > 0:
                p = p.simplify(simplify_px, preserve_topology=True)
                if p.is_empty or p.geom_type != 'Polygon':
                    continue
            pts = []
            for x, y in list(p.exterior.coords)[:-1]:     # drop closing vertex
                x = float(min(max(x, 0.0), tile_px - 1.0))
                y = float(min(max(y, 0.0), tile_px - 1.0))
                pts.append([round(x, 2), round(y, 2)])
            if len(pts) < 3:
                continue
            shapes.append({
                'label': label,
                'points': pts,
                'group_id': None,
                'description': '',
                'shape_type': 'polygon',
                'flags': {},
            })
    return shapes


def write_labelme_json(out_json: Path, image_name: str, tile_px: int, shapes):
    """LabelMe v5 schema, imageData=None so labelme2coco reads pixels from
    the .tif on disk (same convention as resample_gsd.py)."""
    doc = {
        'version': '5.2.1',
        'flags': {},
        'shapes': shapes,
        'imagePath': image_name,
        'imageData': None,
        'imageHeight': tile_px,
        'imageWidth': tile_px,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(doc, f, indent=2)


# ---------------------------------------------------------------------------
# deterministic AOI-level train/val split
# ---------------------------------------------------------------------------
def aoi_split(aoi_ids, val_frac, seed=0):
    """Assign WHOLE AOIs to val. Grouping by AOI (not by tile) is what stops
    an overlapping neighbour tile from leaking into validation."""
    if val_frac <= 0:
        return {a: 'train' for a in aoi_ids}
    ranked = sorted(
        aoi_ids,
        key=lambda a: hashlib.sha256(f'{seed}:{a}'.encode()).hexdigest())
    n_val = max(1, int(round(len(ranked) * val_frac))) if len(ranked) > 1 else 0
    val = set(ranked[:n_val])
    return {a: ('val' if a in val else 'train') for a in aoi_ids}


# ---------------------------------------------------------------------------
# main tiling routine
# ---------------------------------------------------------------------------
def run(args):
    from shapely.geometry import box as shp_box
    from shapely.ops import unary_union

    out = Path(args.out)
    rasters = find_rasters(Path(args.images))
    vectors = find_vectors(Path(args.aoi))
    tile = int(args.tile)
    stride = max(1, int(round(tile * (1.0 - float(args.overlap)))))

    print(f'{len(rasters)} raster(s), {len(vectors)} AOI file(s)')
    print(f'tile={tile}px  overlap={args.overlap:.2f} -> stride={stride}px  '
          f'min-aoi-frac={args.min_aoi_frac}  min-coverage={args.min_coverage}  '
          f'stretch={args.stretch}')

    # ---- pass 1: enumerate candidate tiles (cheap; no pixel reads) --------
    # candidates: (raster, aoi_id, x, y, aoi_frac, n_det)
    # n_det = number of --detections features whose centroid falls in the tile.
    # Inside a PALM-FREE AOI every such detection is a confirmed FALSE POSITIVE,
    # so n_det ranks tiles by how much the model actually gets wrong there.
    # Random tiles in a palm-free polygon are mostly easy bare ground the model
    # never confused; the tiles that teach anything are the ones the model
    # already failed on.
    candidates = []
    seen = set()             # (raster, x, y) -> dedupe overlapping AOIs
    det_cache = {}

    def resolve_det_paths(rp):
        """--detections may be ONE file or a FOLDER of per-image prediction
        files (UAE_245.tif -> UAE_245_palms.gpkg). Per-image pairing matters:
        these files run to hundreds of MB, so loading the whole folder for
        every raster would be both wrong and ruinous for memory."""
        p = Path(args.detections)
        if p.is_file():
            return [p]
        # DET_EXTS, not VECTOR_EXTS: prediction folders sit next to sidecar
        # .json files (UAE_135_stats.json), which match the stem and are not
        # vector data. Bare .json is excluded; .geojson still works.
        cands = sorted(q for q in p.rglob('*')
                       if q.suffix.lower() in DET_EXTS)
        stem = rp.stem.lower()
        hits = [q for q in cands
                if q.stem.lower() == stem or q.stem.lower().startswith(stem + '_')]
        if not hits:      # looser fallback, e.g. palms_UAE_245
            hits = [q for q in cands if stem in q.stem.lower()]
        # one format per image is enough; prefer GeoPackage
        gpkg = [q for q in hits if q.suffix.lower() == '.gpkg']
        return gpkg if gpkg else hits

    def det_query_columns():
        """Read only the attributes --det-query needs; these files carry many
        columns and pulling all of them is the difference between a 60 MB and
        a multi-GB read."""
        if not args.det_query:
            return []
        import keyword
        import re
        names = {n for n in re.findall(r'[A-Za-z_]\w*', args.det_query)
                 if not keyword.iskeyword(n)}
        return sorted(names)

    def det_index_for(rp, crs, bounds):
        if not args.detections:
            return None, None
        paths = resolve_det_paths(rp)
        if not paths:
            print(f'  [det] no prediction file matched {rp.stem} -> '
                  f'0 detections counted for this image')
            return None, None
        key = tuple(str(q) for q in paths)
        if key not in det_cache:
            det_cache.clear()          # hold one image's predictions at a time
            import geopandas as gpd
            import pandas as pd
            from shapely.strtree import STRtree

            cols = det_query_columns()
            frames = []
            for q in paths:
                # Pre-filter by the raster footprint in the LAYER's own CRS so
                # a country-wide file never fully materialises.
                bbox = None
                try:
                    from pyogrio import read_info
                    from rasterio.warp import transform_bounds
                    lc = read_info(q)['crs']
                    if lc and crs is not None:
                        bbox = transform_bounds(crs, lc, *bounds)
                except Exception:                               # noqa: BLE001
                    bbox = None
                try:
                    g = gpd.read_file(q, columns=cols, bbox=bbox)
                except TypeError:        # geopandas without columns= support
                    try:
                        g = gpd.read_file(q, bbox=bbox)
                    except Exception as exc:                    # noqa: BLE001
                        print(f'  [warn] cannot read {q.name}: {exc} -> skipped')
                        continue
                except Exception as exc:                        # noqa: BLE001
                    # An unreadable prediction file must not abort the whole
                    # run: report it and count that image as zero detections.
                    print(f'  [warn] cannot read {q.name}: '
                          f'{type(exc).__name__} -> skipped')
                    continue
                if not g.empty:
                    frames.append(g)
            if not frames:
                det_cache[key] = (None, None)
                return det_cache[key]
            g = frames[0] if len(frames) == 1 else \
                gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                 crs=frames[0].crs)
            n_raw = len(g)
            if args.det_query:
                try:
                    g = g.query(args.det_query)
                except Exception as exc:                        # noqa: BLE001
                    print(f'  [warn] --det-query failed ({exc}); using all. '
                          f'Available columns: {[c for c in g.columns]}')
            if g.crs is not None and crs is not None:
                g = g.to_crs(crs)
            pts = [gm.centroid for gm in g.geometry.values
                   if gm is not None and not gm.is_empty]
            print(f'  [det] {Path(paths[0]).name}: {n_raw} read -> '
                  f'{len(pts)} after filter')
            det_cache[key] = (pts, STRtree(pts)) if pts else (None, None)
        return det_cache[key]

    for rp in rasters:
        try:
            src = rasterio.open(rp)
        except Exception as exc:                                # noqa: BLE001
            print(f'  [warn] cannot open {rp.name}: {exc}')
            continue
        with src:
            feats = load_aoi_features(pair_aoi(rp, vectors), src.crs,
                                      args.aoi_id_field)
            if not feats:
                print(f'  {rp.name}: no usable AOI polygons -> skipped')
                continue
            rbox = shp_box(*src.bounds)
            n_before = len(candidates)
            det_pts, det_tree = det_index_for(rp, src.crs, tuple(src.bounds))
            for aoi_id, geom in feats:
                clipped = geom.intersection(rbox)
                if clipped.is_empty:
                    continue
                # AOI bounds -> integer pixel window, anchored to the raster
                # pixel grid so tiles from overlapping AOIs coincide exactly.
                w = rasterio.windows.from_bounds(*clipped.bounds,
                                                 transform=src.transform)
                c0 = max(0, int(np.floor(w.col_off)))
                r0 = max(0, int(np.floor(w.row_off)))
                c1 = min(src.width, int(np.ceil(w.col_off + w.width)))
                r1 = min(src.height, int(np.ceil(w.row_off + w.height)))
                if c1 <= c0 or r1 <= r0:
                    continue
                gx0 = (c0 // stride) * stride
                gy0 = (r0 // stride) * stride
                for y in range(gy0, r1, stride):
                    for x in range(gx0, c1, stride):
                        key = (str(rp), x, y)
                        if key in seen:
                            continue
                        win = Window(x, y, tile, tile)
                        tb = shp_box(*rasterio.windows.bounds(win,
                                                              src.transform))
                        inter = tb.intersection(clipped)
                        if inter.is_empty:
                            continue
                        frac = inter.area / tb.area
                        if frac < args.min_aoi_frac:
                            continue
                        n_det = 0
                        if det_tree is not None:
                            n_det = sum(
                                1 for p in tree_hits(det_tree, det_pts, tb)
                                if tb.covers(p))
                        if n_det < args.min_detections:
                            continue
                        seen.add(key)
                        candidates.append((rp, aoi_id, x, y, float(frac),
                                           int(n_det)))
            new = candidates[n_before:]
            n_hit = len({c[1] for c in new})
            msg = (f'  {rp.name}: {len(new)} candidate tile(s) '
                   f'from {n_hit}/{len(feats)} AOI polygon(s)')
            if det_tree is not None:
                msg += f'  [{sum(c[5] for c in new)} false positive(s) inside]'
            print(msg)

    if not candidates:
        raise SystemExit(
            'No candidate tiles. Check that (a) the AOI shapefile really '
            'overlaps the imagery (open both in QGIS), (b) --min-aoi-frac is '
            'not too strict for small extents, (c) the AOI CRS is defined.')

    # ---- --dry-run: report the plan, write nothing -------------------------
    # ---- exclude tiles already used elsewhere -----------------------------
    # Tile names are deterministic (<raster>_x<col>_y<row>), and selection
    # ranks by false-positive count, so re-running with a different --seed
    # returns very nearly the SAME tiles: a first attempt at building an
    # evaluation set this way reproduced 1,070 of 1,074 training tiles. A
    # different seed only breaks ties. To obtain a genuinely disjoint set the
    # already-used names must be removed from the candidate pool, and removed
    # BEFORE the per-AOI cap so the cap then selects the next-best unused
    # tiles rather than returning short.
    if args.exclude:
        used = set()
        for spec in args.exclude:
            ep = Path(spec)
            if not ep.exists():
                raise SystemExit(f'--exclude path does not exist: {ep}')
            if ep.is_dir():
                used |= {f.name for f in ep.rglob('*')
                         if f.suffix.lower() in ('.jpg', '.jpeg', '.png',
                                                 '.tif', '.tiff')}
            else:
                used |= {Path(im['file_name']).name for im in
                         json.loads(ep.read_text()).get('images', [])}
        before = len(candidates)
        candidates = [c for c in candidates
                      if f'{c[0].stem}_x{c[2]:06d}_y{c[3]:06d}{ext_of(args)}'
                      not in used]
        print(f'--exclude: {before} -> {len(candidates)} candidates '
              f'({before - len(candidates)} already used, '
              f'{len(used)} name(s) in the exclusion set)')
        if not candidates:
            raise SystemExit(
                'every candidate was excluded. Raise --max-per-aoi or lower '
                '--min-detections so the pool extends beyond what was used.')

    if args.dry_run:
        per_img, per_aoi = {}, {}
        for rp, aoi_id, _x, _y, _f, _n in candidates:
            per_img[rp.name] = per_img.get(rp.name, 0) + 1
            per_aoi[aoi_id] = per_aoi.get(aoi_id, 0) + 1
        print('=' * 70)
        print(f'DRY RUN — {len(candidates)} candidate tile(s) would be cut '
              f'(before the --min-coverage pixel check)')
        print(f'  images with tiles : {len(per_img)} / {len(rasters)}')
        for k in sorted(per_img):
            print(f'    {k:<28s} {per_img[k]:5d}')
        empty = [r.name for r in rasters if r.name not in per_img]
        if empty:
            print(f'  images with NO tiles ({len(empty)}): {", ".join(empty)}')
            print('    -> if that is unexpected, the AOI does not overlap them; '
                  'check CRS and extents in QGIS.')
        print(f'  AOI polygons used : {len(per_aoi)}')
        for k in sorted(per_aoi):
            print(f'    {k:<28s} {per_aoi[k]:5d}')
        if args.detections:
            nd = [c[5] for c in candidates]
            hot_c = [c for c in candidates if c[5] > 0]
            print(f'  false positives   : {sum(nd)} across {len(hot_c)} tile(s) '
                  f'({len(hot_c) / len(nd) * 100:.1f}% of candidates); '
                  f'{len(nd) - len(hot_c)} tile(s) contain none')

            # ---- diversity planning table -------------------------------
            # The question --max-per-aoi actually answers is "how many
            # DISTINCT areas will the negatives come from", which matters far
            # more than raw tile count: 5000 tiles from two images teach less
            # than 1500 tiles from 90 areas. Simulate the caps so the spread
            # is visible before any pixels are written.
            by_aoi = {}
            for c in hot_c:
                by_aoi.setdefault(c[1], []).append(c)
            n_img_hot = len({c[0].name for c in hot_c})
            print(f'  with --min-detections 1: {len(hot_c)} tile(s) '
                  f'across {len(by_aoi)} AOI(s) and {n_img_hot} image(s)')
            print('    cap        tiles   AOIs  images   '
                  'suggested max_iters (at 25% negatives, 1x coverage)')
            for cap in (10, 20, 50, 100, 0):
                sel = []
                for grp in by_aoi.values():
                    g = sorted(grp, key=lambda c: -c[5])
                    sel.extend(g[:cap] if cap else g)
                imgs = len({c[0].name for c in sel})
                aois = len({c[1] for c in sel})
                label = f'--max-per-aoi {cap}' if cap else 'no cap (all)'
                iters = max(1000, -(-len(sel) * 4 // 1000) * 1000)
                print(f'    {label:<18s} {len(sel):6d} {aois:6d} {imgs:7d}   '
                      f'{iters:>7d}')
            print('    (tiles x4 = iterations needed for each tile to be seen '
                  'once at a 25% negative share)')
        print('Re-run without --dry-run to write the tiles.')
        print('=' * 70)
        return

    # ---- per-AOI cap FIRST: diversity beats count -------------------------
    # With many AOIs of very unequal size, a purely global subsample is
    # dominated by the few largest polygons. Capping per AOI first spreads the
    # budget across distinct areas, which is what actually matters for
    # hard-example mining.
    if args.max_per_aoi:
        keyed = {}
        for c in candidates:
            keyed.setdefault(c[1], []).append(c)
        capped = []
        for aoi_id, group in keyed.items():
            # -c[5] first: keep the tiles with the MOST false positives in
            # this AOI, falling back to a deterministic hash for ties (and for
            # the no-detections case, where every n_det is 0 = pure random).
            group.sort(key=lambda c: (-c[5], hashlib.sha256(
                f'{args.seed}:{c[0].name}:{c[2]}:{c[3]}'.encode()).hexdigest()))
            capped.extend(group[:args.max_per_aoi])
        print(f'--max-per-aoi {args.max_per_aoi}: {len(candidates)} -> '
              f'{len(capped)} candidates across {len(keyed)} AOI(s)')
        candidates = capped

    # ---- optional global budget subsample (deterministic) -----------------
    if args.max_tiles and len(candidates) > args.max_tiles:
        candidates.sort(key=lambda c: (-c[5], hashlib.sha256(
            f'{args.seed}:{c[0].name}:{c[1]}:{c[2]}:{c[3]}'.encode()).hexdigest()))
        print(f'--max-tiles {args.max_tiles}: subsampling from '
              f'{len(candidates)} candidates')
        candidates = candidates[:args.max_tiles]

    split_of = aoi_split(sorted({c[1] for c in candidates}),
                         args.val_frac, args.seed)

    # ---- stale output guard ----------------------------------------------
    # Tiles are written INTO the output dir, not over it. A re-run with
    # different selection settings therefore leaves the previous run's tiles
    # behind: they are absent from the new COCO (so training ignores them) but
    # they inflate the on-disk count and make the folder disagree with the
    # manifest. Warn loudly, or clear them with --clean.
    existing = []
    for sub in ('images', 'images_train', 'images_val'):
        d = out / sub
        if d.is_dir():
            existing += [p for p in d.iterdir() if p.is_file()]
    if existing:
        if args.clean:
            for p in existing:
                p.unlink()
            print(f'  [clean] removed {len(existing)} file(s) from a previous '
                  f'run in {out}')
        else:
            print(f'  [WARNING] {out} already holds {len(existing)} file(s) '
                  f'from a previous run. They will NOT be listed in the new '
                  f'COCO, but they will remain on disk and the folder count '
                  f'will not match the manifest. Re-run with --clean (or rm '
                  f'-rf the directory) for a clean dataset.')

    # ---- codec profile ----------------------------------------------------
    ext = ext_of(args)
    qtables = subsampling = None
    if args.format == 'jpg':
        if args.jpeg_ref:
            qtables, subsampling = read_jpeg_profile(Path(args.jpeg_ref))
            print(f'  [codec] JPEG, quantisation tables copied from '
                  f'{Path(args.jpeg_ref).name} (subsampling={subsampling})')
        else:
            print(f'  [codec] JPEG at quality={args.jpeg_quality}. Prefer '
                  f'--jpeg-ref <a training .jpg> so the artefact signature '
                  f'matches the corpus exactly.')

    # ---- seed predictions (loaded lazily, once per CRS) -------------------
    seed_cache = {}

    def seed_for(crs):
        if args.seed_labels is None:
            return None
        k = str(crs)
        if k not in seed_cache:
            print(f'  [seed] indexing {Path(args.seed_labels).name} in {k}')
            seed_cache[k] = load_seed_index(Path(args.seed_labels), crs,
                                            args.seed_query)
        return seed_cache[k]

    # ---- pass 2: write pixels --------------------------------------------
    rows, n_seed_shapes = [], 0
    by_raster = {}
    for rp, aoi_id, x, y, frac, n_det in candidates:
        by_raster.setdefault(rp, []).append((aoi_id, x, y, frac, n_det))

    idx = 0
    for rp, items in by_raster.items():
        with rasterio.open(rp) as src:
            bands = [1, 2, 3] if src.count >= 3 else list(
                range(1, src.count + 1))
            stretch = image_stretch(src, bands) \
                if args.stretch == 'match-train' else None
            geoms, tree = seed_for(src.crs) if args.seed_labels else (None, None)
            kept = 0
            for aoi_id, x, y, frac, n_det in sorted(
                    items, key=lambda t: (t[2], t[1])):
                win = Window(x, y, tile, tile)
                data = src.read(bands, window=win, boundless=True, fill_value=0)
                valid = float(np.count_nonzero(data.any(axis=0))) / (tile * tile)
                if valid < args.min_coverage:
                    continue
                u8 = to_uint8(data, stretch)
                if u8.max() == 0:
                    continue

                split = split_of[aoi_id]
                sub = f'images_{split}' if args.val_frac > 0 else 'images'
                fname = f'{rp.stem}_x{x:06d}_y{y:06d}{ext}'
                tfm = win_transform(win, src.transform)
                if ext == '.jpg':
                    _write_jpeg_tile(u8, out / sub / fname, src.crs, tfm,
                                     qtables, subsampling, args.jpeg_quality)
                else:
                    _write_u8_tile(u8, out / sub / fname, src.crs, tfm)

                shapes = []
                if tree is not None:
                    from shapely.geometry import box as _b
                    tb = _b(*rasterio.windows.bounds(win, src.transform))
                    hits = tree_hits(tree, geoms, tb)
                    clipped = [g.intersection(tb) for g in hits]
                    clipped = [g for g in clipped if not g.is_empty]
                    shapes = polygons_to_labelme_shapes(
                        clipped, tfm, tile, args.seed_min_area_px,
                        args.seed_simplify_px, args.label)
                    n_seed_shapes += len(shapes)
                if tree is not None or args.write_labelme:
                    write_labelme_json(out / sub / f'{Path(fname).stem}.json',
                                       fname, tile, shapes)

                rows.append({
                    'tile': fname, 'split': split, 'source_image': rp.name,
                    'aoi_id': aoi_id, 'col_off': x, 'row_off': y,
                    'tile_px': tile, 'aoi_frac': round(frac, 4),
                    'valid_frac': round(valid, 4),
                    'n_det': n_det,
                    'gsd_m': round(abs(src.transform.a), 6),
                    'crs': str(src.crs), 'n_seed_shapes': len(shapes),
                    'minx': rasterio.windows.bounds(win, src.transform)[0],
                    'miny': rasterio.windows.bounds(win, src.transform)[1],
                    'maxx': rasterio.windows.bounds(win, src.transform)[2],
                    'maxy': rasterio.windows.bounds(win, src.transform)[3],
                })
                kept += 1
                idx += 1
            print(f'  {rp.name}: wrote {kept} tile(s)')

    if not rows:
        raise SystemExit('All candidate tiles were rejected by --min-coverage; '
                         'lower it or check for nodata in the AOIs.')

    # ---- manifest / footprints / provenance ------------------------------
    out.mkdir(parents=True, exist_ok=True)
    man = out / 'tile_manifest.csv'
    with open(man, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    fp_path = None
    try:
        import geopandas as gpd
        from shapely.geometry import box as shp_box2
        gdf = gpd.GeoDataFrame(
            rows,
            geometry=[shp_box2(r['minx'], r['miny'], r['maxx'], r['maxy'])
                      for r in rows],
            crs=rows[0]['crs'])
        fp_path = out / 'tile_footprints.gpkg'
        gdf.to_file(fp_path, driver='GPKG')
    except Exception as exc:                                    # noqa: BLE001
        print(f'[warn] footprint GeoPackage not written ({exc}); '
              f'tile_manifest.csv still has the bounds.')

    prov = {
        'created': datetime.now(timezone.utc).isoformat(),
        'images': str(args.images), 'aoi': str(args.aoi),
        'tile': tile, 'overlap': args.overlap, 'stride': stride,
        'min_aoi_frac': args.min_aoi_frac, 'min_coverage': args.min_coverage,
        'stretch': args.stretch, 'format': args.format, 'jpeg_ref': args.jpeg_ref,
        'jpeg_quality': args.jpeg_quality,
        'val_frac': args.val_frac, 'seed': args.seed,
        'max_tiles': args.max_tiles, 'max_per_aoi': args.max_per_aoi,
        'detections': str(args.detections) if args.detections else None,
        'det_query': args.det_query, 'min_detections': args.min_detections,
        'n_detections_in_tiles': sum(r['n_det'] for r in rows),
        'n_tiles_with_detections': sum(1 for r in rows if r['n_det']),
        'seed_labels': str(args.seed_labels) if args.seed_labels else None,
        'seed_query': args.seed_query,
        'seed_min_area_px': args.seed_min_area_px,
        'seed_simplify_px': args.seed_simplify_px,
        'n_tiles': len(rows), 'n_seed_shapes': n_seed_shapes,
        'n_aoi': len({r['aoi_id'] for r in rows}),
        'splits': {s: sum(1 for r in rows if r['split'] == s)
                   for s in sorted({r['split'] for r in rows})},
    }
    with open(out / 'tiling_provenance.json', 'w') as f:
        json.dump(prov, f, indent=2)

    # ---- optional: emit the empty-annotation COCO (hard NEGATIVES) --------
    # Use this when the AOIs delineate areas that contain NO date palms, e.g.
    # a "No_Datepalm" extent layer. The tiles then need no LabelMe pass at all:
    # they go straight into the hard-negative fine-tune as pure "nothing here
    # is a palm" supervision. Point HN_ROOT in maskrcnn_spatialmamba_s_
    # finetune_hn.py at <out>.
    coco_written = []
    if args.emit_coco == 'empty':
        for split in sorted({r['split'] for r in rows}):
            sub = f'images_{split}' if args.val_frac > 0 else 'images'
            files = [(r['tile'], r['tile_px'], r['tile_px'])
                     for r in rows if r['split'] == split]
            name = 'hard_neg.json' if split == 'train' else \
                f'hard_neg_{split}.json'
            n = build_coco(files, out / 'annotations' / name)
            coco_written.append((out / 'annotations' / name, n, sub))

    # ---- summary ---------------------------------------------------------
    print('=' * 70)
    print(f'AOI tiles written: {len(rows)} from '
          f'{len({r["source_image"] for r in rows})} image(s), '
          f'{prov["n_aoi"]} AOI polygon(s)')
    for s, n in prov['splits'].items():
        print(f'  {s:5s}: {n:5d} tiles  -> {out / ("images_" + s) if args.val_frac > 0 else out / "images"}')
    if args.detections:
        print(f'  false pos  : {prov["n_detections_in_tiles"]} detection(s) '
              f'inside {prov["n_tiles_with_detections"]} tile(s)')
    print(f'  manifest   : {man}')
    if fp_path:
        print(f'  footprints : {fp_path}  (load in QGIS to vet coverage)')
    if args.emit_coco == 'empty':
        for p, n, sub in coco_written:
            print(f'  COCO       : {p}  ({n} images, 0 annotations, from {sub})')
        print('  -> NO annotation pass needed: these are hard NEGATIVES.')
        print('NEXT: set HN_ROOT in maskrcnn_palm_finetune_hn/'
              'maskrcnn_spatialmamba_s_finetune_hn.py to')
        print(f'      {out}')
        print('      then: python tools/train.py <that config>')
    elif args.seed_labels:
        print(f'  seeded     : {n_seed_shapes} pre-filled polygons across '
              f'{sum(1 for r in rows if r["n_seed_shapes"])} tile(s)')
        print('  -> open the tiles in LabelMe: ADD the missed crowns, DELETE '
              'the wrong ones. Every real palm must be labelled.')
        print('NEXT: labelme2coco on each images_* folder, then train with '
              'maskrcnn_palm_finetune_hn/maskrcnn_spatialmamba_s_finetune_fn.py')
    else:
        print('  -> annotate every palm in LabelMe (no seeding requested)')
        print('NEXT: labelme2coco on each images_* folder, then train with '
              'maskrcnn_palm_finetune_hn/maskrcnn_spatialmamba_s_finetune_fn.py')
    print('=' * 70)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', required=True, metavar='DIR|RASTER',
                    help='folder of source images (recursive) or one raster')
    ap.add_argument('--aoi', required=True, metavar='DIR|SHP',
                    help='AOI extent polygons: one shapefile for everything, '
                         'or a folder of per-image shapefiles matched by stem')
    ap.add_argument('--out', required=True, help='output dataset directory')

    ap.add_argument('--tile', type=int, default=1024,
                    help='tile size in px; MUST match the training tile size '
                         '(default 1024)')
    ap.add_argument('--overlap', type=float, default=0.25,
                    help='fractional tile overlap 0-0.5 (default 0.25); some '
                         'overlap keeps crowns from being cut at every edge')
    ap.add_argument('--min-aoi-frac', type=float, default=0.5,
                    help='keep a tile only if this fraction of it lies inside '
                         'the AOI (default 0.5)')
    ap.add_argument('--min-coverage', type=float, default=0.5,
                    help='keep a tile only if this fraction of pixels is valid '
                         '(non-nodata) (default 0.5)')
    ap.add_argument('--stretch', choices=('none', 'match-train'), default='none',
                    help="'none' = raw uint8 passthrough, matches GE_train and "
                         "GE inference (DEFAULT, use for GE). 'match-train' = "
                         "per-image p2/p98, only for a WV-3-style workflow "
                         "whose positives were stretched.")
    ap.add_argument('--aoi-id-field', default=None,
                    help='attribute used to name AOIs (e.g. name, id); '
                         'defaults to <shapefile>-<feature index>')

    ap.add_argument('--val-frac', type=float, default=0.0,
                    help='hold out this fraction of WHOLE AOIs as validation '
                         '(default 0 = single images/ folder). Splitting by '
                         'AOI, not by tile, avoids spatial leakage.')
    ap.add_argument('--detections', default=None, metavar='GPKG|SHP|DIR',
                    help='predicted crowns: one file, or a FOLDER of per-image '
                         'prediction files paired by raster stem '
                         '(UAE_245.tif -> UAE_245_palms.gpkg). Inside '
                         'a PALM-FREE AOI every detection is a confirmed FALSE '
                         'POSITIVE, so this counts them per tile, ranks tiles '
                         'by that count, and (with --min-detections) keeps only '
                         'tiles the model actually gets wrong. Far more useful '
                         'than random tiles of empty desert.')
    ap.add_argument('--det-query', default=None,
                    help='pandas .query() filter for --detections, e.g. '
                         '"score > 0.35" to match the deployed threshold')
    ap.add_argument('--min-detections', type=int, default=0,
                    help='drop tiles with fewer than this many detections '
                         '(default 0 = keep all). Set 1 with --detections to '
                         'keep ONLY tiles containing false positives.')

    ap.add_argument('--max-per-aoi', type=int, default=0,
                    help='cap tiles per AOI polygon BEFORE the global cap '
                         '(deterministic). With many unequal AOIs this is the '
                         'flag that buys diversity — a global cap alone is '
                         'dominated by the largest polygons.')
    ap.add_argument('--exclude', nargs='*', default=[],
                    metavar='COCO_JSON|DIR',
                    help='COCO json(s) and/or image dir(s) whose tiles must '
                         'NOT be produced. Applied before --max-per-aoi, so '
                         'the cap fills with the next-best unused tiles. Use '
                         'this to build an evaluation set disjoint from the '
                         'training set: a different --seed will NOT do it, '
                         'because selection ranks by false-positive count '
                         'and the seed only breaks ties.')
    ap.add_argument('--max-tiles', type=int, default=0,
                    help='global cap on the number of tiles (deterministic '
                         'subsample, applied after --max-per-aoi)')
    ap.add_argument('--seed', type=int, default=0,
                    help='seed for the split / subsample hashes (default 0)')

    ap.add_argument('--seed-labels', default=None, metavar='GPKG|SHP',
                    help='existing predicted crown polygons to pre-fill the '
                         'LabelMe sidecars with (e.g. UAE_palms_master.gpkg). '
                         'Turns annotation into correction.')
    ap.add_argument('--seed-query', default=None,
                    help='pandas .query() filter applied to --seed-labels, '
                         'e.g. "score > 0.5 and is_fp != 1"')
    ap.add_argument('--seed-min-area-px', type=float, default=20.0,
                    help='drop seeded polygons smaller than this (px^2, '
                         'default 20) — removes sliver clip artefacts')
    ap.add_argument('--seed-simplify-px', type=float, default=0.75,
                    help='Douglas-Peucker tolerance in px for seeded polygons '
                         '(default 0.75); keeps LabelMe responsive')
    ap.add_argument('--label', default=LABEL_NAME,
                    help=f'class name written into the sidecars (default '
                         f'{LABEL_NAME}) — must match the training metainfo')
    ap.add_argument('--write-labelme', action='store_true',
                    help='write EMPTY LabelMe sidecars even without seeding')

    ap.add_argument('--format', choices=('tif', 'jpg'), default='tif',
                    help="tile encoding. Use 'jpg' when the corpus you will "
                         "mix against is JPEG (GE_15cm/train_GE is), otherwise "
                         "the codec artefact itself becomes a shortcut the "
                         "classifier can exploit. 'tif' keeps full GeoTIFF "
                         "georeferencing (default).")
    ap.add_argument('--jpeg-ref', default=None, metavar='JPG',
                    help='a training .jpg whose quantisation tables and '
                         'subsampling are copied verbatim, so the new tiles '
                         'carry the corpus artefact signature exactly')
    ap.add_argument('--jpeg-quality', type=int, default=95,
                    help='JPEG quality when --jpeg-ref is not given (95)')

    ap.add_argument('--emit-coco', choices=('none', 'empty'), default='none',
                    help="'empty' writes annotations/hard_neg.json directly "
                         "(0 annotations) — use when the AOIs are PALM-FREE "
                         "areas (a 'No_Datepalm' extent layer): the tiles are "
                         "hard NEGATIVES and need no LabelMe pass. 'none' "
                         "(default) = positives workflow, annotate in LabelMe.")
    ap.add_argument('--clean', action='store_true',
                    help='delete tiles left by a previous run in --out before '
                         'writing, so the folder matches the new COCO exactly')
    ap.add_argument('--dry-run', action='store_true',
                    help='report how many tiles each image/AOI would produce '
                         'and write nothing — always run this first')

    args = ap.parse_args()
    if not 0.0 <= args.overlap < 0.9:
        ap.error('--overlap must be in [0, 0.9)')
    if not 0.0 <= args.val_frac < 1.0:
        ap.error('--val-frac must be in [0, 1)')
    if args.emit_coco == 'empty' and args.seed_labels:
        ap.error('--emit-coco empty writes ZERO annotations, so --seed-labels '
                 'would be silently discarded. Negatives need no labels; drop '
                 'one of the two flags.')
    if args.emit_coco == 'empty' and args.overlap > 0:
        print('[note] overlapping NEGATIVE tiles duplicate the same background '
              'and can over-weight it; --overlap 0 is the usual choice for '
              '--emit-coco empty.')
    run(args)


if __name__ == '__main__':
    main()
