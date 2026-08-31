#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mosaic Tiling Pipeline  —  Production Version
==============================================
Tiles georeferenced mosaics with polygon annotations into fixed-size
image tiles and matching LabelMe JSON annotation files.

Supported annotation formats
-----------------------------
  .shp      Shapefile
  .gdb      File Geodatabase  (requires "layer" key)
  .gpkg     GeoPackage        (requires "layer" key if multi-layer)
  .geojson  GeoJSON

Tiling strategy
---------------
Tiles are generated within each split region polygon, anchored to the
region bounding box top-left corner (snapped to the raster pixel grid),
then clipped to the mosaic bounding box. Partial edge tiles and tiles
straddling two regions are handled by BOUNDARY_MODE.

Overlap policy
--------------
OVERLAP is applied to TRAINING tiles only, in both X and Y directions.
Val/test tiles always use stride = tile_size (no overlap), so the
evaluation set never counts the same crown twice.

Empty tiles
-----------
KEEP_EMPTY_TILES is per-split. Background tiles are training signal:
a detector that never sees bare desert or sabkha has no negative
evidence for them. MAX_EMPTY_FRACTION caps their share of the train
split via a seeded subsample, so the ratio is exact and reproducible
rather than whatever the terrain happened to yield.

Partial crowns
--------------
A crown cut by a tile edge and below MIN_VISIBLE_FRACTION is not
simply dropped: its pixels remain in the image, so dropping the label
teaches "visible palm crown = background". PARTIAL_POLICY="flag"
emits it with flags={"partial": true}, which the LabelMe->COCO
converter must map to iscrowd=1 so COCO ignores rather than scores it.

Contrast stretching
-------------------
When APPLY_CONTRAST_STRETCH = True, per-band stretch parameters are
computed ONCE from a mosaic overview before the tile loop, then applied
uniformly to every tile. This ensures consistent brightness across all
tiles — critical for stable training with ImageNet-pretrained backbones.
Stretch parameters are recorded in tiling_log.json for inference reuse.

Band selection
--------------
Set BANDS per job to select source bands. [1,2,3] = RGB from RGBA.
Leave as None to keep all bands.

═══════════════════════════════════════════════════════════════════
RESOLUTION CHANGE GUIDE
═══════════════════════════════════════════════════════════════════

  Parameter            UAV 5 cm   Aerial 15 cm   WV3 30 cm
  ─────────────────    ────────   ────────────   ─────────
  TILE_SIZE (px)       512        512            512
    Ground coverage: 5cm→26m, 15cm→77m, 30cm→154m per tile.
    512 matches the crop the network actually trains on, so no
    information is thrown away by a later RandomCrop.

  OVERLAP (px)         256        256            256
    50% of TILE_SIZE. Every interior crown then falls wholly
    inside at least one tile, which is what makes PARTIAL_POLICY
    cheap: nothing is lost by ignoring an edge-cut copy.
    Cost: each crown is written up to 4x. Quote unique_crowns,
    not polygon count.

  MIN_VISIBLE_AREA_M2  0.5        0.5            0.1
    GSD-invariant (m²). Lower at coarse resolutions.
    At 30 cm, 0.1 m² ~ 1 pixel; 0.5 m² ~ 5.5 pixels.

  BANDS                None       None           [1,2,3]
    Set [1,2,3] to drop alpha channel from RGBA images.
    For 8-band WV3: [5,3,2] = NIR-R-G composite.

  Per-job override: add "overlap": N or "bands": [...] to any job dict.

═══════════════════════════════════════════════════════════════════

USAGE
-----
1. Edit the CONFIGURATION section.
2. Run:  python tile_pipeline.py

OUTPUT
------
    out_dir/
        train/images/   <job>_train_tile_000000.tif + .json
        val/images/     <job>_val_tile_000000.tif   + .json
        test/images/    <job>_test_tile_000000.tif  + .json
        tile_index.gpkg
        tiling_log.json  (config, balance, stretch params, library versions)

REQUIREMENTS
------------
    pip install rasterio geopandas shapely tqdm numpy
"""

import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.affinity import affine_transform
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.strtree import STRtree
try:
    from tqdm import tqdm
except ImportError:          # progress bar is cosmetic; never block a run on it
    def tqdm(iterable=None, **kw):
        return iterable if iterable is not None else None


# =============================================================================
# CONFIGURATION  —  edit this section only
# =============================================================================

# Both roots come from the environment first, so this file carries no
# machine-specific path. Hard-coding them meant the defaults were correct on
# exactly one workstation and silently wrong everywhere else -- including on
# a replacement machine belonging to the same author.
#
#   Linux/container:  export PALM_DATA_ROOT=/workspace/datasets/source
#   Windows:          set PALM_DATA_ROOT=<drive>:\<your geodatabase>
DATA_ROOT = Path(os.environ.get(
    "PALM_DATA_ROOT", r"/path/to/source-imagery-and-vectors"))
OUTPUT_DIR = Path(os.environ.get(
    "PALM_OUTPUT_DIR", r"/path/to/output-tiles"))
CLASS_NAME = "DatePalm"

# ── UNIVERSAL SETTINGS ───────────────────────────────────────────────────────
# These three rules replace the old per-sensor lookup table. Each is either
# GSD-invariant or derived from the mosaic's own GSD at run time, so ONE
# configuration produces consistent tiles from UAV 5 cm, Aerial 15 cm,
# GE 15 cm and WorldView-3 30 cm without editing anything per source.

TILE_SIZE = 512               # pixels, every source. Matches the crop the
                              # network trains on, so nothing is thrown away
                              # by a later RandomCrop. Ground coverage follows
                              # the GSD: 26 m at 5 cm, 77 m at 15 cm,
                              # 154 m at 30 cm.

OVERLAP_FRACTION = 0.5        # training overlap as a fraction of TILE_SIZE.
                              # 0.5 -> 256 px at any GSD. Guarantees every
                              # interior crown falls wholly inside at least one
                              # tile. Set OVERLAP below to override in pixels.
OVERLAP = None                # None -> derived from OVERLAP_FRACTION.
                              # An int here wins, for one-off experiments.

MIN_VISIBLE_AREA_PX = 4.0     # sliver floor in PIXELS, converted to m² per
                              # mosaic using its own GSD. The old m² constant
                              # had to change per sensor (0.5 / 0.5 / 0.1)
                              # because 0.5 m² is 200 px at 5 cm but 5.5 px at
                              # 30 cm -- the same number meant different things.
                              # A pixel floor means the same thing everywhere.
MIN_VISIBLE_AREA_M2 = None    # None -> derived from MIN_VISIBLE_AREA_PX.

BANDS = "auto"                # "auto": 4+ band source -> [1,2,3] (drops alpha
                              # or extra WV-3 bands); 3-band -> keep all.
                              # None keeps all bands; a list selects exactly.

# Contrast stretching
# When True, per-band p2/p98 stretch is computed ONCE from a mosaic overview
# and applied uniformly to all tiles. Stretch params saved to tiling_log.json
# so the same transform can be applied at inference time.
APPLY_CONTRAST_STRETCH = True
STRETCH_LOWER_PCT      = 2    # lower percentile for stretch (default 2%)
STRETCH_UPPER_PCT      = 98   # upper percentile for stretch (default 98%)

# Stretch scope. Percentiles are computed per SCOPE and applied to every tile
# in it, so the choice decides what "the same brightness" means.
#   "job"   : each mosaic normalised to itself. Every mosaic ends up looking
#             average, which HIDES real radiometric differences between scenes
#             -- and if train comes from one mosaic and test from another, the
#             two splits are transformed differently and the shift is invisible.
#   "group" : mosaics sharing a "stretch_group" key are pooled and share one
#             transform. Scenes from one sensor then stay comparable, which is
#             what a train/val/test split spanning several mosaics needs.
#   "none"  : equivalent to APPLY_CONTRAST_STRETCH = False.
# Jobs without a "stretch_group" key fall back to their own name, i.e. "job"
# behaviour, so setting this to "group" is safe for existing job files.
STRETCH_SCOPE = "group"

# Largest permitted ratio between the p98 values of two mosaics pooled into
# one stretch group. Guards against averaging percentiles across genuinely
# different products; 4x is loose enough for the same sensor processed on
# different dates, tight enough to catch reflectance mixed with raw DN.
MAX_STRETCH_SPAN_RATIO = 4.0

# Polygon filtering
MIN_VISIBLE_FRACTION = 0.5    # a polygon with less than this fraction of its
                              # area inside the tile is "partial"
SIMPLIFY_TOLERANCE_M = 0.0    # simplify vertices (0 = off, recommended)

# What to do with a crown that is cut by the tile edge and falls below
# MIN_VISIBLE_FRACTION.
#   "drop" : omit it (the original behaviour). The crown's pixels stay in the
#            image with no label on them, so the model is actively taught that
#            visible palm crown = background. With 50% overlap every interior
#            crown is fully visible in some other tile, so the lost positive
#            costs nothing -- but the false negative it teaches is real.
#   "flag" : emit the clipped polygon with flags={"partial": true}. The
#            LabelMe->COCO converter must map that to iscrowd=1, which makes
#            COCO ignore the region in both loss and evaluation instead of
#            scoring it as background. This is the correct treatment.
# Use "flag" only if the converter honours it; otherwise partial crowns become
# ordinary instances and you have traded a false negative for a false positive.
PARTIAL_POLICY = "flag"

# Straddling tiles: a tile whose footprint reaches into a different split
# region. In "strict" mode any such tile is dropped. In "mask" mode the foreign
# pixels are blackened before the tile is written, so no imagery crosses the
# split -- the drop is then belt-and-braces, and an over-tight threshold throws
# away every tile along the region boundary. Expressed as a fraction of tile
# area; a foreign overlap smaller than this is tolerated (and masked away).
# 0.0 reproduces the original strict behaviour exactly.
STRADDLE_TOLERANCE = 0.0

# Tile filtering
# KEEP_EMPTY_TILES accepts a bool (applies to every split) or a per-split dict.
# Background tiles are training signal -- a detector that never sees bare
# desert, sabkha or built-up ground learns no negative evidence for them, which
# is exactly the failure the hard-negative work had to repair after the fact.
# They do NOT belong in val/test: an empty tile contributes no ground truth to
# COCO mAP, only opportunities for false positives, so including them silently
# changes what the reported metric means relative to Stages A-C.
KEEP_EMPTY_TILES   = {"train": True, "val": False, "test": False}

# Cap on how much of the TRAIN split may be empty tiles. With 50% overlap over
# a region that is mostly bare ground, empties can outnumber palm tiles several
# times over and dominate the loss. Empty candidates are collected during the
# sweep and a deterministic seeded subsample is written at the end, so the
# fraction is exact and the selection is reproducible.
#   MAX_EMPTY_FRACTION = 0.30 -> empties are at most 30% of that split's tiles
#   None -> keep every empty tile (no cap)
MAX_EMPTY_FRACTION = 0.30
EMPTY_SAMPLE_SEED  = 20260804
INTEGRITY_SAMPLE_SEED = 20260804   # which tiles the post-run check inspects

DROP_NODATA_TILES  = True     # True  = skip entirely black/nodata tiles
MIN_IMAGE_COVERAGE = 0.5      # minimum fraction of valid PIXELS per tile
                              # (a pixel counts as valid if ANY band is valid)
                              # 0.5 = drop tiles with > 50% black area

# Boundary handling
# "strict" → drop any tile not fully inside the split region polygon
#             (matches ArcGIS default, maximum label completeness)
# "mask"   → keep edge tiles, blacken pixels outside the region boundary
#             (maximises tile count, recovers irregular-boundary tiles)
BOUNDARY_MODE = "mask"

# Debugging — writes the clipped split region polygons to a GeoPackage
# so you can inspect the exact boundaries used for tiling in QGIS/ArcGIS
GENERATE_DEBUG_SHAPES = False

# Resume interrupted runs — skips tiles whose .json already exists
SKIP_EXISTING = False

# Parallel tile writing
MAX_WORKERS = 4

# Ceiling on writes queued but not yet on disk. Every queued tile holds its
# pixel buffer alive (512x512x3 uint8 = 768 KiB), and the sweep can enqueue far
# faster than four threads can compress LZW GeoTIFF. Submitting without a bound
# means a large mosaic accumulates the whole dataset in RAM before the first
# flush -- 20,000 tiles is ~15 GB and the process is killed with no useful
# error. The sweep blocks here instead, which costs nothing but bounds memory.
MAX_PENDING_WRITES = 256

# ── JOBS ─────────────────────────────────────────────────────────────────────
# Required keys:
#   name          : unique identifier, used as tile filename prefix
#   mosaic        : path to GeoTIFF
#   shapefile     : annotation file (.shp / .gdb / .gpkg / .geojson)
#   id_field      : "auto" or attribute column name with unique integer IDs
#
# Optional keys (override global defaults):
#   layer         : feature class name (required for .gdb, optional for .gpkg)
#   split_layer   : layer name if split_shapefile is .gdb/.gpkg
#   class_name    : label written into JSON shapes
#   overlap       : training overlap in px (overrides global OVERLAP)
#   bands         : band indices to keep, e.g. [1,2,3]
#
# Split assignment — choose ONE per job:
#   split                         : "train" / "val" / "test"  (Pattern A)
#   split_shapefile + split_field : spatial region file        (Pattern B)

JOBS = [
    {
        "name":            "WV3_Ajman",
        "mosaic":          DATA_ROOT / r"Subset Multiscale Data/Wordview3/Ajman_WV3_2021_30cm.tif",
        "shapefile":       DATA_ROOT / r"National_Date Palm_Vector Data/National_DatePalm_Mapping.gdb",
        "layer":           "WV3_Datepalm_Ajman_Fuj",
        "id_field":        "auto",
        "split_shapefile": DATA_ROOT / r"Subset Multiscale Data/Area_Datepalm/Datepalm_WV3_Area.shp",
        "split_field":     "Task",
        "overlap":         256,
        "bands":           [1, 2, 3],
    },
    {
        "name":            "WV3_Fujairah",
        "mosaic":          DATA_ROOT / r"Subset Multiscale Data/Wordview3/Kalba_WV3_2022_30cm.tif",
        "shapefile":       DATA_ROOT / r"National_Date Palm_Vector Data/National_DatePalm_Mapping.gdb",
        "layer":           "WV3_Datepalm_Ajman_Fuj",
        "id_field":        "auto",
        "split_shapefile": DATA_ROOT / r"Subset Multiscale Data/Area_Datepalm/Datepalm_WV3_Area.shp",
        "split_field":     "Task",
        "overlap":         256,
        "bands":           [1, 2, 3],
    },
]

# =============================================================================
# END OF CONFIGURATION  —  no changes needed below this line
# =============================================================================

LABELME_VERSION = "5.1.1"

# Running offset for id_field="auto", so crown ids stay unique across jobs.
# Deterministic because jobs are processed in the order they are listed.
_AUTO_ID_OFFSET = 0
VALID_SPLITS    = {"train", "val", "test"}


# =============================================================================
# UNIVERSAL SETTING RESOLVERS
# -----------------------------------------------------------------------------
# Everything sensor-dependent is derived here, from the mosaic itself, so the
# same configuration serves 5 cm UAV and 30 cm satellite alike.
# =============================================================================

def resolve_overlap(job: Dict) -> int:
    """Training overlap in pixels. Job override > explicit OVERLAP > fraction."""
    if "overlap" in job and job["overlap"] is not None:
        return int(job["overlap"])
    if OVERLAP is not None:
        return int(OVERLAP)
    return int(round(TILE_SIZE * OVERLAP_FRACTION))


def resolve_min_area_m2(job: Dict, gsd_m: float) -> float:
    """Sliver floor in m², from a pixel floor and this mosaic's GSD.

    A fixed m² threshold is not sensor-neutral: 0.5 m² is ~200 px at 5 cm but
    ~5.5 px at 30 cm, so the same constant discards nothing at one resolution
    and real crown fragments at another. A pixel floor is the invariant.
    """
    if job.get("min_visible_area_m2") is not None:
        return float(job["min_visible_area_m2"])
    if MIN_VISIBLE_AREA_M2 is not None:
        return float(MIN_VISIBLE_AREA_M2)
    return float(MIN_VISIBLE_AREA_PX) * (gsd_m ** 2)


def resolve_bands(job: Dict, band_count: int) -> Optional[List[int]]:
    """Which source bands to keep.

    "auto" takes the first three from any source with four or more bands --
    dropping an RGBA alpha channel, or the extra bands of a multispectral
    WorldView-3 product -- and keeps a three-band source untouched. Anything
    else (a false-colour composite such as [5,3,2]) must be stated explicitly.
    """
    spec = job.get("bands", BANDS)
    if spec == "auto":
        return [1, 2, 3] if band_count >= 4 else None
    return spec


# =============================================================================
# PRE-FLIGHT VALIDATION
# =============================================================================

def preflight_checks(jobs: List[Dict]) -> None:
    """
    Validate all inputs before any processing begins.
    Reports every problem found, not just the first one.
    """
    errors = []

    # Duplicate job names
    names = [j["name"] for j in jobs]
    seen  = set()
    for n in names:
        if n in seen:
            errors.append(f"  Duplicate job name: '{n}'. Names must be unique.")
        seen.add(n)

    for job in jobs:
        name = job.get("name", "<unnamed>")

        # Mosaic
        mosaic = Path(job.get("mosaic", ""))
        if not mosaic.exists():
            errors.append(f"  [{name}] Mosaic not found:\n       {mosaic}")
        else:
            # Dtype check — MMDetection expects uint8
            try:
                with rasterio.open(str(mosaic)) as _src:
                    dtype = _src.dtypes[0]
                    # A non-uint8 source is only a problem if nothing converts
                    # it. APPLY_CONTRAST_STRETCH maps every band through
                    # measured percentiles into 0-255 uint8, which is exactly
                    # the conversion gdal_translate would do -- and it is the
                    # only way multispectral products (float32 reflectance,
                    # uint16 DN) can be tiled at all.
                    if dtype != "uint8" and APPLY_CONTRAST_STRETCH:
                        print(f"  [{name}] dtype='{dtype}' -> uint8 via the "
                              f"p{STRETCH_LOWER_PCT}/p{STRETCH_UPPER_PCT} "
                              f"stretch. The mapping is recorded in "
                              f"tiling_log.json; quote it in Methods.")
                    elif dtype != "uint8":
                        win = rasterio.windows.Window(
                            0, 0, min(512, _src.width), min(512, _src.height)
                        )
                        max_val = int(np.max(_src.read(1, window=win)))
                        cmd = (f"gdal_translate -ot Byte -scale "
                               f"{mosaic.name} {mosaic.stem}_8bit.tif")
                        if dtype in ("uint16", "int16", "int32", "uint32"):
                            if max_val <= 255:
                                errors.append(
                                    f"  [{name}] dtype='{dtype}' but values fit "
                                    f"0-255. Safe cast: gdal_translate -ot Byte "
                                    f"{mosaic.name} {mosaic.stem}_8bit.tif"
                                )
                            else:
                                errors.append(
                                    f"  [{name}] dtype='{dtype}' (max={max_val}). "
                                    f"Rescale to uint8: {cmd}"
                                )
                        elif dtype in ("float32", "float64"):
                            errors.append(
                                f"  [{name}] dtype='{dtype}'. "
                                f"Rescale to uint8: {cmd}"
                            )
            except Exception as e:
                errors.append(f"  [{name}] Cannot open mosaic: {e}")

        # Annotation file
        shp = Path(job.get("shapefile", ""))
        if not shp.exists():
            errors.append(f"  [{name}] Shapefile/GDB not found:\n       {shp}")
        elif shp.suffix.lower() == ".gdb" and "layer" not in job:
            errors.append(
                f"  [{name}] Geodatabase requires a 'layer' key.\n"
                f"       Add  \"layer\": \"<feature_class_name>\"  to this job."
            )

        # Split assignment
        if "split" not in job:
            split_shp = Path(job.get("split_shapefile", ""))
            if not split_shp.exists():
                errors.append(f"  [{name}] Split shapefile not found:\n"
                              f"       {split_shp}")
            if "split_field" not in job:
                errors.append(f"  [{name}] Missing 'split_field' key.")
        else:
            if job["split"] not in VALID_SPLITS:
                errors.append(
                    f"  [{name}] Invalid split '{job['split']}'. "
                    f"Must be one of: {VALID_SPLITS}"
                )

        # Overlap
        job_overlap = resolve_overlap(job)
        if not (0 <= job_overlap < TILE_SIZE):
            errors.append(
                f"  [{name}] overlap={job_overlap} must be "
                f">= 0 and < tile_size={TILE_SIZE}."
            )

        # Bands
        bands = job.get("bands", BANDS)
        if bands is not None and bands != "auto":
            bad = [b for b in bands if not isinstance(b, int) or b < 1]
            if bad:
                errors.append(
                    f"  [{name}] Invalid band indices {bad}. "
                    f"Bands must be positive integers (1-based)."
                )

    # Global settings that are easy to get wrong and expensive to discover late
    if PARTIAL_POLICY not in ("drop", "flag"):
        errors.append(f"  PARTIAL_POLICY must be 'drop' or 'flag', "
                      f"got {PARTIAL_POLICY!r}.")
    if isinstance(KEEP_EMPTY_TILES, dict):
        bad = set(KEEP_EMPTY_TILES) - VALID_SPLITS
        if bad:
            errors.append(f"  KEEP_EMPTY_TILES has unknown split key(s): {bad}. "
                          f"Allowed: {VALID_SPLITS}")
        if KEEP_EMPTY_TILES.get("val") or KEEP_EMPTY_TILES.get("test"):
            errors.append(
                "  KEEP_EMPTY_TILES enables empty tiles in val/test. Those "
                "tiles add no ground truth to COCO mAP -- only chances to "
                "score false positives -- so the metric stops being comparable "
                "with Stages A-C. Set them False, or delete this check "
                "deliberately and say so in Methods.")
    if MAX_EMPTY_FRACTION is not None and not (0.0 <= MAX_EMPTY_FRACTION < 1.0):
        errors.append(f"  MAX_EMPTY_FRACTION must be in [0, 1) or None, "
                      f"got {MAX_EMPTY_FRACTION}.")
    if not (0.0 <= STRADDLE_TOLERANCE < 1.0):
        errors.append(f"  STRADDLE_TOLERANCE must be in [0, 1), "
                      f"got {STRADDLE_TOLERANCE}.")

    if errors:
        raise ValueError(
            "\n\nPre-flight validation failed:\n\n"
            + "\n\n".join(errors)
            + "\n\nFix the above issues before running."
        )

    print(f"  Pre-flight checks passed for {len(jobs)} job(s).")


# =============================================================================
# POST-TILING INTEGRITY CHECK
# =============================================================================

def integrity_check(output_dir: Path) -> List[str]:
    """
    Verify output integrity after all jobs complete.
    Returns a list of warning strings (empty = all clear).
    """
    import random as _random
    warnings_out = []

    for split in VALID_SPLITS:
        img_dir = output_dir / split / "images"
        if not img_dir.exists():
            continue
        tifs  = {p.stem for p in img_dir.glob("*.tif")}
        jsons = {p.stem for p in img_dir.glob("*.json")}

        orphan_tif  = tifs  - jsons
        orphan_json = jsons - tifs
        if orphan_tif:
            warnings_out.append(
                f"[{split}] {len(orphan_tif)} .tif(s) with no matching .json: "
                f"{sorted(orphan_tif)[:3]}{'...' if len(orphan_tif) > 3 else ''}"
            )
        if orphan_json:
            warnings_out.append(
                f"[{split}] {len(orphan_json)} .json(s) with no matching .tif."
            )

        # Seeded: an integrity check that inspects a different 20 files each
        # run cannot be used to confirm that a re-run produced the same thing.
        all_json = sorted(img_dir.glob("*.json"))
        sample = _random.Random(INTEGRITY_SAMPLE_SEED).sample(
            all_json, min(20, len(all_json)))
        for jp in sample:
            try:
                d = json.loads(jp.read_text(encoding="utf-8"))
                if "shapes" not in d or "imagePath" not in d:
                    warnings_out.append(
                        f"[{split}] Malformed JSON: {jp.name}"
                    )
            except Exception as e:
                warnings_out.append(f"[{split}] Unreadable JSON {jp.name}: {e}")

    # Spatial overlap check between splits
    idx_path = output_dir / "tile_index.gpkg"
    if idx_path.exists():
        try:
            idx = gpd.read_file(idx_path)
            for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
                g1 = idx[idx["split"] == s1]
                g2 = idx[idx["split"] == s2]
                if g1.empty or g2.empty:
                    continue
                joined = gpd.sjoin(
                    g1[["geometry"]], g2[["geometry"]],
                    how="inner", predicate="intersects"
                )
                if len(joined) > 0:
                    warnings_out.append(
                        f"Spatial overlap between {s1} and {s2}: "
                        f"{len(joined)} tile pair(s). Check split shapefile."
                    )
        except Exception as e:
            warnings_out.append(f"Spatial overlap check failed: {e}")

    return warnings_out


# =============================================================================
# CLASS BALANCE REPORT
# =============================================================================

def class_balance_report(output_dir: Path) -> Dict:
    """Compute per-split tile count, polygon count, and average palms per tile."""
    report = {}
    for split in VALID_SPLITS:
        img_dir = output_dir / split / "images"
        if not img_dir.exists():
            report[split] = {}
            continue
        json_paths  = list(img_dir.glob("*.json"))
        total_polys = 0
        partial_polys = 0
        empty_tiles = 0
        unique_ids  = set()
        for jp in json_paths:
            try:
                d = json.loads(jp.read_text(encoding="utf-8"))
                shapes = d.get("shapes", [])
                total_polys += len(shapes)
                for sh in shapes:
                    if sh.get("flags", {}).get("partial"):
                        partial_polys += 1
                    elif sh.get("group_id") is not None:
                        unique_ids.add(int(sh["group_id"]))
                if not shapes:
                    empty_tiles += 1
            except Exception:
                pass
        n = len(json_paths)
        report[split] = {
            "tiles":            n,
            "total_polygons":   total_polys,
            "partial_polygons": partial_polys,
            # With overlapping training tiles one crown is written several
            # times. Tile and polygon counts therefore overstate the dataset;
            # this is the number of DISTINCT reference crowns, and it is the
            # one to quote in a paper.
            "unique_crowns":    len(unique_ids),
            "empty_tiles":      empty_tiles,
            "avg_palms_tile":   round(total_polys / n, 1) if n > 0 else 0.0,
        }
    return report


# =============================================================================
# REPRODUCIBILITY RECORD
# =============================================================================

def reproducibility_record() -> Dict:
    """Capture library versions for publication reproducibility."""
    import sys as _sys, platform as _plat
    record = {"python": _sys.version, "platform": _plat.platform()}
    for pkg in ("rasterio", "geopandas", "shapely", "numpy", "fiona"):
        try:
            mod = __import__(pkg)
            record[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            record[pkg] = "not installed"
    return record


# =============================================================================
# VECTOR DATA LOADING
# =============================================================================

def read_vector(path: Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    """Read .shp / .gdb / .gpkg / .geojson into a GeoDataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".gdb":
        return gpd.read_file(str(path), layer=layer)
    if suffix == ".gpkg" and layer:
        return gpd.read_file(str(path), layer=layer)
    return gpd.read_file(str(path))


# =============================================================================
# CONTRAST STRETCHING  —  per-mosaic, applied uniformly to all tiles
# =============================================================================

def _stretch_stats(
    src: rasterio.io.DatasetReader,
    bands: Optional[List[int]],
    lower_pct: float,
    upper_pct: float,
) -> Tuple[List[Tuple[float, float]], int]:
    """
    Compute per-band (p_low, p_high) stretch parameters from a mosaic overview.

    Parameters are computed ONCE before the tile loop and applied uniformly
    to every tile, ensuring consistent brightness across the dataset.
    Black/nodata pixels (value 0) are excluded from the percentile calculation.

    Returns:
        List of (p_low, p_high) tuples, one per output band.
    """
    # Read a downsampled overview for speed
    ovr_factor = max(1, min(src.width, src.height) // 2048)
    out_w = max(1, src.width  // ovr_factor)
    out_h = max(1, src.height // ovr_factor)

    # `indexes` must be passed to read(), not applied afterwards. Without it
    # rasterio reads EVERY band into a buffer shaped for the selected ones and
    # raises DatasetIOShapeError -- which only happens when the selection is a
    # strict subset, i.e. exactly the RGBA and multispectral cases. The old
    # post-hoc slice also indexed the wrong axis positions once bands were
    # non-contiguous.
    band_indices = list(bands) if bands else list(range(1, src.count + 1))
    overview = src.read(
        indexes=band_indices,
        out_shape=(len(band_indices), out_h, out_w),
        resampling=Resampling.nearest,
    )

    params, n_valid_total = [], 0
    for i in range(overview.shape[0]):
        band        = overview[i].astype(np.float32)
        valid       = band[band > 0]
        n_valid_total = max(n_valid_total, len(valid))
        if len(valid) == 0:
            params.append((0.0, 255.0))
            continue
        p_low  = float(np.percentile(valid, lower_pct))
        p_high = float(np.percentile(valid, upper_pct))
        # Scale-aware. The old absolute guard (`< 1`) assumed integer DN: on
        # float32 reflectance in [0, 1] it forces a range of 1.0, i.e. the
        # whole physical scale, flattening the band to near-black.
        span = p_high - p_low
        floor = max(1e-6, abs(p_high) * 1e-3)
        if span < floor:
            p_high = p_low + floor
        params.append((p_low, p_high))

    return params, n_valid_total


def compute_stretch_params(src, bands, lower_pct, upper_pct):
    """Backward-compatible wrapper returning only the parameters."""
    return _stretch_stats(src, bands, lower_pct, upper_pct)[0]


def compute_group_stretch(jobs: List[Dict], scope: str) -> Dict[str, List]:
    """Per-scope stretch parameters, pooled across every mosaic in the scope.

    Percentiles are estimated from each mosaic's overview and averaged, weighted
    by valid-pixel count. Pooling matters whenever one dataset spans several
    mosaics: normalising each to itself erases genuine brightness differences
    between scenes and, worse, can transform train and test differently when
    they come from different mosaics -- a leak of exactly the kind that is
    invisible in the metrics.
    """
    if scope == "none" or not APPLY_CONTRAST_STRETCH:
        return {}
    groups: Dict[str, List[Dict]] = {}
    for j in jobs:
        key = j.get("stretch_group", j["name"]) if scope == "group" else j["name"]
        groups.setdefault(key, []).append(j)

    out: Dict[str, List] = {}
    for key, members in sorted(groups.items()):
        # Pooling assumes the members share a radiometric scale. Averaging
        # percentiles across products that do not -- float32 reflectance in
        # [0, 1] against uint16 DN in the thousands -- gives a stretch wrong
        # for both, and the tiles come out uniformly black or white with no
        # error raised anywhere.
        #
        # The test is on the VALUES, not the dtype. dtype is only a container:
        # the WorldView-3 pair here is float32 and uint16 yet both hold DN in
        # roughly 0-220, so they pool correctly and a dtype test would have
        # blocked a legitimate case. Two rasters whose p98 differ by more than
        # MAX_STRETCH_SPAN_RATIO really are different products.
        acc, weights, spans, dtypes = None, [], [], {}
        for j in members:
            with rasterio.open(j["mosaic"]) as src:
                bands = resolve_bands(j, src.count)
                params, n_valid = _stretch_stats(
                    src, bands, STRETCH_LOWER_PCT, STRETCH_UPPER_PCT)
                dtypes.setdefault(src.dtypes[0], []).append(j["name"])
            if acc is None:
                acc = [[] for _ in params]
            for i, pr in enumerate(params):
                acc[i].append(pr)
            weights.append(max(n_valid, 1))
            spans.append((j["name"], max(hi for _, hi in params)))

        hi_vals = [v for _, v in spans]
        ratio = max(hi_vals) / max(min(hi_vals), 1e-9)
        if ratio > MAX_STRETCH_SPAN_RATIO:
            raise ValueError(
                f"stretch group '{key}': members span incompatible value "
                f"ranges (p{STRETCH_UPPER_PCT} ratio {ratio:.1f}x, limit "
                f"{MAX_STRETCH_SPAN_RATIO}x):\n"
                + "\n".join(f"    {n}: p{STRETCH_UPPER_PCT} max {v:.1f}"
                             for n, v in spans)
                + "\nPooling these would produce a stretch wrong for every "
                  "member. Harmonise the rasters (gdal_translate to a common "
                  "scale), or give each its own stretch_group and state in "
                  "Methods that the scenes were normalised independently.")
        if len(dtypes) > 1:
            print(f"  [note] stretch group '{key}' mixes dtypes ("
                  + "; ".join(f"{d}: {', '.join(n)}" for d, n in dtypes.items())
                  + f") but the value ranges agree within {ratio:.2f}x, so "
                    f"they are the same product in different containers. "
                    f"Pooling.")
        w = np.asarray(weights, float); w /= w.sum()
        pooled = [(float(np.dot(w, [p[0] for p in band])),
                   float(np.dot(w, [p[1] for p in band])))
                  for band in acc]
        out[key] = pooled
        names = ", ".join(j["name"] for j in members)
        print(f"  stretch group '{key}' ({names}): "
              + ", ".join(f"b{i+1}[{lo:.1f},{hi:.1f}]"
                          for i, (lo, hi) in enumerate(pooled)))
    return out


def apply_stretch(
    img_data: np.ndarray,
    stretch_params: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Apply pre-computed per-band stretch parameters to a tile.

    Masked/nodata pixels (value 0) are preserved as 0 after stretching.
    Input img_data is modified and returned as uint8.
    """
    out = np.zeros_like(img_data, dtype=np.uint8)
    for i, (p_low, p_high) in enumerate(stretch_params):
        band      = img_data[i].astype(np.float32)
        zero_mask = (band == 0)
        stretched = np.clip(
            (band - p_low) / (p_high - p_low) * 255.0, 0, 255
        )
        stretched[zero_mask] = 0   # restore masked pixels
        out[i] = stretched.astype(np.uint8)
    return out


# =============================================================================
# REGION MASKING
# =============================================================================

def mask_image_with_region(
    img_data: np.ndarray,
    region_geom,
    tile_transform,
    tile_size: int,
) -> np.ndarray:
    """
    Blacken pixels outside the split region polygon.

    Rasterises the region polygon at tile resolution and multiplies the
    image by the binary mask. Pixels inside = unchanged; outside = 0.
    """
    from rasterio.features import rasterize as _rasterize
    region_mask = _rasterize(
        [(region_geom, 1)],
        out_shape=(tile_size, tile_size),
        transform=tile_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    return img_data * region_mask[np.newaxis, :, :]


# =============================================================================
# TILE GRID
# =============================================================================

def compute_region_tile_grid(
    region_geom,
    raster_transform,
    raster_width: int,
    raster_height: int,
    tile_size: int,
    overlap: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Compute (row, col, x_off, y_off) for all tile positions covering one
    split region. Grid origin anchored to region bounding box top-left,
    snapped to the nearest raster pixel.
    """
    stride = tile_size - overlap
    minx, miny, maxx, maxy = region_geom.bounds
    inv       = ~raster_transform
    col_start = max(0, int(math.floor((inv * (minx, maxy))[0])))
    row_start = max(0, int(math.floor((inv * (minx, maxy))[1])))
    col_end   = min(raster_width,  int(math.ceil((inv * (maxx, miny))[0])))
    row_end   = min(raster_height, int(math.ceil((inv * (maxx, miny))[1])))

    region_w = col_end - col_start
    region_h = row_end - row_start
    if region_w <= 0 or region_h <= 0:
        return []

    n_cols = math.ceil((region_w - overlap) / stride) if region_w > overlap else 1
    n_rows = math.ceil((region_h - overlap) / stride) if region_h > overlap else 1

    return [
        (r, c, col_start + c * stride, row_start + r * stride)
        for r in range(n_rows)
        for c in range(n_cols)
        if col_start + c * stride < raster_width
        and row_start + r * stride < raster_height
    ]


# =============================================================================
# SPLIT REGION HANDLING
# =============================================================================

def load_split_regions(
    shp_path: Path,
    split_field: str,
    split_layer: Optional[str],
    target_crs,
) -> gpd.GeoDataFrame:
    """Load, validate, and reproject a split region file."""
    gdf = read_vector(shp_path, layer=split_layer)
    if split_field not in gdf.columns:
        raise KeyError(
            f"Column '{split_field}' not found in {shp_path.name}.\n"
            f"Available columns: {list(gdf.columns)}"
        )
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    gdf[split_field] = gdf[split_field].astype(str).str.strip().str.lower()
    invalid = set(gdf[split_field].unique()) - VALID_SPLITS
    if invalid:
        raise ValueError(
            f"Split field '{split_field}' contains invalid values: {invalid}.\n"
            f"Allowed values: {VALID_SPLITS}\n"
            f"Values in file: {list(gdf[split_field].unique())}"
        )
    return gdf


def is_tile_in_region(
    tile_geom,
    region_tree: STRtree,
    region_geoms: np.ndarray,
    region_idx: int,
    tolerance: float = 0.0,
) -> bool:
    """Return False if the tile genuinely overlaps a different region.

    `tolerance` is a fraction of tile area. Adjacent regions that merely share
    an edge intersect in a zero-area line and are never affected; the tolerance
    exists for digitising slivers, and for "mask" mode where the foreign pixels
    are blackened anyway. 0.0 keeps the original strict rule.
    """
    tile_area  = tile_geom.area
    floor      = tile_area * max(tolerance, 1e-6)
    candidates = region_tree.query(tile_geom)
    for i in candidates:
        if i == region_idx:
            continue
        inter = region_geoms[i].intersection(tile_geom)
        if not inter.is_empty and inter.area > floor:
            return False
    return True


def keep_empty_for(split: str) -> bool:
    """Resolve KEEP_EMPTY_TILES, which may be a bool or a per-split dict."""
    if isinstance(KEEP_EMPTY_TILES, dict):
        return bool(KEEP_EMPTY_TILES.get(split, False))
    return bool(KEEP_EMPTY_TILES)


# =============================================================================
# POLYGON PROCESSING
# =============================================================================

def clip_polygon(
    polygon,
    tile_geom,
    min_visible_fraction: float,
    min_visible_area_m2: float,
    simplify_tol_m: float,
) -> Tuple[Optional[object], bool]:
    """Clip a polygon to the tile and apply visibility gates.

    Returns (geometry_or_None, is_partial). `is_partial` is True when the
    clipped crown falls below min_visible_fraction -- the caller decides
    whether to drop it or emit it as an ignore region (PARTIAL_POLICY).
    """
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        if polygon.is_empty:
            return None, False

    original_area = polygon.area
    if original_area <= 0:
        return None, False

    clipped = polygon.intersection(tile_geom)
    if clipped.is_empty:
        return None, False

    if isinstance(clipped, GeometryCollection):
        parts = [
            g for g in clipped.geoms
            if isinstance(g, (Polygon, MultiPolygon))
        ]
        if not parts:
            return None, False
        clipped = parts[0] if len(parts) == 1 else MultiPolygon([
            p for g in parts
            for p in (list(g.geoms) if isinstance(g, MultiPolygon) else [g])
        ])

    # A sliver this small carries no usable shape at any of our GSDs; drop it
    # outright rather than turning it into an ignore region.
    if clipped.area < min_visible_area_m2:
        return None, False

    is_partial = (clipped.area / original_area) < min_visible_fraction

    if simplify_tol_m > 0:
        clipped = clipped.simplify(simplify_tol_m, preserve_topology=True)
        if clipped.is_empty:
            return None, False

    return clipped, is_partial


def geometry_to_labelme_shapes(
    geometry,
    polygon_id: int,
    class_name: str,
    world_to_pixel,
    tile_size: int,
    is_partial: bool = False,
) -> List[Dict]:
    """Convert a (Multi)Polygon to LabelMe shape dicts."""
    polys = (
        list(geometry.geoms)
        if isinstance(geometry, MultiPolygon)
        else [geometry]
    )
    shapes = []
    for poly in polys:
        pixel_polygon = affine_transform(poly, world_to_pixel)
        coords = list(pixel_polygon.exterior.coords)[:-1]
        if len(coords) < 3:
            continue
        points = [
            [
                round(min(max(float(x), 0.0), tile_size - 1), 1),
                round(min(max(float(y), 0.0), tile_size - 1), 1),
            ]
            for x, y in coords
        ]
        shapes.append({
            "label":      class_name,
            "points":     points,
            "group_id":   int(polygon_id),
            "shape_type": "polygon",
            # "partial": this crown is cut by the tile edge and is below
            # MIN_VISIBLE_FRACTION. The COCO converter must translate it to
            # iscrowd=1 so it is ignored rather than scored.
            "flags":      {"partial": True} if is_partial else {},
        })
    return shapes


# =============================================================================
# IMAGE I/O
# =============================================================================

def read_tile(
    src: rasterio.io.DatasetReader,
    x_off: int,
    y_off: int,
    tile_size: int,
    bands: Optional[List[int]],
) -> Tuple[np.ndarray, bool]:
    """Read one tile, select bands, zero-pad to full size at mosaic edges."""
    win_w = min(tile_size, src.width  - x_off)
    win_h = min(tile_size, src.height - y_off)
    n_out = len(bands) if bands else src.count

    if win_w <= 0 or win_h <= 0:
        return np.zeros((n_out, tile_size, tile_size), dtype=src.dtypes[0]), False

    data = src.read(window=Window(x_off, y_off, win_w, win_h))
    if bands:
        data = data[[b - 1 for b in bands], :, :]

    if win_w < tile_size or win_h < tile_size:
        full = np.zeros((data.shape[0], tile_size, tile_size), dtype=data.dtype)
        full[:, :win_h, :win_w] = data
        data = full

    nodata   = src.nodata
    has_data = (
        bool((data != nodata).any()) if nodata is not None
        else bool((data != 0).any())
    )
    return data, has_data


def write_tile_pair(
    img_path: Path,
    json_path: Path,
    img_data: np.ndarray,
    shapes: List[Dict],
    tile_transform,
    raster_crs,
    src_profile: Dict,
    tile_size: int,
    n_bands: int,
) -> None:
    """Write one image tile and its JSON annotation."""
    img_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        **src_profile,
        "height":     tile_size,
        "width":      tile_size,
        "count":      n_bands,
        "transform":  tile_transform,
        "crs":        raster_crs,
        "compress":   "lzw",
        "tiled":      True,
        "blockxsize": 256,
        "blockysize": 256,
        "dtype":      img_data.dtype,  # reflects uint8 after stretch
    }
    with rasterio.open(img_path, "w", **profile) as dst:
        dst.write(img_data)

    payload = {
        "version":     LABELME_VERSION,
        "flags":       {},
        "shapes":      shapes,
        "imagePath":   img_path.name,
        "imageData":   None,
        "imageHeight": tile_size,
        "imageWidth":  tile_size,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# WRITE QUEUE
# =============================================================================

def harvest(pending: List, records: List[Dict], counts: Dict,
            job_name: str, failures: List[str],
            keep: int = 0) -> None:
    """Retire completed writes until at most `keep` remain outstanding.

    Counting happens HERE, not at submit time: a tile that failed to write is
    not a tile. The old code incremented the totals when the work was queued,
    so a disk error produced a summary that disagreed with the directory.
    """
    while len(pending) > keep:
        future, tile_name, split, n_palms, tile_geom, idx = pending.pop(0)
        try:
            future.result()
        except Exception as e:                       # noqa: BLE001
            failures.append(f"{tile_name}: {type(e).__name__}: {e}")
            continue
        counts[split]["tiles"]    += 1
        counts[split]["polygons"] += n_palms
        records.append({
            "job_name":  job_name,
            "split":     split,
            "tile_id":   idx,
            "tile_name": tile_name,
            "n_palms":   n_palms,
            "geometry":  tile_geom,
        })


# =============================================================================
# SINGLE JOB PROCESSING
# =============================================================================

def process_job(job: Dict, output_dir: Path,
                group_stretch: Optional[Dict[str, List]] = None) -> Dict:
    """
    Process one mosaic: tile within each split region, write tile pairs.

    Processing order per tile:
      1. Boundary check (strict or mask mode)
      2. Straddling check (data leakage prevention)
      3. Read raw pixels from mosaic
      4. Apply region mask if BOUNDARY_MODE = "mask"
      5. Check DROP_NODATA_TILES (raw has_data)
      6. Apply contrast stretch if APPLY_CONTRAST_STRETCH
      7. Check MIN_IMAGE_COVERAGE (on post-mask, pre-stretch data)
      8. Extract and clip polygon annotations
      9. Write tile image + JSON
    """
    job_name    = job["name"]
    job_overlap = resolve_overlap(job)
    job_class   = job.get("class_name", CLASS_NAME)

    print(f"\n  +-- Job: {job_name}")

    # ── Open mosaic ──────────────────────────────────────────────────────────
    src         = rasterio.open(job["mosaic"])
    raster_crs  = src.crs
    raster_tfm  = src.transform
    src_profile = src.profile.copy()
    gsd         = abs(raster_tfm.a)
    job_bands   = resolve_bands(job, src.count)
    job_min_m2  = resolve_min_area_m2(job, gsd)
    n_out_bands = len(job_bands) if job_bands else src.count

    print(f"  |   mosaic    : {Path(job['mosaic']).name}")
    print(f"  |   size      : {src.width} x {src.height} px, "
          f"GSD={gsd:.4f} m, {src.count} band(s) -> {n_out_bands} out")
    print(f"  |   CRS       : {raster_crs}")
    print(f"  |   derived   : overlap={job_overlap} px "
          f"({job_overlap * gsd:.1f} m), min sliver={job_min_m2:.4f} m² "
          f"({MIN_VISIBLE_AREA_PX:.0f} px), bands={job_bands or 'all'}")

    # ── Contrast stretch parameters ──────────────────────────────────────────
    # Computed ONCE from overview, applied uniformly to all tiles.
    stretch_params = None
    if APPLY_CONTRAST_STRETCH:
        key = job.get("stretch_group", job_name) \
            if STRETCH_SCOPE == "group" else job_name
        if group_stretch and key in group_stretch:
            stretch_params = group_stretch[key]
            print(f"  |   stretch   : shared parameters from group '{key}'")
        else:
            print(f"  |   stretch   : p{STRETCH_LOWER_PCT}/p{STRETCH_UPPER_PCT} "
                  f"from this mosaic's overview")
            stretch_params = compute_stretch_params(
                src, job_bands, STRETCH_LOWER_PCT, STRETCH_UPPER_PCT
            )
        for i, (pl, ph) in enumerate(stretch_params):
            print(f"  |              band {i+1}: [{pl:.1f}, {ph:.1f}] -> [0, 255]")

    # ── Load annotations ─────────────────────────────────────────────────────
    pgdf = read_vector(Path(job["shapefile"]), layer=job.get("layer"))
    if pgdf.crs != raster_crs:
        print(f"  |   reproject : annotations {pgdf.crs} -> {raster_crs}")
        pgdf = pgdf.to_crs(raster_crs)
    pgdf = pgdf[
        pgdf.geometry.notna() & ~pgdf.geometry.is_empty
    ].reset_index(drop=True)

    if job["id_field"] == "auto":
        # Offset by every crown numbered so far. Restarting at 1 per job makes
        # ids collide between mosaics: the LabelMe group_id is what identifies
        # one physical crown across the overlapping tiles it appears in, so a
        # collision silently merges two different palms when anything counts
        # distinct crowns across a split. Two 60-crown mosaics reported 60
        # unique crowns instead of 120.
        global _AUTO_ID_OFFSET
        pgdf["__id__"] = (np.arange(1, len(pgdf) + 1, dtype=np.int64)
                          + _AUTO_ID_OFFSET)
        _AUTO_ID_OFFSET += len(pgdf)
    else:
        pgdf["__id__"] = pgdf[job["id_field"]].astype(np.int64)
        # A user-supplied id field is trusted to be unique across jobs; if it
        # is only unique within a mosaic, the same collision applies.

    poly_tree = STRtree(pgdf.geometry.values)
    poly_geoms = pgdf.geometry.values
    poly_ids   = pgdf["__id__"].values
    print(f"  |   polygons  : {len(pgdf)}")

    # ── Split regions ─────────────────────────────────────────────────────────
    if "split" in job:
        fixed_split = job["split"]
        print(f"  |   split     : fixed -> {fixed_split}")
        x0 = raster_tfm.c; y1 = raster_tfm.f
        x1 = x0 + src.width  * raster_tfm.a
        y0 = y1 + src.height * raster_tfm.e
        regions_gdf = gpd.GeoDataFrame(
            {"split": [fixed_split],
             "geometry": [box(min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))]},
            crs=raster_crs,
        )
        split_col = "split"
    else:
        regions_gdf = load_split_regions(
            Path(job["split_shapefile"]),
            job["split_field"],
            job.get("split_layer"),
            raster_crs,
        )
        split_col = job["split_field"]
        print(f"  |   split     : region-based, "
              f"{dict(regions_gdf[split_col].value_counts())}")

    # Clip regions to mosaic bounding box
    x0 = raster_tfm.c; y1 = raster_tfm.f
    x1 = x0 + src.width  * raster_tfm.a
    y0 = y1 + src.height * raster_tfm.e
    mosaic_bbox = box(min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regions_gdf = regions_gdf.copy()
        regions_gdf["geometry"] = regions_gdf.geometry.intersection(mosaic_bbox)
    regions_gdf = regions_gdf[
        ~regions_gdf.geometry.is_empty
    ].reset_index(drop=True)

    if GENERATE_DEBUG_SHAPES:
        debug_path = output_dir / f"DEBUG_boundary_{job_name}.gpkg"
        regions_gdf.to_file(debug_path, driver="GPKG")
        print(f"  |   debug     : split boundary -> {debug_path.name}")

    region_geoms_arr  = regions_gdf.geometry.values
    region_labels_arr = regions_gdf[split_col].values
    region_tree       = STRtree(region_geoms_arr)

    # ── Per-region loop ───────────────────────────────────────────────────────
    counts  = {s: {"tiles": 0, "polygons": 0} for s in VALID_SPLITS}
    dropped = {
        "partial_edge": 0, "straddling": 0,
        "empty": 0, "nodata": 0, "skipped_existing": 0,
        "partial_polygon": 0, "partial_polygon_flagged": 0,
        "empty_over_budget": 0,
    }
    empty_candidates: List[Dict] = []
    tile_records = []

    def _max_existing_idx(split: str) -> int:
        img_dir = output_dir / split / "images"
        if not img_dir.exists():
            return 0
        prefix = f"{job_name}_{split}_tile_"
        indices = []
        for p in img_dir.glob(f"{prefix}*.tif"):
            try:
                indices.append(int(p.stem.replace(prefix, "")))
            except ValueError:
                pass
        return max(indices) + 1 if indices else 0

    split_counters = {
        s: [_max_existing_idx(s) if SKIP_EXISTING else 0]
        for s in VALID_SPLITS
    }
    if SKIP_EXISTING:
        for s in VALID_SPLITS:
            if split_counters[s][0] > 0:
                print(f"  |   resume    : {s} starts at {split_counters[s][0]}")

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    pending: List = []
    failures: List[str] = []

    for reg_idx, (region_geom, region_split) in enumerate(
        zip(region_geoms_arr, region_labels_arr)
    ):
        reg_overlap = job_overlap if region_split == "train" else 0
        stride      = TILE_SIZE - reg_overlap
        grid = compute_region_tile_grid(
            region_geom, raster_tfm,
            src.width, src.height,
            TILE_SIZE, reg_overlap,
        )
        print(f"  |   region {reg_idx+1:>2}  [{region_split}]: "
              f"{len(grid)} positions  "
              f"(stride={stride} px, overlap={reg_overlap} px)")

        for row, col, x_off, y_off in tqdm(
            grid,
            desc=f"     {job_name} [{region_split}]",
            unit="tile", leave=False,
        ):
            tile_transform = src.window_transform(
                Window(x_off, y_off, TILE_SIZE, TILE_SIZE)
            )
            x0_w = tile_transform.c
            y1_w = tile_transform.f
            x1_w = x0_w + TILE_SIZE * tile_transform.a
            y0_w = y1_w + TILE_SIZE * tile_transform.e
            tile_geom = box(
                min(x0_w, x1_w), min(y0_w, y1_w),
                max(x0_w, x1_w), max(y0_w, y1_w),
            )

            # ── Boundary check ────────────────────────────────────────────────
            if BOUNDARY_MODE == "strict":
                # Drop tiles not fully contained within region
                if not region_geom.covers(tile_geom):
                    dropped["partial_edge"] += 1
                    continue
            elif BOUNDARY_MODE == "mask":
                # Keep tiles that intersect the region; mask outside pixels later
                if not region_geom.intersects(tile_geom):
                    continue

            # ── Straddling check (data leakage prevention) ────────────────────
            if not is_tile_in_region(
                tile_geom, region_tree, region_geoms_arr, reg_idx,
                tolerance=STRADDLE_TOLERANCE,
            ):
                dropped["straddling"] += 1
                continue

            # ── Output paths ──────────────────────────────────────────────────
            split     = region_split
            idx       = split_counters[split][0]
            tile_name = f"{job_name}_{split}_tile_{idx:06d}"
            out_dir   = output_dir / split / "images"
            img_path  = out_dir / f"{tile_name}.tif"
            json_path = out_dir / f"{tile_name}.json"

            if SKIP_EXISTING and json_path.exists():
                dropped["skipped_existing"] += 1
                split_counters[split][0] += 1
                continue

            # ── Read raw pixels ───────────────────────────────────────────────
            img_data, has_data = read_tile(
                src, x_off, y_off, TILE_SIZE, job_bands
            )

            if DROP_NODATA_TILES and not has_data:
                dropped["nodata"] += 1
                continue

            # ── Apply region mask (BOUNDARY_MODE = "mask" only) ───────────────
            if BOUNDARY_MODE == "mask":
                img_data = mask_image_with_region(
                    img_data, region_geom, tile_transform, TILE_SIZE
                )

            # ── Image coverage check (on masked, pre-stretch data) ────────────
            # Checked before stretching so coverage reflects actual image content,
            # not stretch artefacts.
            if MIN_IMAGE_COVERAGE > 0.0:
                # Per PIXEL, not per band-element: a pixel is valid if ANY band
                # carries data. Counting elements made a (0, 5, 3) pixel score
                # 2/3 valid, so a fully-imaged tile could fall below the
                # threshold purely because one band happened to be dark.
                nodata_val = src.nodata
                valid_mask = (
                    (img_data != nodata_val).any(axis=0)
                    if nodata_val is not None
                    else (img_data != 0).any(axis=0)
                )
                if valid_mask.mean() < MIN_IMAGE_COVERAGE:
                    dropped["nodata"] += 1
                    continue

            # ── Apply contrast stretch ────────────────────────────────────────
            # Uses pre-computed mosaic-wide parameters — consistent across tiles.
            if APPLY_CONTRAST_STRETCH and stretch_params is not None:
                img_data = apply_stretch(img_data, stretch_params)

            # ── Extract polygon annotations ───────────────────────────────────
            _a, _b, _c, _d, _e, _f = tuple(~tile_transform)[:6]
            world_to_pixel = (_a, _b, _d, _e, _c, _f)

            # In mask mode, annotations are clipped to the INTERSECTION of
            # the tile and the region polygon — not the full tile extent.
            # This ensures polygons in the blackened (masked) area are
            # excluded from the JSON, preventing the model from being
            # trained to detect objects with no image content beneath them.
            if BOUNDARY_MODE == "mask":
                annotation_geom = tile_geom.intersection(region_geom)
                if annotation_geom.is_empty:
                    annotation_geom = tile_geom
            else:
                annotation_geom = tile_geom

            candidates = [
                (int(poly_ids[i]), poly_geoms[i])
                for i in poly_tree.query(annotation_geom)
                if poly_geoms[i].intersects(annotation_geom)
            ]
            shapes: List[Dict] = []
            n_full = 0
            for poly_id, poly in candidates:
                clipped, is_partial = clip_polygon(
                    poly, annotation_geom,
                    MIN_VISIBLE_FRACTION, job_min_m2,
                    SIMPLIFY_TOLERANCE_M,
                )
                if clipped is None:
                    continue
                if is_partial:
                    if PARTIAL_POLICY == "drop":
                        dropped["partial_polygon"] += 1
                        continue
                    dropped["partial_polygon_flagged"] += 1
                else:
                    n_full += 1
                shapes.extend(geometry_to_labelme_shapes(
                    clipped, poly_id, job_class,
                    world_to_pixel, TILE_SIZE, is_partial=is_partial,
                ))

            # A tile holding only ignore regions has no positive supervision
            # and no usable negative evidence either -- the palm pixels are
            # present but unscored. Treat it as empty, not as a palm tile.
            if n_full == 0:
                if not keep_empty_for(split):
                    dropped["empty"] += 1
                    continue
                # Defer: the empty budget is applied once the sweep knows how
                # many palm tiles this split actually produced.
                empty_candidates.append(dict(
                    split=split, x_off=x_off, y_off=y_off,
                    reg_idx=reg_idx, tile_geom=tile_geom,
                ))
                continue

            # ── Queue write ───────────────────────────────────────────────────
            future = executor.submit(
                write_tile_pair,
                img_path, json_path,
                img_data, shapes,
                tile_transform, raster_crs, src_profile,
                TILE_SIZE, n_out_bands,
            )
            pending.append((
                future, tile_name, split,
                len(shapes), tile_geom, idx,
            ))
            split_counters[split][0] += 1
            harvest(pending, tile_records, counts, job_name, failures,
                    keep=MAX_PENDING_WRITES)

    # ── Empty-tile budget ────────────────────────────────────────────────────
    # Background tiles are deliberate training signal, but with 50% overlap
    # over mostly bare ground they can outnumber palm tiles several times over
    # and dominate the loss. The sweep above deferred every empty tile; now
    # that the palm-tile count per split is known, keep a seeded random subset
    # sized to MAX_EMPTY_FRACTION of the split's final total.
    #
    #   n_empty / (n_palm + n_empty) = f   ->   n_empty = f/(1-f) * n_palm
    #
    # Selection is by seed, not by order, so it does not favour whichever
    # corner of the mosaic happened to be swept first.
    if empty_candidates:
        by_split: Dict[str, List[Dict]] = {}
        for c in empty_candidates:
            by_split.setdefault(c["split"], []).append(c)

        # Drain first: the ratio is taken from counts, and counts are only
        # incremented once a write has actually landed.
        harvest(pending, tile_records, counts, job_name, failures, keep=0)
        rng = np.random.default_rng(EMPTY_SAMPLE_SEED)
        for split, cands in sorted(by_split.items()):
            n_palm = counts[split]["tiles"]
            if MAX_EMPTY_FRACTION is None:
                chosen = list(range(len(cands)))
            elif n_palm == 0:
                # No palm tiles in this split: a ratio is undefined and an
                # all-background split is never what was intended.
                chosen = []
                print(f"  |   [warn] {split}: {len(cands)} empty tile(s) but no "
                      f"palm tiles; keeping none (ratio undefined)")
            else:
                f = float(MAX_EMPTY_FRACTION)
                n_keep = int(math.floor(f / (1.0 - f) * n_palm)) if f < 1.0 \
                    else len(cands)
                n_keep = min(n_keep, len(cands))
                chosen = sorted(rng.choice(len(cands), size=n_keep,
                                           replace=False).tolist()) \
                    if n_keep > 0 else []
            dropped["empty_over_budget"] += len(cands) - len(chosen)
            if chosen:
                print(f"  |   empties   [{split}]: keeping {len(chosen)} of "
                      f"{len(cands)} ({len(chosen) / (n_palm + len(chosen)):.1%} "
                      f"of the split)")

            for k in chosen:
                c   = cands[k]
                idx = split_counters[split][0]
                tile_name = f"{job_name}_{split}_tile_{idx:06d}"
                out_dir   = output_dir / split / "images"
                img_path  = out_dir / f"{tile_name}.tif"
                json_path = out_dir / f"{tile_name}.json"
                if SKIP_EXISTING and json_path.exists():
                    dropped["skipped_existing"] += 1
                    split_counters[split][0] += 1
                    continue

                # Re-read: the sweep deliberately did not hold pixels for tiles
                # it might discard. Every filter (nodata, coverage) already
                # passed, so only the read/mask/stretch chain is repeated.
                tile_transform = src.window_transform(
                    Window(c["x_off"], c["y_off"], TILE_SIZE, TILE_SIZE)
                )
                img_data, _ = read_tile(
                    src, c["x_off"], c["y_off"], TILE_SIZE, job_bands
                )
                if BOUNDARY_MODE == "mask":
                    img_data = mask_image_with_region(
                        img_data, region_geoms_arr[c["reg_idx"]],
                        tile_transform, TILE_SIZE,
                    )
                if APPLY_CONTRAST_STRETCH and stretch_params is not None:
                    img_data = apply_stretch(img_data, stretch_params)

                future = executor.submit(
                    write_tile_pair,
                    img_path, json_path,
                    img_data, [],
                    tile_transform, raster_crs, src_profile,
                    TILE_SIZE, n_out_bands,
                )
                pending.append((
                    future, tile_name, split, 0, c["tile_geom"], idx,
                ))
                split_counters[split][0] += 1
                harvest(pending, tile_records, counts, job_name, failures,
                        keep=MAX_PENDING_WRITES)

    # Wait for all writes
    harvest(pending, tile_records, counts, job_name, failures, keep=0)
    executor.shutdown(wait=True)
    src.close()

    if failures:
        print(f"\n  !! {len(failures)} tile(s) failed to write in "
              f"'{job_name}'. First few:")
        for f in failures[:5]:
            print(f"       {f}")

    # Per-job summary
    print(f"\n     Results for '{job_name}':")
    for s in ("train", "val", "test"):
        if counts[s]["tiles"] > 0:
            note = f"  (overlap={job_overlap} px)" if s == "train" and job_overlap > 0 else ""
            print(f"       {s:5s} : {counts[s]['tiles']:>5} tiles, "
                  f"{counts[s]['polygons']:>5} polygons{note}")
    drop_msgs = [
        ("partial_edge",            "partial edge"),
        ("straddling",              "straddling boundary"),
        ("empty",                   "no polygons"),
        ("empty_over_budget",       "empty, over budget"),
        ("nodata",                  "nodata/coverage"),
        ("partial_polygon",         "crown cut by tile edge"),
        ("partial_polygon_flagged", "crown cut, flagged partial"),
        ("skipped_existing",        "already existed"),
    ]
    for key, label in drop_msgs:
        if dropped[key] > 0:
            print(f"       Dropped ({label:25s}): {dropped[key]}")

    return {
        "job_name":     job_name,
        "counts":       counts,
        "dropped":      dropped,
        "write_failures": failures,
        "stretch_params": [
            {"band": i+1, "p_low": pl, "p_high": ph}
            for i, (pl, ph) in enumerate(stretch_params or [])
        ],
        "tile_records": tile_records,
    }


# =============================================================================
# MAIN
# =============================================================================

def load_jobs_file(path: Path) -> List[Dict]:
    """Read a JSON job list, so a new source never requires editing this file.

    Path-valued keys are converted to Path. Relative paths resolve against the
    job file's own directory, which makes a job file portable between the
    Windows share and the Linux container as long as the layout beneath it
    matches.
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    jobs = doc["jobs"] if isinstance(doc, dict) else doc
    base = Path(path).parent
    for j in jobs:
        for k in ("mosaic", "shapefile", "split_shapefile"):
            if j.get(k):
                q = Path(j[k])
                j[k] = q if q.is_absolute() else (base / q).resolve()
    return jobs


def apply_overrides(pairs: Optional[List[str]]) -> None:
    """--set KEY=VALUE against this module's globals, typed via literal_eval.

    Mirrors palm_inference_pipeline.py so both tools are driven the same way.
    """
    import ast
    g = globals()
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        key, val = pair.split("=", 1)
        key = key.strip()
        if key not in g:
            raise SystemExit(
                f"--set: unknown setting {key!r}. Known settings:\n  "
                + ", ".join(sorted(k for k in g if k.isupper()
                                   and not k.startswith("_"))))
        try:
            parsed = ast.literal_eval(val)
        except Exception:
            parsed = val
        if key.endswith("_DIR") or key == "OUTPUT_DIR":
            parsed = Path(parsed)
        g[key] = parsed
        print(f"  CONFIG override: {key} = {parsed!r}")


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        description="Universal mosaic -> LabelMe tiler. One configuration "
                    "serves UAV 5 cm, Aerial 15 cm, GE 15 cm and WV-3 30 cm; "
                    "overlap, sliver floor and band selection are derived "
                    "from each mosaic.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", default=None,
                    help="JSON file describing the jobs. Omit to use the "
                         "JOBS list defined in this file.")
    ap.add_argument("--out", default=None,
                    help="output directory (overrides OUTPUT_DIR)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    dest="overrides",
                    help="override any upper-case setting, e.g. "
                         "--set TILE_SIZE=1024 --set OVERLAP_FRACTION=0.25")
    ap.add_argument("--only", default=None,
                    help="run only jobs whose name contains this substring "
                         "(same idiom as palm_inference_pipeline.py)")
    ap.add_argument("--resume", action="store_true",
                    help="skip tiles whose .json already exists "
                         "(sets SKIP_EXISTING)")
    ap.add_argument("--dry-run", action="store_true",
                    help="run pre-flight and print the resolved settings per "
                         "job, then exit without writing tiles")
    return ap.parse_args()


def main() -> None:
    t0 = time.time()
    args = parse_args()
    apply_overrides(args.overrides)
    if args.out:
        globals()["OUTPUT_DIR"] = Path(args.out)
    if args.jobs:
        globals()["JOBS"] = load_jobs_file(Path(args.jobs))
    if args.resume:
        globals()["SKIP_EXISTING"] = True
    if args.only:
        kept = [j for j in JOBS if args.only in j["name"]]
        if not kept:
            raise SystemExit(
                f"--only {args.only!r} matched no job. Available: "
                + ", ".join(j["name"] for j in JOBS))
        globals()["JOBS"] = kept
        print(f"  --only {args.only!r}: {len(kept)} job(s)")

    print(f"\n{'='*60}")
    print(f"  Palm Tree Tiling Pipeline  —  Production Version")
    print(f"{'='*60}")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"  Tile size    : {TILE_SIZE} x {TILE_SIZE} px")
    _ov = OVERLAP if OVERLAP is not None else int(round(TILE_SIZE * OVERLAP_FRACTION))
    print(f"  Overlap      : {_ov} px "
          f"(training only; fraction={OVERLAP_FRACTION})")
    print(f"  Sliver floor : {MIN_VISIBLE_AREA_PX} px "
          f"(-> m² per mosaic GSD)")
    print(f"  Boundary     : {BOUNDARY_MODE}")
    print(f"  Stretch      : {APPLY_CONTRAST_STRETCH}")
    print(f"  Min coverage : {MIN_IMAGE_COVERAGE}")
    print(f"  Workers      : {MAX_WORKERS}")
    print(f"  Skip exist   : {SKIP_EXISTING}")
    print(f"  Jobs         : {len(JOBS)}")
    print(f"{'='*60}")

    print("\n  Running pre-flight checks...")
    preflight_checks(JOBS)

    if args.dry_run:
        print("\n  Resolved settings per job (dry run, nothing written):")
        for job in JOBS:
            with rasterio.open(job["mosaic"]) as s_:
                g = abs(s_.transform.a)
                print(f"    {job['name']}: GSD={g:.4f} m, "
                      f"{s_.count} band(s) -> {resolve_bands(job, s_.count) or 'all'}, "
                      f"tile={TILE_SIZE} px ({TILE_SIZE * g:.0f} m), "
                      f"overlap={resolve_overlap(job)} px "
                      f"({resolve_overlap(job) * g:.1f} m), "
                      f"sliver>={resolve_min_area_m2(job, g):.4f} m²")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records  = []
    job_logs     = []
    crs_registry = {}

    print("\n  Stretch parameters:")
    group_stretch = compute_group_stretch(JOBS, STRETCH_SCOPE)

    for job in JOBS:
        result = process_job(job, OUTPUT_DIR, group_stretch)
        all_records.extend(result.pop("tile_records"))
        job_logs.append(result)
        with rasterio.open(job["mosaic"]) as s:
            crs_registry[job["name"]] = str(s.crs)

    # Tile index
    if all_records:
        first_crs  = list(crs_registry.values())[0]
        mixed_crs  = len(set(crs_registry.values())) > 1
        idx        = gpd.GeoDataFrame(all_records, crs=first_crs)
        if mixed_crs:
            print(f"\n  ⚠  Multiple CRSes detected — reprojecting tile index to {first_crs}")
            for jn, jcrs in crs_registry.items():
                if jcrs == first_crs:
                    continue
                mask = idx["job_name"] == jn
                sub  = gpd.GeoDataFrame(idx[mask], crs=jcrs, geometry="geometry")
                idx.loc[mask, "geometry"] = sub.to_crs(first_crs).geometry.values
        idx.to_file(OUTPUT_DIR / "tile_index.gpkg", driver="GPKG")

    # Integrity check
    print("\n  Running post-tiling integrity checks...")
    issues = integrity_check(OUTPUT_DIR)
    if issues:
        print(f"  ⚠  {len(issues)} issue(s) found:")
        for iss in issues:
            print(f"       - {iss}")
    else:
        print("  ✓  All integrity checks passed.")

    # Class balance
    balance = class_balance_report(OUTPUT_DIR)
    repro   = reproducibility_record()

    # Audit log
    elapsed = round(time.time() - t0, 1)
    log = {
        "pipeline_config": {
            "tile_size":            TILE_SIZE,
            "overlap_fraction":     OVERLAP_FRACTION,
            "overlap_explicit_px":  OVERLAP,
            "min_visible_area_px":  MIN_VISIBLE_AREA_PX,
            "min_visible_area_m2":  MIN_VISIBLE_AREA_M2,
            "bands":                BANDS,
            "boundary_mode":        BOUNDARY_MODE,
            "apply_contrast_stretch": APPLY_CONTRAST_STRETCH,
            "stretch_lower_pct":    STRETCH_LOWER_PCT,
            "stretch_upper_pct":    STRETCH_UPPER_PCT,
            "stretch_scope":        STRETCH_SCOPE,
            "min_visible_fraction": MIN_VISIBLE_FRACTION,
            "min_visible_area_m2":  MIN_VISIBLE_AREA_M2,
            "keep_empty_tiles":     KEEP_EMPTY_TILES,
            "max_empty_fraction":   MAX_EMPTY_FRACTION,
            "empty_sample_seed":    EMPTY_SAMPLE_SEED,
            "partial_policy":       PARTIAL_POLICY,
            "straddle_tolerance":   STRADDLE_TOLERANCE,
            "drop_nodata_tiles":    DROP_NODATA_TILES,
            "min_image_coverage":   MIN_IMAGE_COVERAGE,
            "skip_existing":        SKIP_EXISTING,
            "max_workers":          MAX_WORKERS,
        },
        "elapsed_seconds":  elapsed,
        "class_balance":    balance,
        "reproducibility":  repro,
        "job_summaries":    job_logs,
        "crs_per_job":      crs_registry,
    }
    (OUTPUT_DIR / "tiling_log.json").write_text(json.dumps(log, indent=2))

    # A run that lost tiles must not look like a clean run. Anything reading
    # this in a shell script or CI needs the exit code to say so.
    n_failed = sum(len(j.get("write_failures", [])) for j in job_logs)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Finished in {elapsed}s")
    print(f"{'='*60}")

    print(f"\n  Class balance:")
    print(f"     {'Split':<8}  {'Tiles':>7}  {'Polygons':>9}  {'Partial':>8}  "
          f"{'Unique':>8}  {'Avg/tile':>9}  {'Empty':>7}")
    print(f"     {'-'*72}")
    for s in ("train", "val", "test"):
        b = balance.get(s, {})
        if not b:
            continue
        print(f"     {s:<8}  {b['tiles']:>7}  {b['total_polygons']:>9}  "
              f"{b['partial_polygons']:>8}  {b['unique_crowns']:>8}  "
              f"{b['avg_palms_tile']:>9.1f}  {b['empty_tiles']:>7}")
    print("\n     'Unique' counts distinct crowns; overlapping training tiles "
          "repeat each\n     crown up to (tile/stride)^2 times, so quote that "
          "column in the paper.")

    print(f"\n     Output     -> {OUTPUT_DIR}")
    print(f"     Tile index -> tile_index.gpkg")
    print(f"     Audit log  -> tiling_log.json")
    print()

    if n_failed:
        print(f"  {n_failed} tile(s) FAILED to write. The counts above exclude "
              f"them; see tiling_log.json -> job_summaries[].write_failures.")
        sys.exit(2)
    if issues:
        print(f"  Completed with {len(issues)} integrity warning(s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
