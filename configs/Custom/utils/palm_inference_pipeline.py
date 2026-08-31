#!/usr/bin/env python3
# =============================================================================
# palm_inference_pipeline.py  (v2 — country-scale)
# -----------------------------------------------------------------------------
# Production-grade, large-scale instance-segmentation inference for date palm
# detection over remotely sensed imagery (UAV / GE / Aerial / satellite).
#
# Designed for the UAE country-scale GE 15 cm run:
#   ~250 folders x ~100 GeoTIFFs (each folder = one 10x10 km block).
#
# WORKFLOW (three commands):
#
#   1) python palm_inference_pipeline.py scan
#        Reads metadata of every tiff (no pixels), validates CRS/GSD/grid
#        alignment per folder, reports overlaps, estimates tile counts and
#        GPU time, and writes scan_manifest.csv. Run this FIRST — it is fast
#        and catches data problems before any GPU time is spent.
#
#   1b) python palm_inference_pipeline.py calibrate \
#           --gt val_GE.json --img-root /path/to/GE_15cm
#        Derives the F1-optimal SCORE_THR for the deployed checkpoint against
#        the COCO validation ground truth. Needs only the checkpoint already
#        in CONFIG: it runs inference over the val images itself (GPU, a few
#        minutes) and caches the predictions, so re-sweeps (other --iou or
#        --metric values) are instant. An existing pkl can be supplied with
#        --pkl instead. Put the reported threshold into CONFIG.SCORE_THR.
#
#   2) python palm_inference_pipeline.py infer [--shard K/N] [--only NAME]
#        Runs inference. Each FOLDER is processed as ONE virtual mosaic
#        (windowed reads across its tiffs), so palms on internal tiff seams
#        are seen whole — no split-crown loss, no double counting inside a
#        folder. Outputs one GeoPackage + one stats JSON per folder,
#        written atomically; fully resumable. --shard 1/2 and --shard 2/2
#        on two workstations split the folders deterministically.
#
#   3) python palm_inference_pipeline.py merge [--inputs DIR ...]
#        Collects all per-folder outputs (from one or several output dirs,
#        e.g. copied from both workstations), removes duplicates in the
#        border band between adjacent folders (cross-unit centroid NMS),
#        and writes the country master GeoPackage + count_summary.csv +
#        overall statistics (counts, diameter percentiles).
#
# Design choices (locked with the user):
#   * Model         : Stage C unified multi-source, best_GE checkpoint.
#   * Geometry      : area-preserving circle per crown (GEOMETRY_MODE).
#   * Filtering     : confidence + circularity (+ optional diameter band).
#   * Dedup         : tile ownership -> per-folder polygon NMS -> cross-folder
#                     border-band centroid NMS at merge time.
#   * Output        : per-folder GeoPackages + one merged master (EPSG:32640).
#
# Robustness / correctness notes:
#   * Folders are mosaicked virtually from tiff metadata (same CRS, same GSD,
#     pixel-aligned grid required — validated, never assumed). No VRT files,
#     no full-raster loads; every read is windowed.
#   * Geographic CRS (e.g. EPSG:4326 GEE exports) handled: metre GSD is
#     derived from the latitude, so diameters, adaptive overlap and dedup
#     distances stay correct. Never computes "metres" from degree pixels.
#   * Handles >3 band rasters, alpha bands, nodata, non-uint8 dtypes.
#   * Skips tiles with no source coverage / all-nodata (no wasted GPU).
#   * Per-worker rasterio dataset handles (never shared across processes).
#   * Circularity computed on the cleaned MASK (pre-simplification).
#   * CUDA-OOM fallback retries a spiking batch tile-by-tile.
#   * Crashes on one folder do not abort the batch; failures are logged.
#   * Atomic outputs: tmp-file + rename, stats JSON written last; RESUME
#     trusts only folders whose JSON sidecar exists.
# =============================================================================

import os
import sys
import csv
import json
import time
import shutil
import hashlib
import logging
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:           # progress bar is optional; degrade gracefully
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else None

# Geo / vector
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box as shp_box
from shapely.affinity import affine_transform
from shapely.validation import make_valid

# Vision
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
#                                  CONFIG
# =============================================================================
class CONFIG:
    # ---- Model -------------------------------------------------------------
    # Stage C Spatial-Mamba-S adapted by hard-negative fine-tuning for the
    # operational domain. The benchmark checkpoints and their reported metrics
    # are untouched; this is a deployment model.
    #
    # Measured on 2,098 palm-free desert tiles the adaptation never trained on,
    # against the Stage C best_GE checkpoint it was adapted from:
    #   score 0.35: 4.158 -> 0.048 false positives per tile  (-98.9%)
    #   tiles carrying at least one false positive: 90.9% -> 1.6%
    # with GE validation segm mAP@50 unchanged at 0.948, so the suppression
    # cost no recall on labelled palms.
    #
    # Iteration 4000 of the FIRST mining round. A second round on 5,451
    # negatives (round 1's 3,083 plus 2,368 more) was measurably WORSE at every
    # threshold, so more negatives did not help: holding the negative share
    # fixed while enlarging the pool draws the hardest tiles less often, and
    # the easier additions displace the signal. Checkpoint chosen by measuring
    # false positives on held-out tiles, NOT by the run's save_best, which
    # selects on GE validation mAP and cannot see this adaptation at all.
    #
    # CHECKPOINT_FILE may contain a glob; it is resolved at load time and must
    # match exactly one file.
    CONFIG_FILE     = 'configs/Custom/maskrcnn_palm_finetune_hn/maskrcnn_spatialmamba_s_deploy.py'
    CHECKPOINT_FILE = ('work_dirs/Finetune_HN/maskrcnn_spatialmamba_s_finetune_hn/'
                       'best_coco_segm_mAP_50_iter_4000.pth')
    DEVICE          = 'cuda:0'

    # ---- Input / output ----------------------------------------------------
    # INPUT_PATH: root folder holding the per-block folders (UAE_216, ...).
    # Every directory that directly contains rasters becomes one processing
    # UNIT (a virtual mosaic). A single image path also works.
    INPUT_PATH   = '/workspace/datasets/GEE_Geotiff'
    OUTPUT_DIR   = '/workspace/datasets/GEE_Geotiff/output'
    IMAGE_EXTS   = ('.tif', '.tiff', '.img', '.jp2', '.vrt')
    MERGED_NAME  = 'UAE_palms_master'    # basename of the merged master output

    # PROCESS_UNIT:
    #   'folder' -> each directory of rasters is mosaicked into one unit
    #               (recommended: no seam errors inside a folder).
    #   'image'  -> each raster is its own unit (fallback if a folder fails
    #               grid validation; border dedup then runs on every seam).
    PROCESS_UNIT = 'folder'
    # Sub-pixel grid offsets are ALWAYS absorbed by snapping each tiff to the
    # nearest integer pixel of the folder grid: GEE exports carry arbitrary
    # fractional offsets (observed 0.34, 0.49 px), and the worst possible
    # snap error is half a pixel — 7.5 cm on the ground at 15 cm GSD, which
    # is negligible against 3-8 m crowns and all metre-scale logic. Snaps
    # larger than this threshold are logged and recorded in the unit stats
    # for provenance. Hard validation still rejects what snapping cannot
    # fix: mixed CRS, mismatched pixel size, rotated transforms.
    GRID_SNAP_WARN_PX = 0.25

    # ---- Band selection ----------------------------------------------------
    # BANDS: explicit 1-indexed band numbers fed to the model as R,G,B, or
    # None to auto-detect from colorinterp tags (recommended; alpha becomes a
    # validity mask). ALPHA_BAND: None=auto, 0=force off, N=use band N.
    BANDS = None
    ALPHA_BAND = None

    # Pixels whose RGB composite is exactly this value are treated as nodata
    # (black collars / uncovered mosaic area). None disables.
    NODATA_FILL_VALUE = 0

    # ---- Tiling ------------------------------------------------------------
    # TILE_SIZE must match the training tile size (1024) so crowns appear at
    # the pixel scale the mask head learned. Do not lower it for speed.
    TILE_SIZE   = 1024

    # Overlap is derived per-unit from MAX_CROWN_M and the metre GSD, so the
    # seam-free ownership logic holds across 5/15/30 cm imagery.
    ADAPTIVE_OVERLAP = True
    MAX_CROWN_M      = 12.0      # largest expected crown diameter (metres)
    OVERLAP_SAFETY   = 1.15      # multiply the crown-derived overlap by this
    OVERLAP          = 256       # fallback overlap (px)

    BATCH_SIZE  = 4
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 4          # tiles queued per worker (keeps GPU fed)

    # Transient-I/O resilience (network mounts, 9p, NAS): failed window reads
    # are retried with backoff and a reopened file handle before giving up.
    IO_RETRIES     = 3
    IO_RETRY_WAIT_S = 2.0

    # ---- Detection ---------------------------------------------------------
    # Derived for the DEPLOYED checkpoint on the GE val split, and cross-checked
    # against false positives on the held-out desert tiles. Both axes matter:
    # F1 is computed on labelled farmland and is blind to the desert error
    # mode, which is most of the country by area.
    #
    #   score   precision  recall     F1   FP per palm-free tile
    #   0.25      0.9236   0.9241  0.9239         0.071
    #   0.30      0.9334   0.9154  0.9243         0.055   <- F1 optimum
    #   0.35      0.9409   0.9062  0.9232         0.048   (previous default)
    #
    # 0.30 was the F1 optimum on that table and was deployed for the first
    # national run. It is NOT the right operating point, and the reason is a
    # domain gap the GE split cannot show.
    #
    # WHY 0.15 NOW
    #   The GE test split is plantation-dominated. Applied to a settlement
    #   (UAE_373, Ras al-Khaimah), the same model at 0.15 finds 14.4% more
    #   crowns than at 0.30 -- where the GE calibration predicted about 3%.
    #   Visual inspection of the additions on the imagery confirmed them as
    #   real palms in gardens, yards and roadside plantings: circularity
    #   p50 = 0.893 and diameter p50 = 5.84 m, indistinguishable from the
    #   confirmed population, and 83.6% lie within 12 m of a crown the 0.30
    #   run had already accepted. They are the missing members of clusters the
    #   model was already finding, not a new and dubious class of detection.
    #
    #   Measured on UAE_373, against this same code at 0.30 (81,478):
    #       0.25  84,733   +4.0%
    #       0.20  88,438   +8.5%
    #       0.15  93,174  +14.4%
    #   and on 2,098 held-out palm-free desert tiles the fine-tune never saw:
    #       thr    det/tile   tiles with any FP
    #       0.30    0.0558          1.86%
    #       0.25    0.0705          2.10%
    #       0.20    0.0958          2.53%
    #       0.15    0.1287          3.19%
    #   Every step adds roughly 70-77% real crowns to 23-30% false ones, so
    #   each is better than break-even; and even at 0.15, 96.8% of palm-free
    #   tiles stay completely clean, with p95 detections per tile still zero.
    #
    #   0.15 is chosen deliberately as the LOWEST point worth considering
    #   rather than as the final answer, because `score` travels with every
    #   detection: raising the threshold afterwards is a filter on the
    #   delivered layer, while lowering it costs another full re-run. The
    #   final rule is expected to be neighbourhood-conditioned rather than a
    #   flat cut -- keep score >= 0.30, or score >= 0.15 with a neighbour
    #   within ~12 m -- since desert false positives are isolated by
    #   construction while the real additions are clustered.
    #
    # THE MASK GATES ARE NOT THE LIMIT, measured on production imagery:
    # of 117,395 raw detections in UAE_373 at 0.15, the shape gate removed
    # 109, the area gate 0, and polygon NMS 248. The threshold was always the
    # whole story.
    #
    # RE-DERIVE THIS whenever the checkpoint changes:
    #   configs/Custom/Finetune_HN/calibrate_threshold.py  (farmland recall)
    #   configs/Custom/Finetune_HN/eval_hard_negatives.py  (desert cost)
    # and do not set it from the first alone.
    SCORE_THR   = 0.15

    # Mixed-precision inference (torch.autocast float16). ~1.5-2x faster on
    # RTX-class GPUs with negligible accuracy change for detection. If an AMP
    # batch errors, the pipeline retries it in FP32 and disables AMP for the
    # rest of the run (logged loudly).
    USE_AMP = True

    # ---- Geometry ----------------------------------------------------------
    #   'circle'  -> area-preserving circle at the mask centroid (default).
    #   'ellipse' -> best-fit ellipse with an axis-ratio cap.
    #   'polygon' -> simplified outline of the actual mask.
    GEOMETRY_MODE   = 'circle'
    CIRCLE_QUAD_SEGS = 16
    ELLIPSE_VERTICES = 48
    ELLIPSE_MIN_AXIS_RATIO = 0.35

    MORPH_KERNEL_PX     = 3
    APPROX_EPS_FRAC     = 0.002
    SIMPLIFY_TOL_PX     = 0.5
    MIN_MASK_AREA_PX    = 25

    # ---- Shape / size filtering -------------------------------------------
    # SHAPE_GATE selects which measured shape metrics may REJECT a detection
    # (see SHAPE_GATE_MODES):
    #   'circularity' -> 4*pi*A/P^2 >= CIRCULARITY_MIN            (default)
    #   'solidity'    -> area / convex-hull area >= SOLIDITY_MIN
    #   'axis_ratio'  -> short/long side of minAreaRect >= AXIS_RATIO_MIN
    #   'robust'      -> solidity AND axis_ratio (neither uses a perimeter)
    #   'all'         -> all three
    # ALL THREE are always computed and always written to the output
    # attributes, whichever ones gate. That is deliberate: it means a
    # threshold can be re-examined on the delivered layer in GIS rather than
    # by re-running inference over the country.
    #
    # Circularity is perimeter-based, and a digitised perimeter staircases, so
    # the same physical crown scores lower at coarse GSD than at fine -- a
    # descriptor that travels badly as a THRESHOLD across the 5/15/30 cm
    # sources this project spans. A perfect disc scores 0.674 at radius 3 px
    # and 0.887 at radius 40.
    #
    # 'robust' is the GSD-stable alternative, and it needs BOTH metrics: they
    # catch different failures and neither is sufficient alone. Solidity sees
    # concavity (merged crowns, L-shapes) but scores a straight bar 1.000
    # because a bar is convex; the axis ratio sees elongation but scores an
    # L-shape 1.000. See axis_ratio_of for the measured table.
    #
    # Default stays 'circularity' so the completed national run remains
    # reproducible. Change it deliberately, after looking at what the sweep in
    # measure_postproc_recall.py reports.
    SHAPE_GATE            = 'circularity'
    CIRCULARITY_MIN       = 0.60
    CIRCULARITY_SMOOTH_PX = 0     # e.g. 3 reduces pixel-staircase bias
    SOLIDITY_MIN          = 0.85
    AXIS_RATIO_MIN        = 0.55

    # Diameter gate in METRES. Default OFF; set from the reported percentiles
    # after a pilot run on a few folders.
    ENABLE_DIAMETER_FILTER = False
    DIAMETER_MIN_M         = 1.5
    DIAMETER_MAX_M         = 12.0

    # ---- Dedup -------------------------------------------------------------
    # Within a unit: polygon-IoU NMS over ownership survivors.
    GLOBAL_DEDUP_IOU = 0.45
    # Between units (merge step): centroid NMS restricted to detections whose
    # centroid lies within BORDER_BAND_M of the unit boundary, and only pairs
    # from DIFFERENT units may suppress each other (real neighbours inside a
    # unit are never eaten). DIST is the suppression radius.
    CROSS_UNIT_DEDUP        = True
    CROSS_UNIT_DEDUP_DIST_M = 3.0
    BORDER_BAND_M           = 20.0

    # Neighbour-density attribute written to the merged master: for each palm,
    # the number of other detected palms within this radius (metres). Cheap,
    # and lets the analysis stratify cultivated plantations (dense, regular)
    # from isolated detections in open desert, where false positives from
    # palm-like shrubs and native trees concentrate. 0 disables.
    NEIGHBOR_RADIUS_M = 50.0

    # Common projected CRS for the merged master and all metric math.
    # UAE: UTM zone 40N. (Western Abu Dhabi is nominally zone 39, but a single
    # zone keeps the master seamless; distance distortion is negligible for
    # the dedup radii used here.)
    TARGET_CRS = 'EPSG:32640'

    # ---- Resume / fault tolerance -----------------------------------------
    RESUME = True                # skip units whose stats JSON sidecar exists
    GPU_OOM_FALLBACK = True

    # For non-uint8 imagery: one contrast stretch per unit, computed from
    # decimated overviews, applied identically to every tile.
    STRETCH_FROM_OVERVIEW = True

    # ---- Scan --------------------------------------------------------------
    SCAN_RATE_TILES_PER_S = 3.0  # assumed throughput for the ETA estimate

    # ---- Misc --------------------------------------------------------------
    LOG_LEVEL   = 'INFO'


# =============================================================================
#                     LOGGING + PROVENANCE + ENVIRONMENT
# =============================================================================
def setup_logging(level: str, log_dir=None, tag=''):
    """Console logging, plus a persistent per-run log file when log_dir is
    given (survives terminal loss on multi-day runs)."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        handlers.append(logging.FileHandler(log_dir / f'{tag}_{ts}.log'))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers,
        force=True,
    )
    return logging.getLogger('palm_infer')


LOG = logging.getLogger('palm_infer')


def effective_config(cfg) -> dict:
    """Public CONFIG attributes as a plain dict (provenance snapshot)."""
    return {k: getattr(cfg, k) for k in sorted(vars(cfg))
            if k.isupper() and not k.startswith('_')}


def config_hash(cfg) -> str:
    """Stable short hash of every setting that affects detections. Stored in
    each unit's sidecar; merge refuses to silently mix mismatched units."""
    blob = json.dumps(effective_config(cfg), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def tune_gdal_env():
    """Sane GDAL defaults for many-small-files workloads. Existing user
    settings are respected (setdefault only)."""
    os.environ.setdefault('GDAL_CACHEMAX', '512')             # MB, per process
    os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    os.environ.setdefault('GDAL_MAX_DATASET_POOL_SIZE', '512')


# =============================================================================
#                    DISCOVERY: images -> processing units
# =============================================================================
def discover_images(input_path: str, exts) -> list:
    """Recursive raster discovery, case-insensitive on extension (Windows
    tooling produces .TIF as often as .tif)."""
    p = Path(input_path)
    exts = tuple(e.lower() for e in exts)
    if p.is_file():
        return [p] if p.suffix.lower() in exts else []
    if not p.is_dir():
        raise FileNotFoundError(f'INPUT_PATH does not exist: {input_path}')
    found = (f for f in p.rglob('*')
             if f.is_file() and f.suffix.lower() in exts)
    return [Path(f) for f in sorted({f.resolve() for f in found})]


def group_units(images, input_root, unit_mode: str):
    """Group rasters into processing units.

    'folder': one unit per directory that directly contains rasters; unit
    name is the directory path relative to the input root ('_root' for files
    sitting directly in the root).
    'image' : one unit per raster; the name embeds the relative path so
    identically-named tiles in different folders can never collide.
    Returns list of (unit_name, [paths]) sorted by name (deterministic for
    sharding across machines).
    """
    root = Path(input_root).resolve()
    units = {}
    if unit_mode == 'image':
        for img in images:
            try:
                rel = img.resolve().relative_to(root)
                name = str(rel.with_suffix('')).replace(os.sep, '__')
            except ValueError:
                name = img.stem
            units.setdefault(name, []).append(img)
    else:
        for img in images:
            parent = img.parent
            try:
                rel = parent.relative_to(root)
                name = '_root' if str(rel) == '.' \
                    else str(rel).replace(os.sep, '__')
            except ValueError:
                name = parent.name
            units.setdefault(name, []).append(img)
    return sorted((k, sorted(v)) for k, v in units.items())


def apply_shard(units, shard_spec):
    """--shard K/N -> keep units with (index % N) == K-1. Deterministic
    because `units` is name-sorted."""
    if not shard_spec:
        return units
    try:
        k, n = (int(t) for t in shard_spec.split('/'))
        assert 1 <= k <= n
    except Exception:
        raise SystemExit(f'--shard must look like 1/2, got: {shard_spec}')
    return [u for i, u in enumerate(units) if (i % n) == (k - 1)]


# =============================================================================
#                          GEO HELPERS
# =============================================================================
def metre_gsd(transform, crs, center_lat=None):
    """Ground sample distance in METRES per pixel, (gsd_x_m, gsd_y_m).

    Projected CRS: pixel size x the CRS linear-unit factor.
    Geographic CRS: degrees converted at the given latitude. Never returns
    degree-sized numbers as if they were metres.
    """
    px_w = abs(transform.a)
    px_h = abs(transform.e)
    if crs is None:
        return px_w, px_h            # unit unknown; caller warns
    try:
        if crs.is_geographic:
            lat = 0.0 if center_lat is None else float(center_lat)
            mx = 111_320.0 * max(np.cos(np.deg2rad(lat)), 1e-6)
            my = 110_540.0
            return px_w * mx, px_h * my
        try:
            _, factor = crs.linear_units_factor
        except Exception:
            factor = 1.0
        return px_w * factor, px_h * factor
    except Exception:
        return px_w, px_h


def resolve_bands(raster_path, bands_cfg, alpha_cfg):
    """Decide which 1-indexed bands feed the model as R,G,B and which band
    (if any) is the alpha/validity mask. Metadata only; reads no pixels."""
    with rasterio.open(str(raster_path)) as src:
        count = src.count
        try:
            cis = [ci.name.lower() if ci is not None else ''
                   for ci in src.colorinterp]
        except Exception:
            cis = [''] * count

    if bands_cfg is not None:
        rgb = list(bands_cfg)
        if len(rgb) != 3:
            raise ValueError(f'BANDS must list exactly 3 bands, got {rgb}')
        if any(b < 1 or b > count for b in rgb):
            raise ValueError(
                f'{Path(raster_path).name}: BANDS {rgb} out of range '
                f'(raster has {count} band(s))')
        if alpha_cfg is not None:
            alpha = None if alpha_cfg == 0 else int(alpha_cfg)
        else:
            alpha = next((i + 1 for i, n in enumerate(cis) if n == 'alpha'),
                         None)
        return rgb, alpha, f'explicit BANDS={rgb}, alpha={alpha}'

    red = next((i + 1 for i, n in enumerate(cis) if n == 'red'), None)
    grn = next((i + 1 for i, n in enumerate(cis) if n == 'green'), None)
    blu = next((i + 1 for i, n in enumerate(cis) if n == 'blue'), None)
    alpha_tag = next((i + 1 for i, n in enumerate(cis) if n == 'alpha'), None)

    if red and grn and blu:
        rgb = [red, grn, blu]
        note = f'auto: tagged RGB={rgb}'
    elif count == 3:
        rgb = [1, 2, 3]
        note = 'auto: untagged 3-band -> [1,2,3]'
    elif count == 4:
        rgb = [1, 2, 3]
        alpha_tag = alpha_tag or 4
        note = 'auto: 4-band -> RGB[1,2,3] + alpha[4]'
    elif count >= 5:
        rgb = [1, 2, 3]
        note = (f'auto: {count}-band untagged -> first 3 [1,2,3] '
                f'(set BANDS explicitly if this is wrong)')
    else:
        raise ValueError(
            f'{Path(raster_path).name}: needs >= 3 bands, found {count}')

    if alpha_cfg is not None:
        alpha = None if alpha_cfg == 0 else int(alpha_cfg)
    else:
        alpha = alpha_tag
    return rgb, alpha, note


# =============================================================================
#                    VIRTUAL MOSAIC (one folder = one raster)
# =============================================================================
class MosaicGrid:
    """A folder of GeoTIFFs treated as one virtual raster.

    Validates (never assumes): identical CRS, identical pixel size, and
    north-up transforms — raising ValueError otherwise. Sub-pixel grid
    offsets (arbitrary in GEE exports) are absorbed by snapping each member
    to the nearest integer pixel of the common grid; the largest absorbed
    offset is exposed as `max_snap_px` (worst case 0.5 px, i.e. half a GSD
    on the ground — negligible for crown-scale objects).

    Holds metadata only — picklable, safe to ship into DataLoader workers.
    """

    def __init__(self, paths, tol_px=None):    # tol_px kept for API compat
        if not paths:
            raise ValueError('MosaicGrid needs at least one raster')
        metas = []
        for p in paths:
            with rasterio.open(str(p)) as src:
                t = src.transform
                if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
                    raise ValueError(
                        f'{Path(p).name}: rotated/sheared transform — '
                        f'mosaicking requires north-up rasters')
                if t.a <= 0 or t.e >= 0:
                    raise ValueError(
                        f'{Path(p).name}: unexpected axis orientation '
                        f'(a={t.a}, e={t.e})')
                metas.append(dict(
                    path=str(p), transform=t, w=src.width, h=src.height,
                    crs=src.crs, dtype=src.dtypes[0], count=src.count,
                    nodata=src.nodata))

        ref = metas[0]
        self.crs = ref['crs']
        px_w, px_h = ref['transform'].a, -ref['transform'].e
        for m in metas[1:]:
            if m['crs'] != self.crs:
                raise ValueError(
                    f"{Path(m['path']).name}: CRS {m['crs']} differs from "
                    f"{self.crs} — cannot mosaic (use PROCESS_UNIT='image')")
            ta, te = m['transform'].a, -m['transform'].e
            if (abs(ta - px_w) / px_w > 1e-4) or (abs(te - px_h) / px_h > 1e-4):
                raise ValueError(
                    f"{Path(m['path']).name}: pixel size ({ta},{te}) differs "
                    f"from ({px_w},{px_h}) — cannot mosaic")

        x0 = min(m['transform'].c for m in metas)
        y0 = max(m['transform'].f for m in metas)
        self.sources = []
        self.max_snap_px = 0.0     # largest sub-pixel offset absorbed
        for m in metas:
            fc = (m['transform'].c - x0) / px_w
            fr = (y0 - m['transform'].f) / px_h
            coff, roff = round(fc), round(fr)
            self.max_snap_px = max(self.max_snap_px,
                                   abs(fc - coff), abs(fr - roff))
            self.sources.append(dict(
                path=m['path'], coff=int(coff), roff=int(roff),
                w=m['w'], h=m['h'], nodata=m['nodata']))

        self.W = max(s['coff'] + s['w'] for s in self.sources)
        self.H = max(s['roff'] + s['h'] for s in self.sources)
        self.dtype = ref['dtype']
        self.count = ref['count']
        self.nodata = ref['nodata']
        self.transform = Affine(px_w, 0.0, x0, 0.0, -px_h, y0)
        self.bounds = (x0, y0 - self.H * px_h, x0 + self.W * px_w, y0)

    def sources_in(self, x, y, w, h):
        out = []
        for s in self.sources:
            ix0 = max(x, s['coff'])
            ix1 = min(x + w, s['coff'] + s['w'])
            iy0 = max(y, s['roff'])
            iy1 = min(y + h, s['roff'] + s['h'])
            if ix1 > ix0 and iy1 > iy0:
                out.append((s, ix0, iy0, ix1, iy1))
        return out


class MosaicDataset(Dataset):
    """Yields RGB tiles (HWC uint8), top-left mosaic offset, and an emptiness
    flag.

    Tiles are assembled from all intersecting source rasters via windowed
    reads; a coverage mask marks pixels no source provides. Tiles with zero
    coverage are dropped at construction (never read, never sent to GPU).
    Each worker opens its own rasterio handles lazily.
    """

    def __init__(self, grid: MosaicGrid, tile_size, overlap,
                 rgb_bands, alpha_band=None, nodata_fill=0, stretch=None,
                 io_retries=3, io_retry_wait_s=2.0):
        self.grid = grid
        self.tile_size = int(tile_size)
        self.rgb_bands = list(rgb_bands)
        self.alpha_band = alpha_band
        self.nodata_fill = nodata_fill
        self.stretch = stretch
        self.io_retries = int(io_retries)
        self.io_retry_wait_s = float(io_retry_wait_s)
        self._handles = None

        stride = self.tile_size - int(overlap)
        if stride <= 0:
            raise ValueError('OVERLAP must be smaller than TILE_SIZE')

        self.coords = []
        for y in range(0, grid.H, stride):
            for x in range(0, grid.W, stride):
                if grid.sources_in(x, y, self.tile_size, self.tile_size):
                    self.coords.append((x, y))

    def __len__(self):
        return len(self.coords)

    def _handle(self, path):
        if self._handles is None:
            self._handles = {}
        ds = self._handles.get(path)
        if ds is None:
            ds = rasterio.open(path)
            self._handles[path] = ds
        return ds

    def _read_window(self, path, bands, win):
        """Windowed read with retry + handle reopen: survives transient
        failures on network/9p/NAS mounts instead of losing a whole folder."""
        last = None
        for attempt in range(self.io_retries + 1):
            try:
                return self._handle(path).read(bands, window=win)
            except Exception as exc:
                last = exc
                # Drop the (possibly wedged) handle; reopen on next attempt.
                ds = (self._handles or {}).pop(path, None)
                if ds is not None:
                    try:
                        ds.close()
                    except Exception:
                        pass
                if attempt < self.io_retries:
                    time.sleep(self.io_retry_wait_s * (2 ** attempt))
        raise RuntimeError(
            f'read failed after {self.io_retries + 1} attempts: '
            f'{path} {win}: {last}')

    def _read_bands(self, bands, x, y):
        """Assemble (len(bands), ts, ts) plus a coverage mask (ts, ts)."""
        ts = self.tile_size
        out = np.zeros((len(bands), ts, ts), dtype=np.dtype(self.grid.dtype))
        cov = np.zeros((ts, ts), dtype=bool)
        for s, ix0, iy0, ix1, iy1 in self.grid.sources_in(x, y, ts, ts):
            win = Window(ix0 - s['coff'], iy0 - s['roff'],
                         ix1 - ix0, iy1 - iy0)
            data = self._read_window(s['path'], bands, win)
            out[:, iy0 - y:iy1 - y, ix0 - x:ix1 - x] = data
            cov[iy0 - y:iy1 - y, ix0 - x:ix1 - x] = True
        return out, cov

    def _to_uint8(self, tile):
        if tile.dtype == np.uint8:
            return tile
        out = np.zeros_like(tile, dtype=np.uint8)
        for c in range(tile.shape[0]):
            band = tile[c].astype(np.float32)
            if self.stretch is not None:
                lo, hi = self.stretch[c]
            else:
                lo, hi = np.percentile(band, (2, 98))
            if hi <= lo:
                out[c] = 0
            else:
                out[c] = np.clip((band - lo) / (hi - lo) * 255.0,
                                 0, 255).astype(np.uint8)
        return out

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        raw, cov = self._read_bands(self.rgb_bands, x, y)

        # Validity: coverage AND alpha AND nodata checks. The source-nodata
        # test runs on the RAW values (before any uint8 stretch remaps them).
        valid = cov
        if self.alpha_band is not None:
            a, _ = self._read_bands([self.alpha_band], x, y)
            valid = valid & (a[0] > 0)
        if self.grid.nodata is not None:
            valid = valid & ~np.all(raw == self.grid.nodata, axis=0)

        tile = self._to_uint8(raw)
        if self.nodata_fill is not None:
            valid = valid & ~np.all(tile == self.nodata_fill, axis=0)

        tile = tile * valid[None, :, :].astype(tile.dtype)
        tile = np.ascontiguousarray(tile.transpose(1, 2, 0))
        is_empty = bool(tile.max() == 0)
        return tile, int(x), int(y), is_empty


def custom_collate(batch):
    # Top-level (picklable) collate: list of tuples -> tuple of lists.
    return list(zip(*batch))


def compute_unit_stretch(grid: MosaicGrid, rgb_bands,
                         max_sources=6, max_dim=1024):
    """One per-band (lo, hi) stretch for the whole unit (non-uint8 only),
    pooled over decimated reads of up to `max_sources` member rasters."""
    if np.dtype(grid.dtype) == np.uint8:
        return None
    step = max(1, len(grid.sources) // max_sources)
    samples = [[] for _ in rgb_bands]
    for s in grid.sources[::step][:max_sources]:
        with rasterio.open(s['path']) as src:
            scale = max(1, int(max(src.width, src.height) / max_dim))
            arr = src.read(rgb_bands,
                           out_shape=(len(rgb_bands),
                                      max(1, src.height // scale),
                                      max(1, src.width // scale)))
        for c in range(arr.shape[0]):
            band = arr[c].astype(np.float32).ravel()
            band = band[np.isfinite(band)]
            if band.size:
                samples[c].append(band)
    stretch = []
    for c in range(len(rgb_bands)):
        if not samples[c]:
            stretch.append((0.0, 1.0))
            continue
        pool = np.concatenate(samples[c])
        lo, hi = np.percentile(pool, (2, 98))
        if hi <= lo:
            hi = lo + 1.0
        stretch.append((float(lo), float(hi)))
    return stretch


# =============================================================================
#                       MASK -> GEOMETRY + METRICS
# =============================================================================
def clean_mask(mask_u8, kernel_px):
    """Morphological close then open to remove pinholes and speckle."""
    if kernel_px and kernel_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (kernel_px, kernel_px))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k)
    return mask_u8


def contour_pixels(contour, mask_u8):
    """Pixels of `mask_u8` lying inside `contour`, as a full-size uint8 mask.

    A predicted instance can survive cleaning as more than one blob; the
    pipeline keeps only the largest contour, so this is what "the crown"
    actually means downstream. Returning it explicitly keeps any consumer --
    notably the offline recall measurement, which matches masks against
    ground truth -- from scoring a footprint the pipeline would not have
    emitted.
    """
    out = np.zeros_like(mask_u8)
    cv2.drawContours(out, [contour], -1, 1, thickness=cv2.FILLED)
    out &= mask_u8
    return out


def mask_area_px(contour, mask_u8):
    """Area of a crown in whole pixels.

    WHY NOT cv2.contourArea
      findContours traces pixel CENTRES, so the polygon it returns is inset
      by half a pixel all the way round and its area understates the pixel
      footprint by roughly half the perimeter. The error is fractionally
      largest exactly where it does the most damage:

          radius   true px   contourArea    error
             3         29          20.0    -31.0%
             5         81          66.0    -18.5%
            10        317         288.0     -9.1%
            20       1257        1200.0     -4.5%
            40       5025        4912.0     -2.2%

      Two consequences, both biased against small crowns. A genuine young
      palm of 29 px is measured at 20 px and fails MIN_MASK_AREA_PX = 25.
      And every derived diameter is low, so a diameter gate inherits the
      same skew. Counting the pixels removes the bias at any size.

    The count is taken inside the contour's bounding box only, so cost
    scales with the crown rather than with the tile.
    """
    x, y, w, h = cv2.boundingRect(contour)
    sub = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(sub, [contour], -1, 1, thickness=cv2.FILLED,
                     offset=(-x, -y))
    if mask_u8 is not None:
        # Intersect with the mask so a concave crown wrapping around
        # background does not have that background counted as crown.
        sub &= mask_u8[y:y + h, x:x + w]
    return float(int(sub.sum()))


def solidity_of(contour):
    """Contour area / convex-hull area. 1.0 = fully convex.

    WHY THIS EXISTS ALONGSIDE circularity_of
      Circularity is 4*pi*A/P^2, and P is a digitised perimeter: the traced
      boundary staircases, so P is inflated and circularity is depressed by
      an amount that depends on crown size and on how ragged the mask edge
      is. It is a fine shape descriptor and a treacherous THRESHOLD, because
      the same physical crown scores differently at 5 cm and at 30 cm GSD.

      Measured on PERFECT digitised discs, which should score 1.0 exactly:

          radius    circularity   solidity
             3          0.674       0.833
             5          0.763       0.892
            10          0.832       0.947
            20          0.867       0.974
            40          0.887       0.986

      So CIRCULARITY_MIN = 0.60 leaves a flawless 3 px-radius circle only
      0.07 above the gate, and even a 40 px one never exceeds 0.89. At GE
      15 cm a 3 px radius is a 0.9 m crown, so the smallest palms sit right
      on the threshold and any raggedness at that size fails. Solidity uses
      no perimeter at all, so digitisation cannot inflate it.
      It separates the two failure modes circularity conflates: a crown that
      is round but ragged-edged keeps a high solidity, while two merged
      neighbours or an L-shaped false positive have a concave outline and
      score low. Both areas here come from cv2.contourArea, so the half-pixel
      inset that mask_area_px corrects for cancels in the ratio.
    """
    a = cv2.contourArea(contour)
    if a <= 0:
        return 0.0
    hull = cv2.convexHull(contour)
    ha = cv2.contourArea(hull)
    if ha <= 0:
        return 0.0
    return float(min(a / ha, 1.0))


def axis_ratio_of(contour):
    """Short axis / long axis of the minimum-area rectangle. 1.0 = square-ish.

    WHY SOLIDITY IS NOT ENOUGH ON ITS OWN
      Solidity measures concavity and nothing else, so a perfectly straight
      bar scores 1.000 -- it is convex. Substituting solidity for circularity
      would therefore stop rejecting elongated false positives entirely,
      which is a large share of what the shape gate was doing. Measured:

          shape              circularity   solidity   axis ratio
          disc                   0.855       0.965       1.000
          bar 4 x 60             0.145       1.000       0.051
          ellipse 5:1            0.411       0.955       0.200
          two merged crowns      0.555       0.877       0.500
          square                 0.785       1.000       1.000
          L-shape                0.582       0.846       1.000

      Circularity catches every elongated case but is perimeter-based, so it
      is GSD-dependent, and it passes a square at 0.785. Solidity catches
      concave blobs. The axis ratio catches elongation without a perimeter.
      Solidity and axis ratio together cover what circularity covers, and
      unlike circularity they mean the same thing at 5, 15 and 30 cm.
    """
    if len(contour) < 3:
        return 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    long_side = max(w, h)
    if long_side <= 0:
        return 0.0
    return float(min(w, h) / long_side)


def circularity_of(contour, mask_u8=None, smooth_px=0):
    """4*pi*area / perimeter^2. 1.0 = perfect circle. Optional pre-smoothing
    of the mask so pixel staircasing does not bias the gate at coarse GSD."""
    if smooth_px and smooth_px > 0 and mask_u8 is not None:
        k = smooth_px * 2 + 1
        blur = cv2.GaussianBlur(mask_u8 * 255, (k, k), 0)
        _, sm = cv2.threshold(blur, 127, 1, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(sm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            contour = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perim * perim))


# Ordered gate names. Every consumer reports drops against this list, so a
# stats sidecar, the log line and the offline sweep all use one vocabulary.
GATE_NAMES = ('no_contour', 'area', 'circularity', 'solidity', 'axis_ratio')

# Which shape metrics may REJECT, per SHAPE_GATE mode. Every metric is always
# measured and written out whatever the mode; this only decides what gates.
SHAPE_GATE_MODES = {
    'circularity': ('circularity',),                  # deployed default
    'solidity':    ('solidity',),
    'axis_ratio':  ('axis_ratio',),
    'robust':      ('solidity', 'axis_ratio'),        # perimeter-free pair
    'all':         ('circularity', 'solidity', 'axis_ratio'),
}


def analyse_mask(mask_u8, cfg):
    """Apply the per-detection mask gates to ONE instance mask.

    Returns (info, reason). `reason` is None when the detection survives, and
    otherwise names the gate that rejected it (one of GATE_NAMES). `info`
    carries everything downstream needs, so nothing is recomputed:
        contour, area_px (true pixel count), circularity, solidity, centroid

    THIS IS THE ONLY PLACE THE GATES ARE APPLIED. process_unit calls it, and
    so does the offline recall measurement, which is the point: a measurement
    that reimplements the gates measures the reimplementation. Adding a gate
    means editing this function and GATE_NAMES, and every consumer follows.
    """
    m = clean_mask(mask_u8, cfg.MORPH_KERNEL_PX)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 'no_contour'
    cnt = max(contours, key=cv2.contourArea)

    area_px = mask_area_px(cnt, m)
    if area_px < cfg.MIN_MASK_AREA_PX:
        return None, 'area'

    circ = circularity_of(cnt, mask_u8=m,
                          smooth_px=getattr(cfg, 'CIRCULARITY_SMOOTH_PX', 0))
    sol = solidity_of(cnt)
    ar = axis_ratio_of(cnt)

    # Which metrics actually gate is configurable; ALL of them are measured
    # and written to the output whatever the mode, so a threshold can be
    # revisited in GIS on the delivered layer rather than by re-running the
    # country.
    mode = getattr(cfg, 'SHAPE_GATE', 'circularity')
    try:
        active = SHAPE_GATE_MODES[mode]
    except KeyError:
        raise ValueError(
            f'SHAPE_GATE={mode!r} is not one of '
            f'{sorted(SHAPE_GATE_MODES)}') from None
    if 'circularity' in active and circ < cfg.CIRCULARITY_MIN:
        return None, 'circularity'
    if 'solidity' in active and sol < getattr(cfg, 'SOLIDITY_MIN', 0.0):
        return None, 'solidity'
    if 'axis_ratio' in active and ar < getattr(cfg, 'AXIS_RATIO_MIN', 0.0):
        return None, 'axis_ratio'

    cx, cy = _contour_centroid(cnt)
    return dict(mask=contour_pixels(cnt, m), contour=cnt, area_px=area_px,
                circularity=circ, solidity=sol, axis_ratio=ar,
                cx=cx, cy=cy), None


def _largest_polygon(geom):
    """Largest Polygon from whatever make_valid returned, or None."""
    from shapely.geometry.base import BaseMultipartGeometry
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if geom.area > 0 else None
    if isinstance(geom, BaseMultipartGeometry):
        polys = [g for g in geom.geoms
                 if isinstance(g, Polygon) and g.area > 0]
        if not polys:
            return None
        return max(polys, key=lambda g: g.area)
    return None


def contour_to_simplified_polygon(contour, x_off, y_off,
                                  approx_eps_frac, simplify_tol_px):
    perim = cv2.arcLength(contour, True)
    eps = max(approx_eps_frac * perim, 0.5)
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) < 3:
        return None
    pts = approx[:, 0, :].astype(np.float64)
    pts[:, 0] += x_off
    pts[:, 1] += y_off
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = _largest_polygon(make_valid(poly))
        if poly is None:
            return None
    if simplify_tol_px and simplify_tol_px > 0:
        poly = poly.simplify(simplify_tol_px, preserve_topology=True)
        poly = _largest_polygon(poly) if not isinstance(poly, Polygon) else poly
        if poly is None:
            return None
    if poly.is_empty or not poly.is_valid or poly.area <= 0:
        return None
    return poly


def _contour_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] > 1e-6:
        return M['m10'] / M['m00'], M['m01'] / M['m00']
    pts = contour[:, 0, :]
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def contour_to_circle(contour, x_off, y_off, area_px, quad_segs):
    """Area-preserving circle centred on the mask centroid."""
    cx, cy = _contour_centroid(contour)
    r = float(np.sqrt(max(area_px, 1.0) / np.pi))
    from shapely.geometry import Point
    circ = Point(cx + x_off, cy + y_off).buffer(r, quad_segs=quad_segs)
    return circ if (circ.is_valid and circ.area > 0) else None


def contour_to_ellipse(contour, x_off, y_off, n_vertices, min_axis_ratio):
    if len(contour) < 5:               # fitEllipse needs >= 5 points
        return None
    (ecx, ecy), (MA, ma), angle = cv2.fitEllipse(contour)
    major = max(MA, ma)
    minor = min(MA, ma)
    if major <= 0:
        return None
    minor = max(minor, min_axis_ratio * major)
    a, b = major / 2.0, minor / 2.0
    theta = np.deg2rad(angle)
    t = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    ex = a * np.cos(t)
    ey = b * np.sin(t)
    rx = ex * np.cos(theta) - ey * np.sin(theta) + ecx + x_off
    ry = ex * np.sin(theta) + ey * np.cos(theta) + ecy + y_off
    poly = Polygon(np.column_stack([rx, ry]))
    if not poly.is_valid:
        poly = _largest_polygon(make_valid(poly))
    if poly is None or poly.is_empty or poly.area <= 0:
        return None
    return poly


def build_geometry(contour, x_off, y_off, area_px, cfg):
    mode = getattr(cfg, 'GEOMETRY_MODE', 'circle')
    if mode == 'circle':
        return contour_to_circle(contour, x_off, y_off, area_px,
                                 cfg.CIRCLE_QUAD_SEGS)
    if mode == 'ellipse':
        g = contour_to_ellipse(contour, x_off, y_off, cfg.ELLIPSE_VERTICES,
                               cfg.ELLIPSE_MIN_AXIS_RATIO)
        return g if g is not None else contour_to_circle(
            contour, x_off, y_off, area_px, cfg.CIRCLE_QUAD_SEGS)
    return contour_to_simplified_polygon(
        contour, x_off, y_off, cfg.APPROX_EPS_FRAC, cfg.SIMPLIFY_TOL_PX)


def equiv_diameter_m(area_px, pixel_area_m2):
    """Diameter (m) of the circle with the same area as the crown mask."""
    area_m2 = area_px * pixel_area_m2
    if area_m2 <= 0:
        return 0.0
    return float(2.0 * np.sqrt(area_m2 / np.pi))


# =============================================================================
#                          DEDUP PRIMITIVES
# =============================================================================
def global_polygon_nms(records, iou_thr):
    """Greedy NMS over polygons in pixel space (true polygon IoU, STRtree)."""
    if not records:
        return []
    from shapely.strtree import STRtree

    polys = [r['poly_px'] for r in records]
    order = sorted(range(len(records)),
                   key=lambda i: records[i]['score'], reverse=True)

    tree = STRtree(polys)
    suppressed = [False] * len(records)
    kept = []

    for i in order:
        if suppressed[i]:
            continue
        kept.append(records[i])
        gi = polys[i]
        ai = gi.area
        for j in tree.query(gi):
            j = int(j)
            if j == i or suppressed[j]:
                continue
            gj = polys[j]
            if not gi.intersects(gj):
                continue
            inter = gi.intersection(gj).area
            union = ai + gj.area - inter
            if union > 0 and (inter / union) >= iou_thr:
                if records[j]['score'] <= records[i]['score']:
                    suppressed[j] = True
    return kept


def cross_unit_centroid_nms(xy, scores, unit_idx, dist_thr):
    """Greedy centroid NMS in a metric CRS, restricted to pairs from
    DIFFERENT units — genuine close neighbours inside one unit are never
    suppressed. Returns a boolean keep-mask."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=bool)
    xy = np.asarray(xy, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    unit_idx = np.asarray(unit_idx)
    order = np.argsort(-scores)
    keep = np.ones(n, dtype=bool)
    suppressed = np.zeros(n, dtype=bool)

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None

    if cKDTree is not None:
        tree = cKDTree(xy)
        for i in order:
            if suppressed[i]:
                continue
            for j in tree.query_ball_point(xy[i], r=dist_thr):
                if j == i or suppressed[j] or unit_idx[j] == unit_idx[i]:
                    continue
                if scores[j] <= scores[i]:
                    suppressed[j] = True
                    keep[j] = False
        return keep

    # Grid-hash fallback (no SciPy). Bucket size = dist_thr.
    cell = max(dist_thr, 1e-6)
    grid = {}
    for idx in range(n):
        gx, gy = int(xy[idx, 0] // cell), int(xy[idx, 1] // cell)
        grid.setdefault((gx, gy), []).append(idx)
    for i in order:
        if suppressed[i]:
            continue
        gx, gy = int(xy[i, 0] // cell), int(xy[i, 1] // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if j == i or suppressed[j] or unit_idx[j] == unit_idx[i]:
                        continue
                    d2 = ((xy[i, 0] - xy[j, 0]) ** 2
                          + (xy[i, 1] - xy[j, 1]) ** 2)
                    if d2 <= dist_thr * dist_thr and scores[j] <= scores[i]:
                        suppressed[j] = True
                        keep[j] = False
    return keep


class _AmpState:
    """Process-wide AMP switch: once a batch fails under autocast, AMP is
    disabled for the rest of the run (correctness beats speed)."""
    enabled = False


def _run_detector(model, imgs):
    from mmdet.apis import inference_detector
    if _AmpState.enabled and torch.cuda.is_available():
        with torch.autocast('cuda', dtype=torch.float16):
            return inference_detector(model, imgs)
    return inference_detector(model, imgs)


def safe_inference(model, imgs, oom_fallback=True):
    """Batched inference with two safety nets:
    * AMP failure -> retry this batch FP32 and disable AMP from then on;
    * CUDA OOM    -> retry the batch tile-by-tile (bs=1), skip a tile that
                     still OOMs rather than aborting the whole unit."""
    try:
        return _run_detector(model, imgs)
    except RuntimeError as exc:
        is_oom = 'out of memory' in str(exc).lower()
        if _AmpState.enabled and not is_oom:
            LOG.warning(f'    AMP batch failed ({exc}); disabling AMP and '
                        f'retrying this batch in FP32.')
            _AmpState.enabled = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return safe_inference(model, imgs, oom_fallback=oom_fallback)
        if not (oom_fallback and is_oom):
            raise
        LOG.warning('    CUDA OOM on batch; retrying tile-by-tile (bs=1).')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results = []
        for im in imgs:
            try:
                results.append(_run_detector(model, [im])[0])
            except RuntimeError as exc2:
                if 'out of memory' in str(exc2).lower():
                    LOG.error('    tile OOM at bs=1; skipping tile.')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    results.append(None)
                else:
                    raise
        return results


# =============================================================================
#                        SINGLE-UNIT INFERENCE
# =============================================================================
def process_unit(model, unit_name, paths, cfg: 'CONFIG'):
    """Run the full pipeline on one unit (folder mosaic or single raster).
    Returns (gdf, stats_dict)."""
    t0 = time.time()

    grid = MosaicGrid(paths)

    # Metre GSD (geographic-CRS aware) drives every metric quantity.
    center_lat = None
    try:
        if grid.crs is not None and grid.crs.is_geographic:
            center_lat = (grid.bounds[1] + grid.bounds[3]) / 2.0
    except Exception:
        pass
    gsd_x, gsd_y = metre_gsd(grid.transform, grid.crs, center_lat)
    pixel_area_m2 = gsd_x * gsd_y
    if grid.crs is None or pixel_area_m2 <= 0:
        LOG.warning(f'    {unit_name}: no CRS / zero pixel size; '
                    f'metric quantities will be meaningless.')

    overlap_px = cfg.OVERLAP
    if getattr(cfg, 'ADAPTIVE_OVERLAP', False) and pixel_area_m2 > 0:
        gsd = max(gsd_x, gsd_y)
        crown_px = cfg.MAX_CROWN_M / gsd
        overlap_px = int(np.ceil(crown_px * cfg.OVERLAP_SAFETY))
        overlap_px = max(overlap_px, 16)
        overlap_px = min(overlap_px, cfg.TILE_SIZE - cfg.TILE_SIZE // 4)
        adaptive_note = f'GSD {gsd * 100:.1f} cm/px -> adaptive overlap'
    else:
        adaptive_note = 'fixed overlap'

    # Ownership boxes must TILE the mosaic exactly, and that only holds for an
    # even overlap. A tile at x owns [x+margin, x+TILE-margin) with
    # margin = overlap//2, and the next tile starts at x + TILE - overlap. Its
    # ownership therefore opens at x + TILE - ceil(overlap/2) while its
    # predecessor's closes at x + TILE - floor(overlap/2): identical when
    # overlap is even, but one pixel apart when it is odd, leaving a 1 px
    # column owned TWICE. The duplicate pair is near-coincident so polygon NMS
    # almost always absorbs it -- which is exactly why this went unnoticed --
    # but relying on a downstream filter to repair a partition that should be
    # exact is not a property worth keeping. The adaptive branch above rounds
    # up (12 m / 0.135 m * 1.15 = 103 px, odd, on this project's GE imagery);
    # this catches a hand-set CONFIG.OVERLAP too.
    if overlap_px % 2:
        overlap_px += 1
    LOG.info(f'    {adaptive_note} {overlap_px} px '
             f'({overlap_px * max(gsd_x, gsd_y):.1f} m); ownership margin '
             f'{overlap_px // 2} px')

    rgb_bands, alpha_band, band_note = resolve_bands(
        paths[0], cfg.BANDS, cfg.ALPHA_BAND)
    LOG.info(f'    {len(paths)} raster(s), mosaic {grid.W}x{grid.H} px; '
             f'bands: {band_note}')
    if grid.max_snap_px > getattr(cfg, 'GRID_SNAP_WARN_PX', 0.25):
        LOG.warning(f'    grid snap absorbed a {grid.max_snap_px:.2f} px '
                    f'sub-pixel offset ({grid.max_snap_px * max(gsd_x, gsd_y) * 100:.1f} cm '
                    f'on the ground) — negligible, noted for provenance.')

    stretch = None
    if getattr(cfg, 'STRETCH_FROM_OVERVIEW', True):
        stretch = compute_unit_stretch(grid, rgb_bands)
        if stretch is not None:
            LOG.info(f'    stretch (per-band lo/hi): '
                     f'{[(round(l, 1), round(h, 1)) for l, h in stretch]}')

    dataset = MosaicDataset(
        grid, cfg.TILE_SIZE, overlap_px,
        rgb_bands=rgb_bands, alpha_band=alpha_band,
        nodata_fill=getattr(cfg, 'NODATA_FILL_VALUE', None),
        stretch=stretch,
        io_retries=getattr(cfg, 'IO_RETRIES', 3),
        io_retry_wait_s=getattr(cfg, 'IO_RETRY_WAIT_S', 2.0))
    loader_kw = dict(batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                     collate_fn=custom_collate, shuffle=False,
                     pin_memory=False)
    if cfg.NUM_WORKERS > 0:
        loader_kw['prefetch_factor'] = getattr(cfg, 'PREFETCH_FACTOR', 4)
    loader = DataLoader(dataset, **loader_kw)

    margin = overlap_px // 2
    # A crown clipped by a tile edge has its centre within one crown RADIUS of
    # that edge, so the margin must be at least a radius wide for the clipped
    # copy to fall outside the tile's ownership box and be dropped in favour
    # of the whole copy from the neighbouring tile. If it is not, seams emit
    # truncated crowns alongside the real ones -- undersized, and only
    # sometimes absorbed by NMS. Worth stating out loud rather than trusting
    # that MAX_CROWN_M was set generously.
    if pixel_area_m2 > 0:
        crown_radius_px = (cfg.MAX_CROWN_M / max(gsd_x, gsd_y)) / 2.0
        if margin < crown_radius_px:
            LOG.warning(
                f'    ownership margin {margin} px is narrower than the '
                f'{crown_radius_px:.0f} px radius of a {cfg.MAX_CROWN_M} m '
                f'crown; crowns on tile seams may be emitted truncated. '
                f'Raise OVERLAP_SAFETY or MAX_CROWN_M.')

    records = []
    raw_count = 0
    diam_samples = []
    # Per-gate rejection tally. 'after_filters' alone cannot say whether a
    # missing crown was rejected for its shape, its size, or because another
    # tile owned it, which is precisely the question that matters when the
    # count is the published product.
    drops = {'ownership': 0, 'diameter': 0, 'geometry': 0, 'georef': 0}
    for _g in GATE_NAMES:
        drops[_g] = 0

    pbar = tqdm(loader, total=len(loader), unit='batch',
                desc=f'    {unit_name[:28]:28s}', leave=False,
                dynamic_ncols=True)
    for batch in pbar:
        imgs, xs, ys, empties = batch
        live = [(im, xs[i], ys[i]) for i, im in enumerate(imgs)
                if not empties[i]]
        if not live:
            continue
        live_imgs = [t[0] for t in live]

        results = safe_inference(
            model, live_imgs,
            oom_fallback=getattr(cfg, 'GPU_OOM_FALLBACK', True))

        for k, result in enumerate(results):
            if result is None:          # tile skipped after OOM at bs=1
                continue
            tx, ty = live[k][1], live[k][2]

            # Seam-free ownership box for this tile.
            vx_min = margin if tx > 0 else 0
            vy_min = margin if ty > 0 else 0
            vx_max = cfg.TILE_SIZE - margin if (tx + cfg.TILE_SIZE) < grid.W \
                else cfg.TILE_SIZE
            vy_max = cfg.TILE_SIZE - margin if (ty + cfg.TILE_SIZE) < grid.H \
                else cfg.TILE_SIZE

            pred = result.pred_instances
            keep = pred.scores > cfg.SCORE_THR          # gate 1: confidence
            if keep.sum() == 0:
                continue

            if not hasattr(pred, 'masks') or pred.masks is None:
                raise RuntimeError(
                    'Model output has no instance masks. This pipeline needs '
                    'a Mask R-CNN-style segmentation model; check CONFIG_FILE '
                    'and that the checkpoint matches it.')

            scores = pred.scores[keep].cpu().numpy()
            bboxes = pred.bboxes[keep].cpu().numpy()
            masks = pred.masks[keep].cpu().numpy().astype(np.uint8)
            raw_count += len(scores)

            for j in range(len(scores)):
                # Cheap pre-reject on the BOX centre before any mask work.
                # The mask lies inside its own box, so its centroid cannot be
                # further from the box centre than HALF the longer box side;
                # clean_mask's closing can push the boundary out by at most
                # the kernel radius, which is added on. Anything beyond that
                # cannot possibly own, so it is discarded without paying for
                # the mask work. Ownership itself is decided below, on the
                # centroid.
                #
                # Half, not a whole box side: a full side exceeds the
                # ownership margin for this project's imagery (a 12 m crown
                # is ~89 px at 13.5 cm, against a 52 px margin), which would
                # make the test vacuously true and pay full mask cost for
                # every border detection.
                bx = bboxes[j]
                bcx = (bx[0] + bx[2]) / 2.0
                bcy = (bx[1] + bx[3]) / 2.0
                slack = (max(bx[2] - bx[0], bx[3] - bx[1]) / 2.0
                         + max(cfg.MORPH_KERNEL_PX, 1))
                if not (vx_min - slack <= bcx < vx_max + slack
                        and vy_min - slack <= bcy < vy_max + slack):
                    drops['ownership'] += 1
                    continue

                info, reason = analyse_mask(masks[j], cfg)
                if reason is not None:
                    drops[reason] += 1
                    continue

                # Ownership decided on the MASK CENTROID, which is the point
                # the emitted geometry is actually built around (circle mode
                # centres on it). Deciding on the box centre instead let the
                # two disagree for any asymmetric mask, so a crown could be
                # owned by one tile while its geometry landed in the next --
                # counted twice at the seam, or not at all.
                cx, cy = info['cx'], info['cy']
                if not (vx_min <= cx < vx_max and vy_min <= cy < vy_max):
                    drops['ownership'] += 1
                    continue

                area_px = info['area_px']
                circ = info['circularity']
                diam_m = equiv_diameter_m(area_px, pixel_area_m2)
                if cfg.ENABLE_DIAMETER_FILTER:          # gate: size in metres
                    if not (cfg.DIAMETER_MIN_M <= diam_m
                            <= cfg.DIAMETER_MAX_M):
                        drops['diameter'] += 1
                        continue

                poly_px = build_geometry(info['contour'], tx, ty, area_px, cfg)
                if poly_px is None:
                    drops['geometry'] += 1
                    continue

                records.append({
                    'poly_px': poly_px,
                    'score': float(scores[j]),
                    'circularity': round(circ, 4),
                    'solidity': round(info['solidity'], 4),
                    'axis_ratio': round(info['axis_ratio'], 4),
                    'diam_m': round(diam_m, 3),
                    'area_px': area_px,
                })
                diam_samples.append(diam_m)

        try:
            pbar.set_postfix(kept=len(records), raw=raw_count, refresh=False)
        except (AttributeError, TypeError):
            pass

    n_before_dedup = len(records)
    records = global_polygon_nms(records, cfg.GLOBAL_DEDUP_IOU)
    n_after_dedup = len(records)

    # Georeference: mosaic pixel coords -> CRS coords via affine transform.
    # rasterio Affine [a,b,c,d,e,f] -> shapely affine [a,b,d,e,c,f]
    t = grid.transform
    shp_mat = [t.a, t.b, t.d, t.e, t.c, t.f]

    geoms, scores_out, circ_out, sol_out, diam_out = [], [], [], [], []
    ar_out = []
    for r in records:
        g = affine_transform(r['poly_px'], shp_mat)
        if g.is_empty or not g.is_valid:
            # Counted separately from 'geometry': this rejection happens
            # AFTER dedup, so folding it in would break the identity
            # raw_detections - after_filters == sum(pre-dedup drops).
            drops['georef'] += 1
            continue
        geoms.append(g)
        scores_out.append(r['score'])
        circ_out.append(r['circularity'])
        sol_out.append(r['solidity'])
        ar_out.append(r['axis_ratio'])
        diam_out.append(r['diam_m'])

    # solidity travels with every crown even when it is not the active gate,
    # so the delivered layer can be re-thresholded on shape in GIS without
    # re-running inference.
    gdf = gpd.GeoDataFrame(
        {
            'id': range(len(geoms)),
            'score': scores_out,
            'circularity': circ_out,
            'solidity': sol_out,
            'axis_ratio': ar_out,
            'diam_m': diam_out,
            'unit': [unit_name] * len(geoms),
            'geometry': geoms,
        },
        crs=grid.crs,
    )

    pct = {}
    if diam_samples:
        arr = np.array(diam_samples)
        for q in (1, 5, 50, 95, 99):
            pct[f'p{q}'] = round(float(np.percentile(arr, q)), 2)

    stats = {
        'unit': unit_name,
        'n_images': len(paths),
        'mosaic_px': [grid.W, grid.H],
        'max_snap_px': round(grid.max_snap_px, 3),
        'bounds': list(grid.bounds),
        'crs': str(grid.crs),
        'gsd_m': [round(gsd_x, 4), round(gsd_y, 4)],
        'tiles': len(dataset),
        'raw_detections': raw_count,
        'after_filters': n_before_dedup,
        'after_dedup': n_after_dedup,
        'final_count': len(geoms),
        # Where every detection that did not survive was lost. The pre-dedup
        # keys (everything except 'georef') sum to
        # raw_detections - after_filters; 'dropped_nms' covers the dedup step
        # and 'georef' the affine transform after it, so
        #   final_count == raw_detections - sum(dropped) - dropped_nms
        # is an identity the run can be checked against.
        'dropped': dict(drops),
        'dropped_nms': n_before_dedup - n_after_dedup,
        'overlap_px': int(overlap_px),
        'ownership_margin_px': int(margin),
        'shape_gate': getattr(cfg, 'SHAPE_GATE', 'circularity'),
        'diam_pct': pct,
        'seconds': round(time.time() - t0, 1),
        'tiles_per_s': round(len(dataset) / max(time.time() - t0, 1e-6), 2),
        'score_thr': cfg.SCORE_THR,
        'checkpoint': str(cfg.CHECKPOINT_FILE),
        'config_hash': config_hash(cfg),
        'amp': bool(_AmpState.enabled),
        'finished_utc': utc_now(),
    }
    return gdf, stats


# =============================================================================
#                      OUTPUT PATHS + ATOMIC WRITES
# =============================================================================
def unit_gpkg_path(out_dir: Path, unit_name: str) -> Path:
    return out_dir / f'{unit_name}_palms.gpkg'


def unit_stats_path(out_dir: Path, unit_name: str) -> Path:
    return out_dir / f'{unit_name}_stats.json'


# Equal-area CRS for crown-area statistics. TARGET_CRS (UTM 40N, central
# meridian 57E) carries the WHOLE country west of its central meridian --
# western Abu Dhabi sits ~5.5 deg off it, outside zone 40N's nominal 54-60E
# range -- so polygon areas measured in TARGET_CRS are inflated by roughly
# 0.7% in the west and understated near 57E. Geometry stays in TARGET_CRS
# (seamless, and what ArcGIS users expect); only the area column is measured
# here. Lambert azimuthal equal-area centred on the UAE is exact for area by
# construction, so the bias vanishes rather than merely shrinking.
EQUAL_AREA_CRS = ('+proj=laea +lat_0=24 +lon_0=54 +x_0=0 +y_0=0 '
                  '+datum=WGS84 +units=m +no_defs')


# Attribute schema of the merged master, in write order. Units produced by
# different pipeline versions carry different columns -- `solidity` was added
# after the first national run -- and merge appends them into ONE GPKG layer,
# whose schema is fixed by whichever unit happens to be written first.
#
# That makes a mixed-version output directory fail by ORDER, which is the
# worst way to fail: appending a frame that has a column the layer lacks
# raises FieldError ("Could not find field index for field 'solidity'") and
# aborts the merge part-way, while the reverse order succeeds silently and
# fills NaN. Sorting units alphabetically decides which happens, so the same
# data merges or crashes depending on unit names.
#
# Every frame is therefore conformed to this list before writing: missing
# columns are added as null, unexpected ones are dropped. Parquet gets the
# same treatment -- one file per unit there, but a partitioned read still
# needs a single schema.
MASTER_COLUMNS = ('id', 'score', 'circularity', 'solidity', 'axis_ratio',
                  'diam_m', 'area_m2', 'lon', 'lat', 'unit')


def master_columns(cfg):
    """MASTER_COLUMNS plus the neighbour-density column, whose NAME depends on
    NEIGHBOR_RADIUS_M (nbr_50m at the default 50 m). Deriving it here rather
    than hardcoding one keeps a changed radius -- or SciPy being absent, which
    skips the column entirely -- from silently dropping it at conform time."""
    cols = list(MASTER_COLUMNS)
    r = int(float(getattr(cfg, 'NEIGHBOR_RADIUS_M', 0) or 0))
    if r > 0:
        cols.insert(cols.index('unit'), f'nbr_{r}m')
    return cols


def conform_schema(gdf, columns):
    """Reindex a unit frame onto the master attribute schema: add missing
    columns as null, drop unexpected ones, fix the order."""
    out = gdf.copy()
    geom = out.geometry.name
    for c in columns:
        if c not in out.columns:
            out[c] = np.nan
    extra = [c for c in out.columns if c not in columns and c != geom]
    if extra:
        # Warned once per distinct column set, not once per unit: dropping an
        # attribute the analysis expected is not something to discover from a
        # missing field in QGIS months later, but 224 identical warnings would
        # be scrolled past just as effectively as none.
        seen = conform_schema.__dict__.setdefault('_warned', set())
        key = tuple(sorted(extra))
        if key not in seen:
            seen.add(key)
            LOG.warning(f'conform_schema is dropping column(s) not in the '
                        f'master schema: {extra}. Add them to '
                        f'MASTER_COLUMNS if they belong in the output.')
    return out[list(columns) + [geom]].set_geometry(geom)


def normalize_geometry(gdf):
    """Coerce a crown layer to valid, homogeneous MultiPolygon.

    WHY THIS EXISTS
      Crown polygons come out of mask contours as a MIX of Polygon and
      MultiPolygon. GDAL then registers the GPKG layer in
      gpkg_geometry_columns with the generic type 'GEOMETRY'. QGIS reads
      that; ARCGIS PRO DOES NOT -- Pro requires one declared geometry type
      per feature class and silently shows nothing otherwise. Coercing every
      row to MultiPolygon makes geopandas declare MULTIPOLYGON, which every
      reader accepts.

      make_valid() is applied first because contour-derived rings
      self-intersect often enough that Pro's geoprocessing tools fail
      mid-run on an otherwise readable layer. It can return a
      GeometryCollection (e.g. a polygon plus a dangling line), so the
      polygonal parts are extracted before the MultiPolygon wrap.

    Returns a copy; the input is not modified. Rows whose geometry is null
    or has no polygonal part after repair are DROPPED, and the count is
    logged -- silently keeping empty geometries would corrupt the layer for
    exactly the readers this function exists to satisfy.
    """
    if gdf is None or len(gdf) == 0:
        return gdf

    def _fix(g):
        if g is None or g.is_empty:
            return None
        if not g.is_valid:
            g = make_valid(g)
        gt = g.geom_type
        if gt == 'Polygon':
            return MultiPolygon([g])
        if gt == 'MultiPolygon':
            return g
        if gt == 'GeometryCollection':
            # make_valid on a self-intersecting ring can emit polygons mixed
            # with degenerate lines/points; keep only the polygonal parts.
            polys = [p for p in g.geoms if p.geom_type == 'Polygon']
            return MultiPolygon(polys) if polys else None
        return None                      # lines/points: not a crown

    out = gdf.copy()
    out['geometry'] = out.geometry.apply(_fix)
    bad = out.geometry.isna()
    if bad.any():
        LOG.warning(f'    normalize_geometry: dropped {int(bad.sum())} '
                    f'row(s) with no polygonal geometry after repair')
        out = out[~bad].copy()
    return out


def write_unit_outputs(out_dir: Path, unit_name: str, gdf, stats):
    """Atomic-ish: gpkg to tmp then rename; stats JSON written last, so the
    presence of the JSON certifies a complete gpkg (RESUME trusts the JSON)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = unit_gpkg_path(out_dir, unit_name)
    tmp = gpkg.with_name(gpkg.name + '.tmp.gpkg')
    if tmp.exists():
        tmp.unlink()
    if len(gdf):
        # MULTIPOLYGON, not the generic 'GEOMETRY' type geopandas declares for
        # a mixed Polygon/MultiPolygon column — see normalize_geometry().
        normalize_geometry(gdf).to_file(tmp, driver='GPKG', layer=unit_name)
        os.replace(tmp, gpkg)
    else:
        # Empty layer: preferred so merge sees a file per unit, but some
        # geopandas/ogr versions refuse empty writes — the sidecar's
        # final_count == 0 is authoritative either way and merge accepts a
        # missing gpkg for zero-count units.
        try:
            empty = gpd.GeoDataFrame(
                {'id': [], 'score': [], 'circularity': [], 'solidity': [],
                 'axis_ratio': [], 'diam_m': [], 'unit': []},
                geometry=gpd.GeoSeries([], crs=gdf.crs), crs=gdf.crs)
            # An empty layer must still DECLARE its type, or its
            # gpkg_geometry_columns row reads 'GEOMETRY' and Pro rejects the
            # file exactly as it rejects a mixed-type populated one.
            empty.to_file(tmp, driver='GPKG', layer=unit_name,
                          geometry_type='MULTIPOLYGON')
            os.replace(tmp, gpkg)
        except Exception as exc:
            LOG.warning(f'    empty-layer write not supported here ({exc}); '
                        f'relying on the stats sidecar (count=0).')
            if tmp.exists():
                tmp.unlink()
    sp = unit_stats_path(out_dir, unit_name)
    sp_tmp = sp.with_suffix('.json.tmp')
    with open(sp_tmp, 'w') as fh:
        json.dump(stats, fh, indent=1)
    os.replace(sp_tmp, sp)
    return gpkg


# =============================================================================
#                              COMMAND: scan
# =============================================================================
def cmd_scan(cfg, args):
    images = discover_images(cfg.INPUT_PATH,
                             tuple(e.lower() for e in cfg.IMAGE_EXTS))
    if not images:
        LOG.error(f'No images under {cfg.INPUT_PATH} ({cfg.IMAGE_EXTS})')
        sys.exit(1)
    units = group_units(images, cfg.INPUT_PATH, cfg.PROCESS_UNIT)
    LOG.info(f'{len(images)} raster(s) in {len(units)} unit(s).')

    rows = []
    total_tiles = 0
    problems = 0
    crs_seen = set()
    for name, paths in tqdm(units, desc='scanning', unit='unit'):
        row = {'unit': name, 'n_images': len(paths)}
        try:
            grid = MosaicGrid(paths)
            center_lat = None
            if grid.crs is not None and grid.crs.is_geographic:
                center_lat = (grid.bounds[1] + grid.bounds[3]) / 2.0
            gx, gy = metre_gsd(grid.transform, grid.crs, center_lat)
            crs_seen.add(str(grid.crs))
            gsd = max(gx, gy)
            ov = int(np.ceil(cfg.MAX_CROWN_M / gsd * cfg.OVERLAP_SAFETY)) \
                if gsd > 0 else cfg.OVERLAP
            stride = max(cfg.TILE_SIZE - ov, cfg.TILE_SIZE // 4)
            nx = int(np.ceil(grid.W / stride))
            ny = int(np.ceil(grid.H / stride))
            est = nx * ny

            # Internal overlap check: member footprints should only touch.
            boxes = [shp_box(s['coff'], s['roff'],
                             s['coff'] + s['w'], s['roff'] + s['h'])
                     for s in grid.sources]
            from shapely.strtree import STRtree
            tree = STRtree(boxes)
            n_ovl = 0
            for i, b in enumerate(boxes):
                for j in tree.query(b):
                    j = int(j)
                    if j <= i:
                        continue
                    if b.intersection(boxes[j]).area > 0:
                        n_ovl += 1
            row.update(status='OK', crs=str(grid.crs),
                       gsd_cm=round(gsd * 100, 2),
                       mosaic_px=f'{grid.W}x{grid.H}',
                       est_tiles=est, overlapping_pairs=n_ovl)
            total_tiles += est
            if n_ovl:
                LOG.warning(f'  {name}: {n_ovl} member pair(s) OVERLAP '
                            f'(expected edge-to-edge). Overlapped area is '
                            f'read once via the mosaic, so counts stay '
                            f'correct, but verify the download grid.')
        except Exception as exc:
            problems += 1
            row.update(status=f'PROBLEM: {exc}', crs='', gsd_cm='',
                       mosaic_px='', est_tiles='', overlapping_pairs='')
            LOG.error(f'  {name}: {exc}')
        rows.append(row)

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / 'scan_manifest.csv'
    with open(mpath, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hrs = total_tiles / max(cfg.SCAN_RATE_TILES_PER_S, 0.1) / 3600.0
    LOG.info('=' * 64)
    LOG.info(f'SCAN SUMMARY: {len(units)} units, {len(images)} rasters, '
             f'{problems} problem unit(s)')
    LOG.info(f'  CRS seen        : {sorted(crs_seen)}')
    LOG.info(f'  est. tiles      : {total_tiles:,}')
    LOG.info(f'  est. GPU time   : {hrs:.1f} h at '
             f'{cfg.SCAN_RATE_TILES_PER_S} tiles/s '
             f'({hrs/2:.1f} h per workstation with --shard over 2)')
    LOG.info(f'  manifest        : {mpath}')
    if problems:
        LOG.warning('Fix (or accept per-image fallback for) the PROBLEM '
                    'units before infer; they will otherwise fail cleanly '
                    'and be listed in the run summary.')


# =============================================================================
#                              COMMAND: infer
# =============================================================================
def cmd_infer(cfg, args):
    torch.backends.cudnn.benchmark = True     # fixed 1024 px input size
    _AmpState.enabled = bool(getattr(cfg, 'USE_AMP', False)
                             and torch.cuda.is_available())
    if _AmpState.enabled:
        LOG.info('AMP (float16 autocast) enabled; auto-falls back to FP32 '
                 'if a batch fails.')

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector
    # Scope is set in the MAIN process only; workers just read tiles.
    init_default_scope('mmdet')

    images = discover_images(cfg.INPUT_PATH,
                             tuple(e.lower() for e in cfg.IMAGE_EXTS))
    if not images:
        LOG.error(f'No images under {cfg.INPUT_PATH} ({cfg.IMAGE_EXTS})')
        sys.exit(1)
    units = group_units(images, cfg.INPUT_PATH, cfg.PROCESS_UNIT)
    if args.units_file:
        # An EXPLICIT unit list, matched exactly, not by substring.
        #
        # Folder names alone are not a safe specification of what to process.
        # This project's input tree carries the same 20x20 km grid cell under
        # several prefixes from successive download passes -- 01_Common_
        # Candidates__UAE_374, 03_Recovered__UAE_374 and Done_only_WS1__UAE_374
        # are one cell, not three -- plus quarantine folders and the desert
        # tiles the hard negatives were built from. Cross-unit dedup only
        # reaches BORDER_BAND_M from a unit edge, so a wholly duplicated cell
        # is counted twice in the national total and nothing downstream
        # notices.
        #
        # Naming the units removes that class of error entirely, and lets a
        # re-run reproduce a previous run's coverage exactly by taking the
        # list from its stats sidecars.
        wanted = [ln.strip() for ln in Path(args.units_file).read_text()
                  .splitlines() if ln.strip() and not ln.startswith('#')]
        have = {n for n, _ in units}
        missing = [w for w in wanted if w not in have]
        if missing:
            LOG.error(f'{len(missing)} unit(s) in {args.units_file} are not '
                      f'present under INPUT_PATH, e.g. {missing[:5]}')
            sys.exit(1)
        keep = set(wanted)
        units = [(n, p) for n, p in units if n in keep]
        LOG.info(f'--units-file: {len(units)} of {len(have)} unit(s) selected')
    if args.only:
        units = [(n, p) for n, p in units if args.only in n]
        if not units:
            LOG.error(f'--only {args.only!r} matched no unit')
            sys.exit(1)
    units = apply_shard(units, args.shard)
    LOG.info(f'{len(units)} unit(s) to process'
             + (f' (shard {args.shard})' if args.shard else ''))

    if args.dry_run:
        for n, p in units:
            LOG.info(f'  would process: {n} ({len(p)} rasters)')
        return

    ckpt = resolve_checkpoint(cfg)

    # Model is loaded lazily: a fully-resumed run never touches the GPU.
    model = [None]

    def get_model():
        if model[0] is None:
            LOG.info(f'Loading model: {ckpt}')
            model[0] = init_detector(Config.fromfile(cfg.CONFIG_FILE),
                                     ckpt, device=cfg.DEVICE)
        return model[0]

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Provenance snapshot: the exact effective settings of this run.
    snap = {'config': effective_config(cfg), 'config_hash': config_hash(cfg),
            'checkpoint_resolved': ckpt, 'shard': args.shard,
            'started_utc': utc_now()}
    ts0 = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(out_dir / f'run_config_{ts0}.json', 'w') as fh:
        json.dump(snap, fh, indent=1, default=str)

    summary_rows = []
    done_s = 0.0
    done_n = 0
    for idx, (name, paths) in enumerate(units, 1):
        eta = ''
        if done_n:
            rem = (len(units) - idx + 1) * (done_s / done_n)
            eta = f' | ETA {rem/3600:.1f} h'
        LOG.info(f'[{idx}/{len(units)}] {name} ({len(paths)} rasters){eta}')

        # Resume needs BOTH sidecar and gpkg (sidecar alone could outlive a
        # gpkg the user deleted by hand).
        if cfg.RESUME and unit_stats_path(out_dir, name).exists() \
                and unit_gpkg_path(out_dir, name).exists():
            try:
                with open(unit_stats_path(out_dir, name)) as fh:
                    st = json.load(fh)
                LOG.info(f"    resume: complete "
                         f"({st.get('final_count', '?')} palms) -> skip.")
                summary_rows.append({'unit': name, 'status': 'RESUMED',
                                     'final_count': st.get('final_count', '')})
                continue
            except Exception as exc:
                LOG.warning(f'    resume: unreadable stats ({exc}); redoing.')

        try:
            gdf, stats = process_unit(get_model(), name, paths, cfg)
            write_unit_outputs(out_dir, name, gdf, stats)
            done_s += stats['seconds']
            done_n += 1
            dp = stats['diam_pct']
            dp_str = (f"p1={dp.get('p1')} p50={dp.get('p50')} "
                      f"p99={dp.get('p99')}") if dp else 'n/a'
            LOG.info(f"    raw={stats['raw_detections']} "
                     f"filtered={stats['after_filters']} "
                     f"deduped={stats['after_dedup']} "
                     f"FINAL={stats['final_count']}  "
                     f"({stats['seconds']}s)  diam(m): {dp_str}")
            # Say where the losses went, not just how many there were.
            dr = stats.get('dropped') or {}
            if any(dr.values()) or stats.get('dropped_nms'):
                LOG.info('    dropped: ' + '  '.join(
                    f'{k}={v}' for k, v in
                    list(dr.items()) + [('nms', stats.get('dropped_nms', 0))]
                    if v))
            summary_rows.append({'unit': name, 'status': 'OK',
                                 'final_count': stats['final_count']})
        except Exception as exc:  # one bad unit must not abort the batch
            LOG.error(f'    FAILED: {name}: {exc}')
            LOG.debug(traceback.format_exc())
            summary_rows.append({'unit': name, 'status': f'FAILED: {exc}',
                                 'final_count': ''})

    # Per-run summary (merge writes the authoritative country totals).
    tag = (args.shard or '1/1').replace('/', 'of')
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    spath = out_dir / f'infer_summary_shard{tag}_{ts}.csv'
    with open(spath, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['unit', 'status', 'final_count'])
        w.writeheader()
        w.writerows(summary_rows)
    n_ok = sum(1 for r in summary_rows if r['status'] in ('OK', 'RESUMED'))
    n_fail = len(summary_rows) - n_ok
    LOG.info(f'infer done: {n_ok} ok, {n_fail} failed -> {spath}')
    if n_fail:
        LOG.warning('Re-run the same command to retry failed units '
                    '(finished units are skipped via RESUME).')


# =============================================================================
#                            COMMAND: calibrate
# =============================================================================
def resolve_checkpoint(cfg) -> str:
    """Resolve CONFIG.CHECKPOINT_FILE (may be a glob) to exactly one file."""
    import glob as _glob
    ckpt = cfg.CHECKPOINT_FILE
    if any(ch in ckpt for ch in '*?['):
        hits = sorted(_glob.glob(ckpt))
        if len(hits) != 1:
            LOG.error(f'CHECKPOINT_FILE glob matched {len(hits)} files: '
                      f'{hits[:5]} — must match exactly one.')
            sys.exit(1)
        ckpt = hits[0]
    if not Path(ckpt).exists():
        LOG.error(f'Checkpoint not found: {ckpt}')
        sys.exit(1)
    return ckpt


def _greedy_match(dts, gts, iou_thr):
    """COCO-style greedy matching for one image. `dts` must already be in
    DESCENDING score order. Returns a bool list aligned with dts: True if
    the detection matched a (previously unmatched) GT at IoU >= iou_thr."""
    from pycocotools import mask as maskUtils
    if not dts:
        return []
    if not gts:
        return [False] * len(dts)
    ious = maskUtils.iou(dts, gts, [0] * len(gts))   # (ndt, ngt)
    gt_taken = np.zeros(len(gts), dtype=bool)
    out = []
    for di in range(len(dts)):
        cand = np.where(~gt_taken)[0]
        matched = False
        if len(cand):
            best = cand[np.argmax(ious[di, cand])]
            if ious[di, best] >= iou_thr:
                gt_taken[best] = True
                matched = True
        out.append(matched)
    return out


def _val_predictions(cfg, args, coco):
    """Predictions for the calibration sweep, from (in priority order):
      1. --pkl PATH          an existing DumpDetResults pkl;
      2. the cache pkl       written by a previous calibrate run;
      3. fresh GPU inference with CONFIG's model over the GT json's images
         (--img-root), cached afterwards so re-sweeps are free.
    Returns a list of DumpDetResults-style records."""
    import pickle
    from pycocotools import mask as maskUtils

    if args.pkl:
        LOG.info(f'predictions: {args.pkl}')
        with open(args.pkl, 'rb') as f:
            return pickle.load(f)

    cache = Path(args.cache) if args.cache else \
        Path(cfg.OUTPUT_DIR) / 'calibration_preds.pkl'
    if cache.exists() and not args.recalc:
        LOG.info(f'predictions: cached {cache} (pass --recalc to redo)')
        with open(cache, 'rb') as f:
            return pickle.load(f)

    if not args.img_root:
        LOG.error('No --pkl given and no cache found: pass --img-root (the '
                  'directory the GT json\'s file_name entries are relative '
                  'to) so calibrate can run inference itself.')
        sys.exit(1)

    # ---- Fresh inference over the validation images -----------------------
    torch.backends.cudnn.benchmark = True
    _AmpState.enabled = bool(getattr(cfg, 'USE_AMP', False)
                             and torch.cuda.is_available())
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector
    init_default_scope('mmdet')
    ckpt = resolve_checkpoint(cfg)
    LOG.info(f'Loading model for calibration: {ckpt}')
    model = init_detector(Config.fromfile(cfg.CONFIG_FILE), ckpt,
                          device=cfg.DEVICE)

    img_root = Path(args.img_root)
    imgs = sorted(coco.imgs.values(), key=lambda im: im['id'])
    floor = 0.01           # keep everything scoreable; bounds pkl size
    records = []
    n_missing = 0
    batch_paths, batch_meta = [], []

    def flush():
        if not batch_paths:
            return
        results = safe_inference(model, batch_paths,
                                 oom_fallback=getattr(cfg, 'GPU_OOM_FALLBACK',
                                                      True))
        for res, im in zip(results, batch_meta):
            if res is None:
                continue
            pred = res.pred_instances
            keep = pred.scores > floor
            scores = pred.scores[keep].cpu().numpy()
            bboxes = pred.bboxes[keep].cpu().numpy()
            rles = None
            if hasattr(pred, 'masks') and pred.masks is not None:
                masks = pred.masks[keep].cpu().numpy().astype(np.uint8)
                rles = [maskUtils.encode(np.asfortranarray(m))
                        for m in masks]
            records.append({
                'img_id': int(im['id']),
                'ori_shape': (int(im['height']), int(im['width'])),
                'pred_instances': {'scores': scores, 'bboxes': bboxes,
                                   'masks': rles},
            })
        batch_paths.clear()
        batch_meta.clear()

    for im in tqdm(imgs, desc='calibrate: inference', unit='img'):
        p = img_root / im['file_name']
        if not p.exists():
            n_missing += 1
            continue
        batch_paths.append(str(p))
        batch_meta.append(im)
        if len(batch_paths) >= cfg.BATCH_SIZE:
            flush()
    flush()

    if n_missing:
        LOG.warning(f'{n_missing}/{len(imgs)} GT image file(s) not found '
                    f'under {img_root} — check --img-root.')
    if not records:
        LOG.error('No validation images could be processed.')
        sys.exit(1)

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix('.pkl.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(records, f)
    os.replace(tmp, cache)
    LOG.info(f'predictions cached -> {cache}')
    return records


def cmd_calibrate(cfg, args):
    """Derive the F1-optimal SCORE_THR for the deployed checkpoint against
    COCO validation GT. Works from an existing predictions pkl, or runs
    inference itself when only the checkpoint is available (see
    _val_predictions).

    Matching: per image, predictions in descending score order are greedily
    matched to the unmatched GT with the highest IoU >= --iou (COCO-style,
    the same convention as extract_instance_errors.py). Because greedy
    matching processes predictions in score order, the match set at any
    threshold t is exactly the score>=t prefix of the full match sequence —
    so one matching pass yields the exact P/R/F1 curve over all thresholds.
    """
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils

    coco = COCO(args.gt)
    results = _val_predictions(cfg, args, coco)
    use_segm = args.metric == 'segm'
    iou_thr = float(args.iou)

    total_gt = 0
    scores_all, matched_all = [], []
    n_unknown = 0
    for r in results:
        img_id = int(r['img_id'])
        if img_id not in coco.imgs:
            n_unknown += 1
            continue
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))
        total_gt += len(anns)

        pi = r['pred_instances']
        scores = np.asarray(pi['scores'], dtype=np.float64)
        if len(scores) == 0:
            continue
        order = np.argsort(-scores)

        if use_segm:
            pm = pi.get('masks')
            if pm is None:
                LOG.error('predictions have no masks; rerun with '
                          '--metric bbox or use a segmentation model.')
                sys.exit(1)

            def enc(m):
                if isinstance(m, dict):          # already RLE
                    return m
                return maskUtils.encode(
                    np.asfortranarray(np.asarray(m, dtype=np.uint8)))
            dts = [enc(pm[i]) for i in order]
            gts = [coco.annToRLE(a) for a in anns]
        else:
            bx = np.asarray(pi['bboxes'], dtype=np.float64)[order]
            dts = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in bx]
            gts = [a['bbox'] for a in anns]

        matched = _greedy_match(dts, gts, iou_thr)
        scores_all.extend(scores[order])
        matched_all.extend(matched)

    if n_unknown:
        LOG.warning(f'{n_unknown} prediction image(s) not in the GT json — '
                    f'check that the predictions match this annotation file.')
    if not scores_all or total_gt == 0:
        LOG.error('Nothing to calibrate (no predictions or no GT).')
        sys.exit(1)

    sc = np.asarray(scores_all)
    mt = np.asarray(matched_all)
    order = np.argsort(-sc)
    sc, mt = sc[order], mt[order]
    tp = np.cumsum(mt)                     # at threshold = sc[i] (inclusive)
    fp = np.cumsum(~mt)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / total_gt
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)

    best = int(np.argmax(f1))
    LOG.info('=' * 64)
    LOG.info(f'CALIBRATION ({args.metric}, IoU>={iou_thr}, '
             f'{total_gt:,} GT instances, {len(sc):,} predictions)')
    LOG.info(f'{"thr":>6} {"prec":>7} {"rec":>7} {"F1":>7}')
    for t in np.arange(0.05, 0.96, 0.05):
        i = int(np.searchsorted(-sc, -t, side='right')) - 1
        if i < 0:
            continue
        LOG.info(f'{t:6.2f} {prec[i]:7.3f} {rec[i]:7.3f} {f1[i]:7.3f}')
    LOG.info('-' * 64)
    LOG.info(f'F1-OPTIMAL THRESHOLD: {sc[best]:.3f}  '
             f'(P={prec[best]:.3f} R={rec[best]:.3f} F1={f1[best]:.3f})')
    LOG.info(f'-> set CONFIG.SCORE_THR = {round(float(sc[best]), 3)}')


# =============================================================================
#                              COMMAND: repair
# =============================================================================
def cmd_repair(cfg, args):
    """Rewrite existing per-unit gpkgs as valid, homogeneous MULTIPOLYGON.

    WHY THIS IS A SEPARATE COMMAND
      Units written before normalize_geometry() existed carry a mixed
      Polygon/MultiPolygon column, so GDAL declared their layer type as the
      generic 'GEOMETRY'. QGIS reads that; ArcGIS Pro does not. Re-running
      inference to fix a metadata declaration would be absurd -- the
      geometries themselves are correct -- so this pass repairs the files in
      place, reading and rewriting only. It touches NO stats sidecar and
      re-runs NO model, so a partially complete inference run is unaffected
      and RESUME still sees exactly the units it saw before.

      Safe to run WHILE inference is still going: units are processed one at
      a time and skipped if their sidecar is missing (i.e. still being
      written), and each rewrite goes to a temp file that is renamed into
      place only on success.
    """
    in_dirs = [Path(d) for d in (args.inputs or [cfg.OUTPUT_DIR])]
    targets = []
    for d in in_dirs:
        for gp in sorted(d.glob('*_palms.gpkg')):
            # The sidecar certifies the gpkg is complete (write_unit_outputs
            # writes it last). No sidecar => still in flight; leave it alone.
            if unit_stats_path(d, gp.name[:-len('_palms.gpkg')]).exists():
                targets.append(gp)
    if not targets:
        LOG.error(f'No completed *_palms.gpkg under {in_dirs}.')
        return

    LOG.info(f'repair: {len(targets)} unit gpkg(s) under {in_dirs}')
    n_ok = n_skip = n_fail = 0
    for gp in tqdm(targets, desc='repair', unit='unit'):
        try:
            g = gpd.read_file(gp)
        except Exception as exc:
            LOG.warning(f'  {gp.name}: unreadable ({exc}); left untouched')
            n_fail += 1
            continue
        layer = gp.name[:-len('_palms.gpkg')]
        if g.empty:
            # A zero-crown unit can still carry the SAME layer-name/type bug
            # as a populated one -- pre-fix code wrote these with
            # to_file(..., driver='GPKG') and no layer= or geometry_type=, so
            # the layer is named after the temp path and declared generic
            # 'GEOMETRY'. An empty feature COUNT is not the same thing as an
            # empty FIX; skipping unconditionally here (the original bug)
            # left exactly these files broken forever, --force included.
            try:
                import pyogrio
                cur = [str(r[0]) for r in pyogrio.list_layers(gp)]
            except Exception:
                cur = []
            if cur == [layer] and not args.force:
                n_skip += 1
                continue
            # write_unit_outputs's OWN attempt at a typed empty write
            # (geometry_type='MULTIPOLYGON') is wrapped in a try/except that
            # falls back to "no gpkg at all" for exactly this reason: some
            # pyogrio/GDAL builds reject a declared type on zero rows. Retrying
            # the same call here would just fail the same way, so match that
            # established fallback instead -- delete the broken, empty,
            # unopenable file. merge() already treats a missing gpkg on a
            # zero-count sidecar as authoritative (contributes 0, no error),
            # so this makes these units behave identically to the ones
            # write_unit_outputs produced with no gpkg in the first place.
            try:
                gp.unlink()
                n_ok += 1
                LOG.info(f'  {gp.name}: empty + unrepairable layer type on '
                        f'this GDAL build; deleted (0 crowns; sidecar '
                        f'final_count is authoritative for merge)')
            except OSError as exc:
                LOG.warning(f'  {gp.name}: could not delete ({exc}); '
                            f'left untouched')
                n_fail += 1
            continue
        # Layer NAME, independent of geometry. Units written before this fix
        # carry the temp filename as their layer name -- e.g.
        # '<unit>_palms.gpkg.tmp' -- because to_file() derived it from the
        # temp path and os.replace renamed only the file. ArcGIS Pro refuses
        # any feature class whose name contains a period, which is the actual
        # reason these files would not open there.
        try:
            import pyogrio
            cur = [str(r[0]) for r in pyogrio.list_layers(gp)]
        except Exception:
            cur = []
        name_ok = (cur == [layer])
        types = set(g.geometry.geom_type.dropna().unique())
        geom_ok = (types == {'MultiPolygon'}
                   and bool(g.geometry.is_valid.all()))
        if name_ok and geom_ok and not args.force:
            n_skip += 1               # clean name, homogeneous, valid
            continue
        before = len(g)
        g = normalize_geometry(g)
        tmp = gp.with_name(gp.name + '.repair.gpkg')
        if tmp.exists():
            tmp.unlink()
        try:
            g.to_file(tmp, driver='GPKG', layer=layer)
            os.replace(tmp, gp)        # atomic: original survives a crash
            n_ok += 1
            if len(g) != before:
                LOG.warning(f'  {gp.name}: {before} -> {len(g)} features '
                            f'({before - len(g)} unrepairable); the stats '
                            f'sidecar still reports the ORIGINAL count')
        except Exception as exc:
            LOG.warning(f'  {gp.name}: rewrite failed ({exc}); original kept')
            n_fail += 1
            if tmp.exists():
                tmp.unlink()

    LOG.info(f'repair done: {n_ok} rewritten, {n_skip} already clean, '
             f'{n_fail} failed')
    if n_fail:
        LOG.warning('Failed units keep their original file and are still '
                    'readable by QGIS/GeoPandas; merge is unaffected.')


# =============================================================================
#                              COMMAND: merge
# =============================================================================
def cmd_merge(cfg, args):
    from pyproj import CRS as PJCRS

    in_dirs = [Path(d) for d in (args.inputs or [cfg.OUTPUT_DIR])]
    stats_files = []
    for d in in_dirs:
        stats_files.extend(sorted(d.glob('*_stats.json')))
    if not stats_files:
        LOG.error(f'No *_stats.json under {in_dirs} — run infer first.')
        sys.exit(1)

    entries = []   # {unit, gpkg, stats}
    seen = set()
    for sp in stats_files:
        with open(sp) as fh:
            st = json.load(fh)
        if 'unit' not in st:      # not a unit sidecar (e.g. a stray file)
            LOG.debug(f'  skipping non-unit stats file: {sp.name}')
            continue
        name = st['unit']
        if name in seen:
            LOG.warning(f'  duplicate unit {name} (multiple input dirs); '
                        f'keeping the first, ignoring {sp}')
            continue
        seen.add(name)
        gp = sp.parent / f'{name}_palms.gpkg'
        if not gp.exists():
            if int(st.get('final_count', -1)) == 0:
                gp = None            # legitimately empty unit (count 0)
            else:
                LOG.warning(f'  {name}: stats present but gpkg missing; '
                            f'skipped — re-run infer for this unit.')
                continue
        entries.append({'unit': name, 'gpkg': gp, 'stats': st})
    LOG.info(f'{len(entries)} unit(s) to merge from {len(in_dirs)} dir(s).')

    # Provenance check: a country total is only meaningful if every unit was
    # produced by the same checkpoint / threshold / settings. Mixtures are
    # reported loudly (merge still proceeds — the master records per-unit
    # provenance columns is overkill, but the warning must not be missable).
    for key in ('checkpoint', 'score_thr', 'config_hash'):
        vals = {str(e['stats'].get(key, '?')) for e in entries}
        if len(vals) > 1:
            LOG.warning('=' * 64)
            LOG.warning(f'PROVENANCE MISMATCH: units were produced with '
                        f'different {key!r}: {sorted(vals)[:4]}')
            LOG.warning('The merged total mixes operating points. Re-run '
                        'infer for the outdated units unless intentional.')
            LOG.warning('=' * 64)

    target_crs = PJCRS.from_user_input(cfg.TARGET_CRS)

    # ---- Pass A: border-band candidates for cross-unit dedup --------------
    # Only detections within BORDER_BAND_M of their unit's boundary can be
    # duplicates of a neighbouring unit; interior detections are auto-kept.
    band = float(cfg.BORDER_BAND_M)
    cand_xy, cand_score, cand_unit, cand_id = [], [], [], []
    for ui, e in enumerate(tqdm(entries, desc='pass A (borders)',
                                unit='unit')):
        if e['gpkg'] is None:
            continue
        g = gpd.read_file(e['gpkg'])
        if g.empty:
            continue
        if 'id' not in g:
            g['id'] = range(len(g))
        gp = g.to_crs(target_crs) if g.crs != target_crs else g
        cen = gp.geometry.centroid
        # Unit boundary in the target CRS; distance-to-boundary defines the
        # candidate band (exact even when reprojection rotates the box).
        bx = shp_box(*e['stats']['bounds'])
        bseries = gpd.GeoSeries([bx], crs=e['stats']['crs']).to_crs(target_crs)
        boundary = bseries.iloc[0].exterior
        d = cen.distance(boundary)
        m = (d <= band).values
        if not m.any():
            continue
        cand_xy.append(np.column_stack([cen.x.values[m], cen.y.values[m]]))
        cand_score.append(gp['score'].values[m] if 'score' in gp
                          else np.ones(int(m.sum())))
        cand_unit.append(np.full(int(m.sum()), ui))
        cand_id.append(gp['id'].values[m].astype(np.int64))

    suppressed = set()   # (unit_idx, local_id)
    if cfg.CROSS_UNIT_DEDUP and cand_xy:
        xy = np.concatenate(cand_xy)
        sc = np.concatenate(cand_score)
        uu = np.concatenate(cand_unit)
        ii = np.concatenate(cand_id)
        keep = cross_unit_centroid_nms(xy, sc, uu,
                                       cfg.CROSS_UNIT_DEDUP_DIST_M)
        for k, u, lid in zip(keep, uu, ii):
            if not k:
                suppressed.add((int(u), int(lid)))
        LOG.info(f'  border candidates: {len(sc):,}; cross-unit duplicates '
                 f'removed: {len(suppressed):,} '
                 f'(radius {cfg.CROSS_UNIT_DEDUP_DIST_M} m)')

    # ---- Pass B: stream-write the master, collect statistics --------------
    out_dir = in_dirs[0]
    fmt = getattr(args, 'format', 'gpkg') or 'gpkg'
    if fmt == 'parquet':
        # One Parquet file per unit under a single directory, which is what
        # every Parquet reader (DuckDB, GeoPandas, Pro 3.3+) treats as one
        # dataset. Streaming per unit also avoids holding 50M geometries in
        # RAM, which a single-file write would require.
        merged_path = out_dir / cfg.MERGED_NAME
        if merged_path.exists():
            shutil.rmtree(merged_path)
        merged_path.mkdir(parents=True)
    else:
        merged_path = out_dir / f'{cfg.MERGED_NAME}.gpkg'
        if merged_path.exists():
            merged_path.unlink()

    global_id = 0
    total = 0
    per_unit_rows = []
    diam_chunks = []
    # Fixed for the whole merge: the GPKG layer schema is set by the first
    # unit written, so it must not depend on which unit that happens to be.
    out_columns = master_columns(cfg)
    first_write = True
    for ui, e in enumerate(tqdm(entries, desc='pass B (write)', unit='unit')):
        if e['gpkg'] is None:
            per_unit_rows.append({'unit': e['unit'], 'count': 0})
            continue
        g = gpd.read_file(e['gpkg'])
        if g.empty:
            per_unit_rows.append({'unit': e['unit'], 'count': 0})
            continue
        if 'id' not in g:
            g['id'] = range(len(g))
        drop = g['id'].apply(lambda v: (ui, int(v)) in suppressed)
        g = g[~drop].copy()
        if g.crs != target_crs:
            g = g.to_crs(target_crs)
        # Repair and homogenise BEFORE the id/count accounting, so a dropped
        # unrepairable ring cannot desync count_summary.csv from the master,
        # and before any metric math, so it cannot contribute a nonsense area.
        g = normalize_geometry(g)
        if g.empty:
            per_unit_rows.append({'unit': e['unit'], 'count': 0})
            continue
        g['id'] = range(global_id, global_id + len(g))
        global_id += len(g)
        total += len(g)
        per_unit_rows.append({'unit': e['unit'], 'count': len(g)})
        if 'diam_m' in g:
            diam_chunks.append(g['diam_m'].values.astype(np.float32))
        if len(g):
            # Analysis-ready columns: metric crown area and WGS84 centroid.
            cen = g.geometry.centroid
            # Area in an equal-area CRS, not in TARGET_CRS — see the
            # EQUAL_AREA_CRS comment; measuring in UTM 40N inflates western
            # Abu Dhabi crowns by ~0.7%.
            g['area_m2'] = g.geometry.to_crs(EQUAL_AREA_CRS).area.round(2)
            ll = gpd.GeoSeries(cen, crs=target_crs).to_crs('EPSG:4326')
            g['lon'] = ll.x.round(7)
            g['lat'] = ll.y.round(7)
            # Neighbour density (palms within NEIGHBOR_RADIUS_M, self
            # excluded): computed per unit, which is exact except within one
            # radius of a unit border — negligible for stratification.
            r_nbr = float(getattr(cfg, 'NEIGHBOR_RADIUS_M', 0) or 0)
            if r_nbr > 0:
                try:
                    from scipy.spatial import cKDTree
                    xy = np.column_stack([cen.x.values, cen.y.values])
                    tree = cKDTree(xy)
                    counts = tree.query_ball_point(xy, r=r_nbr,
                                                   return_length=True)
                    g[f'nbr_{int(r_nbr)}m'] = np.asarray(counts) - 1
                except ImportError:
                    pass          # SciPy absent: skip the optional column
            g = conform_schema(g, out_columns)
            if fmt == 'parquet':
                # Safe filename: unit names come from directory names and can
                # carry characters Parquet readers dislike in a path.
                safe = ''.join(c if (c.isalnum() or c in '-_') else '_'
                               for c in e['unit'])
                g.to_parquet(merged_path / f'{safe}.parquet',
                             compression='zstd', index=False)
            else:
                g.to_file(merged_path, driver='GPKG', layer='palms',
                          mode='w' if first_write else 'a')
            first_write = False

    # ---- Statistics --------------------------------------------------------
    stats_out = {
        'total_palms': int(total),
        'units': len(entries),
        'cross_unit_duplicates_removed': len(suppressed),
        'target_crs': str(cfg.TARGET_CRS),
        'area_crs': EQUAL_AREA_CRS,
        'merged_format': fmt,
        'merged_gpkg': str(merged_path),
        'created_utc': utc_now(),
    }
    if diam_chunks:
        d = np.concatenate(diam_chunks)
        stats_out['diameter_m'] = {
            f'p{q}': round(float(np.percentile(d, q)), 2)
            for q in (1, 5, 25, 50, 75, 95, 99)}
        stats_out['diameter_mean_m'] = round(float(d.mean()), 2)

    csv_path = out_dir / 'count_summary.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['unit', 'count'])
        w.writeheader()
        w.writerows(sorted(per_unit_rows, key=lambda r: r['unit']))
        w.writerow({'unit': 'TOTAL', 'count': total})

    # Name must NOT end in `_stats.json` (that pattern is reserved for unit
    # sidecars, which merge globs for on the next run).
    jpath = out_dir / 'country_statistics.json'
    with open(jpath, 'w') as fh:
        json.dump(stats_out, fh, indent=1)

    LOG.info('=' * 64)
    LOG.info(f'MERGE DONE. Total palms: {total:,} '
             f'({len(suppressed):,} cross-unit duplicates removed)')
    LOG.info(f'  master  : {merged_path}')
    LOG.info(f'  summary : {csv_path}')
    LOG.info(f'  stats   : {jpath}')


# =============================================================================
#                                 MAIN
# =============================================================================
def apply_overrides(cfg, pairs):
    """--set KEY=VALUE overrides of CONFIG (typed via literal_eval, falling
    back to string), so per-machine differences (paths, DEVICE, shard output
    dirs) never require editing the file on each workstation."""
    import ast
    for pair in pairs or []:
        if '=' not in pair:
            raise SystemExit(f'--set expects KEY=VALUE, got: {pair}')
        key, val = pair.split('=', 1)
        key = key.strip()
        if not hasattr(cfg, key):
            raise SystemExit(f'--set: unknown CONFIG key {key!r}')
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            parsed = val
        setattr(cfg, key, parsed)
        LOG.info(f'CONFIG override: {key} = {parsed!r}')


def main():
    parser = argparse.ArgumentParser(
        description='Country-scale date-palm inference pipeline '
                    '(scan -> calibrate -> infer -> merge). Edit the CONFIG '
                    'block, or override per-run with --set KEY=VALUE.')
    sub = parser.add_subparsers(dest='cmd', required=True)

    def add_common(p):
        p.add_argument('--set', action='append', metavar='KEY=VALUE',
                       help='override a CONFIG attribute (repeatable), e.g. '
                            "--set SCORE_THR=0.45 --set DEVICE='cuda:1'")

    p_scan = sub.add_parser('scan', help='validate data, estimate cost '
                                         '(no GPU)')
    add_common(p_scan)

    p_cal = sub.add_parser('calibrate',
                           help='derive the F1-optimal SCORE_THR for the '
                                'deployed checkpoint vs COCO validation GT')
    p_cal.add_argument('--gt', required=True, help='COCO annotation json')
    p_cal.add_argument('--img-root', default=None,
                       help='directory the GT json file_name entries are '
                            'relative to; needed (once) when no --pkl is '
                            'given, so calibrate runs inference itself with '
                            "CONFIG's model")
    p_cal.add_argument('--pkl', default=None,
                       help='optional existing predictions pkl '
                            '(tools/test.py --out format); skips inference')
    p_cal.add_argument('--cache', default=None,
                       help='where to cache self-run predictions (default: '
                            'OUTPUT_DIR/calibration_preds.pkl)')
    p_cal.add_argument('--recalc', action='store_true',
                       help='ignore the prediction cache and re-run inference')
    p_cal.add_argument('--metric', choices=('segm', 'bbox'), default='segm')
    p_cal.add_argument('--iou', default=0.5, type=float)
    add_common(p_cal)

    p_inf = sub.add_parser('infer', help='run inference (resumable)')
    p_inf.add_argument('--shard', default=None,
                       help='K/N split of units across machines, e.g. 1/2')
    p_inf.add_argument('--units-file', default=None,
                       help='file of unit names, one per line (# comments '
                            'allowed), matched EXACTLY. Use this rather than '
                            '--only when the input tree holds the same grid '
                            'cell under several prefixes; generate it from a '
                            'previous run\'s *_stats.json names to reproduce '
                            'that run\'s coverage. Aborts if any listed unit '
                            'is absent, so a typo cannot silently shrink the '
                            'run.')
    p_inf.add_argument('--only', default=None,
                       help='process only units whose name CONTAINS this '
                            '(substring). Convenient but blunt: --only '
                            'UAE_32 also matches UAE_320..329. Prefer '
                            '--units-file for production runs.')
    p_inf.add_argument('--dry-run', action='store_true',
                       help='list units and exit')
    add_common(p_inf)

    p_rep = sub.add_parser('repair',
                           help='rewrite existing per-unit gpkgs as valid '
                                'MULTIPOLYGON (fixes ArcGIS Pro); no GPU, '
                                'no re-inference')
    p_rep.add_argument('--inputs', nargs='*', default=None,
                       help='output dirs to repair (default: '
                            'CONFIG.OUTPUT_DIR)')
    p_rep.add_argument('--force', action='store_true',
                       help='rewrite even units that are already homogeneous '
                            'and valid')
    add_common(p_rep)

    p_mrg = sub.add_parser('merge', help='merge per-unit outputs into the '
                                         'country master + statistics')
    p_mrg.add_argument('--inputs', nargs='*', default=None,
                       help='output dirs to merge (default: CONFIG.OUTPUT_DIR)'
                            '; pass several when shards ran on different '
                            'machines and were copied together')
    p_mrg.add_argument('--format', choices=('gpkg', 'parquet'), default='gpkg',
                       help="master output format. 'gpkg' writes one "
                            "GeoPackage (layer 'palms'); 'parquet' writes a "
                            'directory of one GeoParquet file per unit, which '
                            'is far faster at country scale and is read as a '
                            'single dataset by DuckDB and GeoPandas. Convert '
                            'either to File Geodatabase with ogr2ogr -f '
                            'OpenFileGDB for ArcGIS Pro.')
    add_common(p_mrg)

    args = parser.parse_args()
    cfg = CONFIG
    tune_gdal_env()
    global LOG
    LOG = setup_logging(cfg.LOG_LEVEL)          # console-only for overrides
    apply_overrides(cfg, getattr(args, 'set', None))
    # Re-init with the (possibly overridden) output dir for the log file.
    LOG = setup_logging(cfg.LOG_LEVEL, log_dir=Path(cfg.OUTPUT_DIR) / 'logs',
                        tag=args.cmd)

    if args.cmd == 'scan':
        cmd_scan(cfg, args)
    elif args.cmd == 'calibrate':
        cmd_calibrate(cfg, args)
    elif args.cmd == 'infer':
        cmd_infer(cfg, args)
    elif args.cmd == 'repair':
        cmd_repair(cfg, args)
    elif args.cmd == 'merge':
        cmd_merge(cfg, args)


if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
