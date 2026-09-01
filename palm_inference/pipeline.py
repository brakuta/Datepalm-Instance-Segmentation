"""Country-scale date palm inference pipeline — throughput-optimised.

Architecture (v2):
  1. Discover all GeoTIFFs under config.input_roots (recursive).
  2. Build/load a tile-level manifest.
  3. For each input GeoTIFF:
     - Producer thread: pulls batches from the DataLoader and runs GPU
       inference back-to-back. Predictions go onto a bounded queue.
     - Pool of vectoriser threads: drain predictions, emit polygons.
     - Writer: streams polygons to per-input GeoPackage in chunks.

Design rationale
----------------
The previous version serialised GPU inference and CPU vectorisation:

        GPU  ####          ####          ####
        CPU      ##########    ##########    ##########

Producer–consumer pipelining runs them in parallel so the GPU never waits:

        GPU  ####  ####  ####  ####  ####  ####
        CPU    ##########  ##########  ##########

On a 24 GB GPU with the FP16 deployed model at 1024² and batch 6, the
forward pass is ~180 ms. Per-tile vectorisation (mask smoothing + contours
+ polygon emission) is ~30–80 ms depending on palm density. With 4
vectoriser threads the CPU side finishes well within the GPU budget, so
GPU utilisation approaches 100%.

Additional optimisations:
  * Masks are cropped to their bounding boxes on the GPU before transfer
    to host memory — see fast_detector.py.
  * Polygon construction uses direct numpy ellipse points instead of
    shapely Point().buffer().scale().rotate() (~20x faster per polygon).
  * Shapely geometry validation and simplification are only invoked when
    numerically needed (most emitted ellipses are already valid).
"""
from __future__ import annotations

import ctypes
import gc
import logging
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.windows
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import InferenceConfig
from .fast_detector import FastBatchedDetector

logger = logging.getLogger("palm_inference.pipeline")


# =============================================================================
# 1. INPUT DISCOVERY + MANIFEST
# =============================================================================

def discover_inputs(roots: Iterable[Path], min_size: int = 100_000) -> List[Path]:
    """Recursively find all .tif/.tiff files under the given roots.
    Skips files smaller than `min_size` bytes.
    """
    roots = list(roots)
    found = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            logger.warning(f"Input root does not exist: {root}")
            continue
        for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            for p in root.rglob(ext):
                try:
                    if p.stat().st_size >= min_size:
                        found.append(p)
                except OSError:
                    continue
    found = sorted(set(found))
    logger.info(f"Discovered {len(found)} GeoTIFFs across {len(roots)} roots")
    return found


def build_manifest(inputs: List[Path], cfg: InferenceConfig) -> pd.DataFrame:
    """Build a tile-level manifest: one row per (input, x, y) work item."""
    stride = cfg.tile_size - cfg.overlap
    rows = []
    for input_path in tqdm(inputs, desc="Building manifest"):
        try:
            with rasterio.open(input_path) as src:
                W, H = src.width, src.height
                rel = _resolve_rel_path(input_path, cfg.input_roots)
        except Exception as e:
            logger.error(f"Cannot open {input_path}: {e}")
            continue

        xs = list(range(0, max(1, W - cfg.tile_size + 1), stride))
        if xs and xs[-1] + cfg.tile_size < W:
            xs.append(W - cfg.tile_size)
        elif not xs:
            xs = [0]
        ys = list(range(0, max(1, H - cfg.tile_size + 1), stride))
        if ys and ys[-1] + cfg.tile_size < H:
            ys.append(H - cfg.tile_size)
        elif not ys:
            ys = [0]

        for y in ys:
            for x in xs:
                rows.append({
                    "input_path": str(input_path),
                    "rel_path": str(rel),
                    "image_w": W, "image_h": H,
                    "x": x, "y": y,
                    "done": False,
                })

    df = pd.DataFrame(rows)
    logger.info(f"Manifest: {len(df):,} tile jobs across {len(inputs)} inputs")
    return df


def _resolve_rel_path(path: Path, roots: List[Path]) -> Path:
    for root in roots:
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    return Path(path.name)


def load_or_build_manifest(inputs: List[Path],
                           cfg: InferenceConfig) -> pd.DataFrame:
    """Load existing manifest if present and consistent; else build fresh."""
    if cfg.manifest_path.exists():
        logger.info(f"Loading existing manifest: {cfg.manifest_path}")
        df = pd.read_parquet(cfg.manifest_path)

        stored_tile_size = df.attrs.get("tile_size", None)
        stored_overlap   = df.attrs.get("overlap",    None)
        if (stored_tile_size is not None and stored_tile_size != cfg.tile_size) or \
           (stored_overlap    is not None and stored_overlap    != cfg.overlap):
            logger.warning(
                f"Manifest tile_size/overlap ({stored_tile_size}/{stored_overlap}) "
                f"differs from current config ({cfg.tile_size}/{cfg.overlap}). "
                f"Rebuilding manifest."
            )
            cfg.manifest_path.unlink()
        else:
            existing_inputs = set(df["input_path"].unique())
            new_inputs = [p for p in inputs if str(p) not in existing_inputs]
            if new_inputs:
                logger.info(f"Manifest extension: {len(new_inputs)} new inputs found")
                extension = build_manifest(new_inputs, cfg)
                df = pd.concat([df, extension], ignore_index=True)
                df.attrs["tile_size"] = cfg.tile_size
                df.attrs["overlap"]   = cfg.overlap
                df.to_parquet(cfg.manifest_path)
            n_done = df["done"].sum()
            logger.info(f"Manifest: {len(df):,} tiles, {n_done:,} already done "
                        f"({100*n_done/max(len(df),1):.1f}%)")
            return df

    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_manifest(inputs, cfg)
    df.attrs["tile_size"] = cfg.tile_size
    df.attrs["overlap"]   = cfg.overlap
    df.to_parquet(cfg.manifest_path)
    logger.info(f"Manifest persisted to {cfg.manifest_path}")
    return df


# =============================================================================
# 2. DATASET
# =============================================================================

class TiledGeoTiffDataset(Dataset):
    """One instance per input GeoTIFF. Each __getitem__ reads one tile via
    windowed rasterio I/O. Workers each open their own handle.
    """
    def __init__(self, input_path: str,
                 tile_origins: List[Tuple[int, int]],
                 tile_size: int):
        self.input_path = input_path
        self.origins = tile_origins
        self.tile_size = tile_size
        self._ds = None

    def __len__(self):
        return len(self.origins)

    def _ensure_open(self):
        if self._ds is None:
            os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
            self._ds = rasterio.open(self.input_path)

    def __del__(self):
        if self._ds is not None:
            try:
                self._ds.close()
            except Exception:
                pass
            self._ds = None

    def __getitem__(self, idx):
        self._ensure_open()
        x, y = self.origins[idx]
        ts = self.tile_size
        window = Window(x, y, ts, ts)
        tile = self._ds.read(
            indexes=[1, 2, 3],
            window=window,
            boundless=True,
            fill_value=0,
        )
        nodata_frac = float((tile.sum(axis=0) == 0).mean())
        # CHW RGB -> HWC BGR
        tile = np.ascontiguousarray(tile.transpose(1, 2, 0)[..., ::-1])
        return tile, x, y, nodata_frac


def collate_tiles(batch):
    tiles = [b[0] for b in batch]
    xs = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ys = torch.tensor([b[2] for b in batch], dtype=torch.long)
    nodata = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return tiles, xs, ys, nodata


# =============================================================================
# 3. VECTORIZATION HELPERS
# =============================================================================

def _intra_tile_nms(scores: np.ndarray,
                    bboxes: np.ndarray,
                    iou_threshold: float = 0.5) -> np.ndarray:
    """Vectorised greedy bbox-IoU NMS. Returns keep mask.

    Secondary suppression after the model's NMS (trained at 0.7). Targets
    duplicate proposals on the same palm crown without merging genuinely
    adjacent palms (empirical inter-crown IoU 0.1–0.3 vs duplicate IoU 0.5+).
    """
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if iou_threshold >= 1.0 or n == 1:
        return np.ones(n, dtype=bool)

    x1 = bboxes[:, 0].astype(np.float32)
    y1 = bboxes[:, 1].astype(np.float32)
    x2 = bboxes[:, 2].astype(np.float32)
    y2 = bboxes[:, 3].astype(np.float32)
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = np.argsort(scores)[::-1].astype(np.int64)
    keep = np.ones(n, dtype=bool)
    idxs = order.copy()

    while idxs.size > 0:
        i = idxs[0]
        rest = idxs[1:]
        if rest.size == 0:
            break
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1);    h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        suppress = iou > iou_threshold
        if suppress.any():
            keep[rest[suppress]] = False
        idxs = rest[~suppress]
    return keep


def _mask_iou_dedup_cropped(masks: List[np.ndarray],
                            mb: np.ndarray,
                            scores: np.ndarray,
                            iou_threshold: float = 0.3) -> np.ndarray:
    """Mask-pixel-IoU dedup operating on bbox-cropped masks.

    Two masks can be compared via their intersection on the overlap of their
    bounding boxes. Areas are computed once. Only bbox-overlapping pairs are
    examined, which is a small fraction of total pairs for sparse detections.
    """
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if iou_threshold >= 1.0 or n == 1:
        return np.ones(n, dtype=bool)

    areas = np.array([int(m.sum()) for m in masks], dtype=np.int64)
    order = np.argsort(scores)[::-1]
    suppressed = np.zeros(n, dtype=bool)

    for i_idx, i in enumerate(order):
        if suppressed[i]:
            continue
        bi = mb[i]
        for j in order[i_idx + 1:]:
            if suppressed[j]:
                continue
            bj = mb[j]
            # Intersection bbox.
            ix1 = max(bi[0], bj[0]); iy1 = max(bi[1], bj[1])
            ix2 = min(bi[2], bj[2]); iy2 = min(bi[3], bj[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            # Crop both masks to the intersection rectangle.
            mi = masks[i][iy1 - bi[1]:iy2 - bi[1], ix1 - bi[0]:ix2 - bi[0]]
            mj = masks[j][iy1 - bj[1]:iy2 - bj[1], ix1 - bj[0]:ix2 - bj[0]]
            if mi.size == 0 or mj.size == 0:
                continue
            inter = int(np.logical_and(mi, mj).sum())
            if inter == 0:
                continue
            union = int(areas[i] + areas[j] - inter)
            if union > 0 and (inter / union) > iou_threshold:
                suppressed[j] = True
    return ~suppressed


# Pre-computed unit-circle vertices, used for fast ellipse polygon emission.
_ELLIPSE_N = 32
_ELLIPSE_T = np.linspace(0, 2 * np.pi, _ELLIPSE_N, endpoint=False)
_ELLIPSE_COS = np.cos(_ELLIPSE_T).astype(np.float64)
_ELLIPSE_SIN = np.sin(_ELLIPSE_T).astype(np.float64)


def _ellipse_polygon_coords(cx: float, cy: float, ra: float, rb: float,
                            angle_deg: float) -> np.ndarray:
    """Return (N, 2) ellipse vertex array in image-pixel coords.

    Direct numpy construction, ~20x faster than
        Point(cx, cy).buffer(ra).scale(yfact=rb/ra).rotate(-angle_deg).
    """
    a = np.deg2rad(-angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    ex = ra * _ELLIPSE_COS
    ey = rb * _ELLIPSE_SIN
    x = cx + ex * ca - ey * sa
    y = cy + ex * sa + ey * ca
    return np.column_stack([x, y])


# =============================================================================
# 4. TILE -> POLYGON VECTORIZATION
# =============================================================================

def vectorize_tile(
    pred: dict,
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    image_w: int,
    image_h: int,
    overlap: int,
    score_thr: float,
    min_area_px: int,
    max_components: int,
    simplify_tol_px: float,
    intra_tile_nms_iou: float = 0.5,
    mask_iou_dedup_thr: float = 0.3,
    min_circularity: float = 0.35,
    max_aspect_ratio: float = 4.0,
) -> List[dict]:
    """Fast tile vectorisation with shape-aware filtering.

    Expects pred with cropped masks (one per instance, at bbox resolution)
    as produced by FastBatchedDetector.predict_batch.
    """
    from shapely.geometry import Polygon

    scores_t = pred["scores"]
    if scores_t is None or scores_t.numel() == 0:
        return []

    # Score filtering already applied on GPU. Defensive re-check in case the
    # caller bypasses GPU-side filter.
    scores = scores_t.numpy()
    keep_score = scores > score_thr
    if not keep_score.any():
        return []

    scores = scores[keep_score]
    bboxes = pred["bboxes"].numpy()[keep_score]
    masks: List[np.ndarray] = [pred["masks"][i] for i in np.where(keep_score)[0]]
    mb: np.ndarray = pred["mask_bboxes"][keep_score]

    # --- Stage 1: bbox-IoU duplicate suppression ---
    if intra_tile_nms_iou < 1.0:
        keep_nms = _intra_tile_nms(scores, bboxes,
                                   iou_threshold=intra_tile_nms_iou)
        scores = scores[keep_nms]
        bboxes = bboxes[keep_nms]
        masks = [m for m, k in zip(masks, keep_nms) if k]
        mb = mb[keep_nms]

    if len(scores) == 0:
        return []

    # --- Stage 2: smooth masks (kernel is fixed size, work is linear in
    # the combined mask-crop area) ---
    smoothed: List[np.ndarray] = []
    for mask in masks:
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            smoothed.append(mask.astype(np.uint8))
            continue
        blurred = cv2.GaussianBlur(mask * 255, (5, 5), sigmaX=1.5)
        smoothed.append((blurred > 127).astype(np.uint8))

    # --- Stage 3: mask-IoU dedup on cropped masks ---
    if mask_iou_dedup_thr < 1.0 and len(scores) > 1:
        keep_mask_iou = _mask_iou_dedup_cropped(smoothed, mb, scores,
                                                iou_threshold=mask_iou_dedup_thr)
        scores = scores[keep_mask_iou]
        smoothed = [m for m, k in zip(smoothed, keep_mask_iou) if k]
        mb = mb[keep_mask_iou]

    if len(scores) == 0:
        return []

    # --- Valid zone (seam-free tile assignment) ---
    margin = overlap // 2
    vx_min = margin if tile_x > 0 else 0
    vy_min = margin if tile_y > 0 else 0
    interior_right  = (tile_x + tile_w) < image_w
    interior_bottom = (tile_y + tile_h) < image_h
    vx_max = (tile_w - margin) if interior_right  else tile_w
    vy_max = (tile_h - margin) if interior_bottom else tile_h

    out = []
    for mask_sm, score, box in zip(smoothed, scores, mb):
        bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Contour extraction on the cropped mask.
        contours, _ = cv2.findContours(
            mask_sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or len(contours) > max_components:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px:
            continue

        # Shape filters.
        perimeter = cv2.arcLength(cnt, closed=True)
        circularity = (4 * np.pi * area_px / (perimeter ** 2)) if perimeter > 0 else 0.0
        if circularity < min_circularity:
            continue

        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        aspect = max(w_, h_) / max(min(w_, h_), 1)
        if aspect > max_aspect_ratio:
            continue

        # Centroid in mask-local coords -> tile-local coords.
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_mask = M["m10"] / M["m00"]
        cy_mask = M["m01"] / M["m00"]
        cx_local = cx_mask + bx1  # + bx1 translates mask-local -> tile-local
        cy_local = cy_mask + by1

        # Valid-zone check (tile-local).
        cx_ok = (vx_min <= cx_local < vx_max) if interior_right  else (vx_min <= cx_local <= vx_max)
        cy_ok = (vy_min <= cy_local < vy_max) if interior_bottom else (vy_min <= cy_local <= vy_max)
        if not (cx_ok and cy_ok):
            continue

        # Ellipse in image-pixel coords.
        cx_img = cx_local + tile_x
        cy_img = cy_local + tile_y

        if len(cnt) >= 5:
            try:
                (_, _), (ea, eb), angle = cv2.fitEllipse(cnt)
                ra = max(ea, eb) / 2.0
                rb = min(ea, eb) / 2.0
                if ra / max(rb, 1e-3) > 2.5:
                    rb = ra / 2.5
                coords = _ellipse_polygon_coords(cx_img, cy_img, ra, rb, angle)
            except cv2.error:
                # Fallback: circle from mean distance of contour to centroid.
                dists = np.sqrt((cnt[:, 0, 0].astype(float) - cx_mask) ** 2 +
                                (cnt[:, 0, 1].astype(float) - cy_mask) ** 2)
                r = float(dists.mean())
                coords = _ellipse_polygon_coords(cx_img, cy_img, r, r, 0.0)
        else:
            (_, _), radius_enc = cv2.minEnclosingCircle(cnt)
            r = float(radius_enc)
            coords = _ellipse_polygon_coords(cx_img, cy_img, r, r, 0.0)

        try:
            poly = Polygon(coords)
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue

        # Light simplification — fitted ellipses are already near-minimal;
        # simplification only pays off for the odd degenerate case.
        if simplify_tol_px > 0:
            try:
                simplified = poly.simplify(simplify_tol_px, preserve_topology=True)
                if not simplified.is_empty and simplified.is_valid:
                    poly = simplified
            except Exception:
                pass

        out.append({
            "polygon_px": poly,
            "score":      float(score),
            "area_px":    float(area_px),
            "circularity": float(circularity),
        })
    return out


def georeference_polygons(records: List[dict],
                          transform: rasterio.Affine) -> List[dict]:
    """Apply Affine transform to translate image-pixel polygons -> CRS coords."""
    from shapely.affinity import affine_transform
    a, b, c, d, e, f = (transform.a, transform.b, transform.c,
                        transform.d, transform.e, transform.f)
    matrix = [a, b, d, e, c, f]
    out = []
    for r in records:
        geom = affine_transform(r["polygon_px"], matrix)
        if geom.is_empty or not geom.is_valid:
            continue
        out.append({
            "geometry":    geom,
            "score":       r["score"],
            "area_px":     r["area_px"],
            "circularity": r.get("circularity", None),
        })
    return out


# =============================================================================
# 5. STREAMING WRITER
# =============================================================================

class StreamingGeoWriter:
    def __init__(self, output_path: Path, crs, fmt: str = "gpkg"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.crs = crs
        self.fmt = fmt
        self.buffer: List[dict] = []
        self.total_written = 0
        self.next_id = 0
        if self.output_path.exists():
            self.output_path.unlink()

    def add(self, records: List[dict]):
        for r in records:
            r["id"] = self.next_id
            self.next_id += 1
            self.buffer.append(r)

    def flush(self):
        if not self.buffer:
            return
        gdf = gpd.GeoDataFrame(self.buffer, crs=self.crs, geometry="geometry")
        if self.fmt == "gpkg":
            mode = "a" if self.output_path.exists() else "w"
            gdf.to_file(self.output_path, driver="GPKG", mode=mode,
                        engine="pyogrio")
        elif self.fmt == "shp":
            mode = "a" if self.output_path.exists() else "w"
            gdf.to_file(self.output_path, mode=mode, engine="pyogrio")
        elif self.fmt == "parquet":
            chunk_path = self.output_path.with_suffix(
                f".part{self.total_written:08d}.parquet"
            )
            gdf.to_parquet(chunk_path)
        self.total_written += len(self.buffer)
        self.buffer = []

    def close(self):
        self.flush()


# =============================================================================
# 6. FAST-COPY (WSL2 bind-mount bypass)
# =============================================================================

def _drop_file_cache(path: Path):
    if os.name != "posix":
        return
    try:
        with open(path, "rb") as f:
            ctypes.CDLL("libc.so.6").posix_fadvise(f.fileno(), 0, 0, 4)
    except Exception:
        pass


_FAST_TMP = Path("/data/_palm_inference_tmp")


def _is_wsl() -> bool:
    try:
        return 'microsoft' in Path('/proc/version').read_text().lower()
    except OSError:
        return False


def _get_fast_copy(src: Path) -> tuple[Path, bool]:
    """Copy src to native Linux filesystem for fast I/O, if and only if it is
    on a slow WSL2 bind-mount. Reuses an existing copy if one is present.

    WSL2 only: on a native Linux machine paths under /mnt/ are ordinary
    mounts, and copying every mosaic into /data would only waste disk."""
    if not _is_wsl():
        return src, False
    src_str = str(src)
    slow_prefixes = ('/workspace', '/mnt/', '/run/desktop')
    if not any(src_str.startswith(p) for p in slow_prefixes):
        return src, False

    try:
        _FAST_TMP.mkdir(parents=True, exist_ok=True)
        dst = _FAST_TMP / src.name
        # Reuse existing copy if it matches source size (crash-safe restart).
        if dst.exists():
            try:
                if dst.stat().st_size == src.stat().st_size:
                    logger.info(f"  [fast-copy] reusing existing {dst.name}")
                    return dst, True
            except OSError:
                pass
            dst.unlink()
        logger.info(f"  [fast-copy] {src.name} -> {_FAST_TMP} ...")
        t0 = time.time()
        import shutil
        shutil.copy2(src, dst)
        mb = dst.stat().st_size / 1e6
        elapsed = time.time() - t0
        logger.info(f"  [fast-copy] done: {mb:.0f} MB in {elapsed:.1f}s "
                    f"({mb/elapsed:.0f} MB/s)")
        return dst, True
    except Exception as e:
        logger.warning(f"  [fast-copy] failed ({e}), using original path")
        return src, False


def _delete_fast_copy(path: Path):
    try:
        if path.exists() and str(path).startswith(str(_FAST_TMP)):
            path.unlink()
            logger.info(f"  [fast-copy] deleted temp copy: {path.name}")
    except Exception as e:
        logger.warning(f"  [fast-copy] cleanup failed: {e}")


def _output_path_for(input_path: Path, rel_path: Path,
                     cfg: InferenceConfig) -> Path:
    suffix = {"gpkg": ".gpkg", "shp": ".shp", "parquet": ".parquet"}[cfg.output_format]
    return cfg.output_root / rel_path.with_suffix(suffix)


# =============================================================================
# 7. PRODUCER–CONSUMER PIPELINE
# =============================================================================

# Sentinel to signal queue shutdown.
_SENTINEL = object()


def process_one_input(
    input_path: Path,
    rel_path: Path,
    tile_jobs: pd.DataFrame,
    detector: FastBatchedDetector,
    cfg: InferenceConfig,
    pbar: Optional[tqdm] = None,
) -> Tuple[int, int]:
    """Run inference on all pending tiles for one input GeoTIFF.

    Pipelined execution:
        DataLoader (N workers, reads tiles from disk)
          |
          v
        GPU producer thread (runs forward pass, queues results)
          |
          v
        Vectoriser thread pool (emits polygons)
          |
          v
        Main thread (writes to GPKG)
    """
    fast_path, was_copied = _get_fast_copy(input_path)

    with rasterio.open(fast_path) as src:
        crs = src.crs
        transform = src.transform
        image_w, image_h = src.width, src.height

    output_path = _output_path_for(input_path, rel_path, cfg)
    writer = StreamingGeoWriter(output_path, crs=crs, fmt=cfg.output_format)

    origins = list(zip(
        tile_jobs["x"].astype(int).tolist(),
        tile_jobs["y"].astype(int).tolist(),
    ))
    dataset = TiledGeoTiffDataset(str(fast_path), origins, cfg.tile_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        collate_fn=collate_tiles,
        pin_memory=False,
        shuffle=False,
        persistent_workers=False,
    )

    # Queues sized modestly: we want back-pressure so memory stays bounded,
    # but large enough that short stalls on either side do not block the GPU.
    gpu_queue: queue.Queue = queue.Queue(maxsize=4)    # GPU output -> vectorisers
    vec_queue: queue.Queue = queue.Queue(maxsize=16)   # vectorisers -> writer

    stats = {"tiles": 0, "polys": 0, "t_start": time.time(),
             "last_log_t": time.time(), "batch_i": 0}
    stop_flag = threading.Event()

    # ------------------- GPU producer -------------------
    def gpu_producer():
        try:
            for tiles, xs, ys, nodata_fracs in loader:
                if stop_flag.is_set():
                    break
                keep_mask = nodata_fracs.numpy() < cfg.nodata_skip_threshold
                if not keep_mask.any():
                    gpu_queue.put(("empty", len(tiles), None, None, None))
                    continue
                kept_tiles = [t for t, k in zip(tiles, keep_mask) if k]
                kept_xs = xs[keep_mask].tolist()
                kept_ys = ys[keep_mask].tolist()
                preds = detector.predict_batch(kept_tiles)
                gpu_queue.put(("batch", len(tiles), preds, kept_xs, kept_ys))
        except Exception as e:
            logger.exception(f"GPU producer failed: {e}")
            stop_flag.set()
        finally:
            gpu_queue.put(_SENTINEL)

    # ------------------- Vectorisers -------------------
    def _vectorise_one(pred, tx, ty):
        px = vectorize_tile(
            pred=pred,
            tile_x=int(tx), tile_y=int(ty),
            tile_w=cfg.tile_size, tile_h=cfg.tile_size,
            image_w=image_w, image_h=image_h,
            overlap=cfg.overlap,
            score_thr=cfg.score_threshold,
            min_area_px=cfg.min_polygon_area_px,
            max_components=cfg.max_mask_components,
            simplify_tol_px=cfg.simplify_tolerance_px,
            intra_tile_nms_iou=cfg.intra_tile_nms_iou,
            mask_iou_dedup_thr=cfg.mask_iou_dedup_thr,
            min_circularity=cfg.min_circularity,
            max_aspect_ratio=cfg.max_aspect_ratio,
        )
        return georeference_polygons(px, transform)

    # Persistent vectoriser pool (built once, reused across all batches).
    n_vec_threads = max(1, getattr(cfg, "num_vec_threads", 4))
    vec_pool = ThreadPoolExecutor(max_workers=n_vec_threads,
                                  thread_name_prefix="vec")

    def gpu_consumer_dispatcher():
        """Pulls GPU batches, submits each tile to the vectoriser pool,
        and forwards results to the writer queue in submission order."""
        try:
            while True:
                item = gpu_queue.get()
                if item is _SENTINEL:
                    break
                kind, n_tiles_in_batch, preds, kept_xs, kept_ys = item
                if kind == "empty":
                    vec_queue.put(("progress", n_tiles_in_batch, None))
                    continue
                # Submit all tiles in this batch; collect in submission order.
                futures = [vec_pool.submit(_vectorise_one, p, tx, ty)
                           for p, tx, ty in zip(preds, kept_xs, kept_ys)]
                records_all = []
                for f in futures:
                    try:
                        records_all.extend(f.result())
                    except Exception as e:
                        logger.warning(f"vectorise failed: {e}")
                vec_queue.put(("records", n_tiles_in_batch, records_all))
        except Exception as e:
            logger.exception(f"Dispatcher failed: {e}")
            stop_flag.set()
        finally:
            vec_queue.put(_SENTINEL)

    # Launch producer and dispatcher.
    t_prod = threading.Thread(target=gpu_producer, name="gpu-producer",
                              daemon=True)
    t_disp = threading.Thread(target=gpu_consumer_dispatcher, name="dispatcher",
                              daemon=True)
    t_prod.start()
    t_disp.start()

    # Main thread drains the writer queue.
    try:
        while True:
            item = vec_queue.get()
            if item is _SENTINEL:
                break
            kind, n_in_batch, payload = item
            if kind == "records":
                writer.add(payload)
                stats["polys"] += len(payload)
            stats["tiles"] += n_in_batch
            stats["batch_i"] += 1

            if pbar is not None:
                pbar.update(n_in_batch)

            if len(writer.buffer) >= cfg.write_chunk_size:
                writer.flush()

            if stats["batch_i"] % cfg.log_interval == 0:
                now = time.time()
                elapsed = now - stats["last_log_t"]
                tile_rate = (cfg.batch_size * cfg.log_interval) / max(elapsed, 1e-6)
                logger.info(
                    f"[{input_path.name}] batch {stats['batch_i']}: "
                    f"{tile_rate:.1f} tiles/s, polygons so far: {stats['polys']:,}"
                )
                stats["last_log_t"] = now
    finally:
        stop_flag.set()
        t_prod.join(timeout=60)
        t_disp.join(timeout=60)
        vec_pool.shutdown(wait=True)

    writer.close()
    total_elapsed = time.time() - stats["t_start"]
    logger.info(
        f"[{input_path.name}] DONE: {stats['tiles']:,} tiles -> "
        f"{stats['polys']:,} polygons in {total_elapsed/60:.1f} min "
        f"({stats['tiles']/max(total_elapsed,1e-6):.1f} tiles/s avg) -> {output_path}"
    )

    if cfg.drop_file_cache:
        _drop_file_cache(fast_path)
    if was_copied:
        _delete_fast_copy(fast_path)

    return stats["tiles"], stats["polys"]


def run_pipeline(cfg: InferenceConfig):
    """Top-level orchestrator."""
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output root: {cfg.output_root}")
    logger.info(f"Config:\n{asdict(cfg)}")

    inputs = discover_inputs(cfg.input_roots, cfg.min_file_size_bytes)
    if not inputs:
        logger.error("No inputs discovered. Exiting.")
        return
    manifest = load_or_build_manifest(inputs, cfg)

    logger.info(f"Loading model: {cfg.checkpoint_file}")
    detector = FastBatchedDetector(cfg)

    pending_per_input = manifest[~manifest["done"]].groupby("input_path")
    total_pending = (~manifest["done"]).sum()
    logger.info(f"Pending tiles: {total_pending:,} across "
                f"{pending_per_input.ngroups} inputs")

    pbar = tqdm(total=total_pending, desc="Tiles", unit="tile")

    for input_path_str, tile_jobs in pending_per_input:
        input_path = Path(input_path_str)
        if not input_path.exists():
            logger.warning(f"Input vanished, skipping: {input_path}")
            continue
        rel_path = Path(tile_jobs["rel_path"].iloc[0])

        try:
            process_one_input(input_path, rel_path, tile_jobs,
                              detector, cfg, pbar=pbar)
            mask = manifest["input_path"] == input_path_str
            manifest.loc[mask, "done"] = True
            manifest.to_parquet(cfg.manifest_path)
        except Exception as e:
            logger.exception(f"Failed processing {input_path}: {e}")
            continue

        torch.cuda.empty_cache()
        gc.collect()

    pbar.close()
    logger.info("Pipeline complete. Run postprocess.merge_and_dedup() next.")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Country-scale date palm inference pipeline (v2, pipelined)"
    )
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--score-thr", type=float, default=0.35)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    cfg = InferenceConfig(
        input_roots=[Path(p) for p in args.input_root],
        output_root=Path(args.output_root),
        config_file=Path(args.config_file),
        checkpoint_file=Path(args.checkpoint),
        device=args.device,
        batch_size=args.batch_size,
        score_threshold=args.score_thr,
        use_fp16=not args.no_fp16,
    )
    run_pipeline(cfg)
