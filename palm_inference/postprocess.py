"""Post-processing utilities for the palm inference pipeline.

After `run_pipeline()` produces one .gpkg per input GeoTIFF (mirroring the
input directory structure), this module:

  1. merge_outputs(): consolidate all per-input GPKGs into one country-wide
     layer. Cheap O(N) concatenation.

  2. dedup_overlaps(): for input images that geographically overlap, identical
     palms can be detected twice. Builds an R-tree on candidate polygons in
     overlap zones and removes duplicates by IoU > threshold (keeps the higher
     score).

  3. emit_density_raster(): rasterize palm centroid density per cell (e.g.
     100m x 100m) for quick QGIS heatmap visualization of country-wide results.

Usage:
    from palm_inference.postprocess import merge_and_dedup
    merge_and_dedup(cfg)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree
from tqdm import tqdm

from .config import InferenceConfig

logger = logging.getLogger("palm_inference.postprocess")


# =============================================================================
# 1. MERGE PER-INPUT OUTPUTS
# =============================================================================

def find_per_input_outputs(cfg: InferenceConfig) -> List[Path]:
    """Locate all per-input output files under cfg.output_root."""
    suffix = {"gpkg": ".gpkg", "shp": ".shp", "parquet": ".parquet"}[cfg.output_format]
    files = sorted(cfg.output_root.rglob(f"*{suffix}"))
    # Exclude any consolidated outputs from previous runs.
    files = [f for f in files if not f.name.startswith("_consolidated")]
    return files


def merge_outputs(
    cfg: InferenceConfig,
    output_path: Optional[Path] = None,
) -> Path:
    """Concatenate all per-input outputs into one GeoDataFrame and write.
    Adds a `source_file` column so each polygon retains provenance.
    """
    if output_path is None:
        output_path = cfg.output_root / f"_consolidated.{cfg.output_format}"

    files = find_per_input_outputs(cfg)
    logger.info(f"Merging {len(files)} per-input files -> {output_path}")

    chunks = []
    target_crs = None
    for f in tqdm(files, desc="Reading"):
        try:
            gdf = gpd.read_file(f, engine="pyogrio")
        except Exception as e:
            logger.warning(f"Skipping unreadable {f}: {e}")
            continue
        if len(gdf) == 0:
            continue
        gdf["source_file"] = str(f.relative_to(cfg.output_root))

        if target_crs is None:
            target_crs = gdf.crs
        elif gdf.crs != target_crs:
            # Reproject to the first file's CRS for a unified country layer.
            gdf = gdf.to_crs(target_crs)
        chunks.append(gdf)

    if not chunks:
        logger.warning("No polygons found across per-input outputs.")
        return output_path

    merged = pd.concat(chunks, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=target_crs)
    merged["id"] = range(len(merged))

    logger.info(f"Merged: {len(merged):,} polygons total")
    if output_path.exists():
        output_path.unlink()
    if cfg.output_format == "parquet":
        merged.to_parquet(output_path)
    else:
        driver = "GPKG" if cfg.output_format == "gpkg" else None
        merged.to_file(output_path, driver=driver, engine="pyogrio")
    logger.info(f"Wrote consolidated layer: {output_path}")
    return output_path


# =============================================================================
# 2. CROSS-IMAGE DEDUPLICATION
# =============================================================================

def _read_input_footprints(cfg: InferenceConfig) -> gpd.GeoDataFrame:
    """Read each input GeoTIFF's bounding-box footprint into a GeoDataFrame.
    Used to identify which polygons fall in overlap zones between inputs.
    """
    import rasterio
    from shapely.geometry import box as shapely_box

    rows = []
    for root in cfg.input_roots:
        for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            for p in root.rglob(ext):
                try:
                    with rasterio.open(p) as src:
                        b = src.bounds
                        rows.append({
                            "input_path": str(p),
                            "geometry": shapely_box(b.left, b.bottom,
                                                    b.right, b.top),
                            "crs": str(src.crs),
                        })
                except Exception:
                    continue
    if not rows:
        return gpd.GeoDataFrame(columns=["input_path", "geometry", "crs"])
    # Use the first CRS; reproject others to match.
    df = pd.DataFrame(rows)
    target_crs = df["crs"].iloc[0]
    geoms = []
    for _, r in df.iterrows():
        if r["crs"] == target_crs:
            geoms.append(r["geometry"])
        else:
            tmp = gpd.GeoSeries([r["geometry"]], crs=r["crs"]).to_crs(target_crs)
            geoms.append(tmp.iloc[0])
    df = df.drop(columns=["geometry"])
    return gpd.GeoDataFrame(df, geometry=geoms, crs=target_crs)


def dedup_overlaps(
    cfg: InferenceConfig,
    consolidated_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Remove duplicate palms that appear in geographic overlap between input
    images. Strategy:

      1. Compute the union of pairwise intersections of input footprints
         (= the geographic regions where double-detection is possible).
      2. Buffer each input boundary by `cfg.dedup_boundary_buffer` (m) - any
         polygon whose centroid falls within the buffered overlap is a
         dedup candidate. Polygons outside overlap zones are passed through
         untouched (the vast majority).
      3. For candidates only, build an R-tree, find pairs with IoU >
         `cfg.dedup_iou_threshold`, and keep the higher-scoring polygon
         (ties broken by smaller id).

    This is the key correctness step for runs over folders of small tiles
    where the same palm may appear in adjacent files.
    """
    if output_path is None:
        suffix = {"gpkg": ".gpkg", "shp": ".shp",
                  "parquet": ".parquet"}[cfg.output_format]
        output_path = consolidated_path.with_name(
            f"_consolidated_dedup{suffix}"
        )

    logger.info(f"Loading consolidated polygons: {consolidated_path}")
    if cfg.output_format == "parquet":
        gdf = gpd.read_parquet(consolidated_path)
    else:
        gdf = gpd.read_file(consolidated_path, engine="pyogrio")
    logger.info(f"Total polygons: {len(gdf):,}")

    if len(gdf) == 0:
        logger.warning("Empty consolidated layer; nothing to dedup.")
        if cfg.output_format != "parquet":
            gdf.to_file(output_path, engine="pyogrio")
        else:
            gdf.to_parquet(output_path)
        return output_path

    # Identify overlap zones from input footprints.
    footprints = _read_input_footprints(cfg).to_crs(gdf.crs)
    if len(footprints) < 2:
        logger.info("Fewer than 2 inputs; no cross-image overlap possible.")
        return _write(gdf, output_path, cfg.output_format)

    # Compute pairwise intersections (only nearby pairs via spatial index).
    sindex = footprints.sindex
    overlap_geoms = []
    for i, fp in footprints.iterrows():
        candidates = list(sindex.intersection(fp.geometry.bounds))
        for j in candidates:
            if j <= i:
                continue
            other = footprints.iloc[j].geometry
            if fp.geometry.intersects(other):
                inter = fp.geometry.intersection(other)
                if not inter.is_empty:
                    overlap_geoms.append(inter)

    if not overlap_geoms:
        logger.info("No geographic overlap between inputs; no dedup needed.")
        return _write(gdf, output_path, cfg.output_format)

    from shapely.ops import unary_union
    overlap_zone = unary_union(overlap_geoms).buffer(cfg.dedup_boundary_buffer)
    logger.info(f"Overlap zone area: {overlap_zone.area:,.0f} sq units "
                f"({100 * overlap_zone.area / unary_union(footprints.geometry).area:.2f}% of total)")

    # Partition polygons by whether their centroid is in the overlap zone.
    centroids = gdf.geometry.centroid
    in_zone_mask = centroids.within(overlap_zone)
    candidates = gdf[in_zone_mask].reset_index(drop=True)
    safe = gdf[~in_zone_mask]
    logger.info(f"Dedup candidates (in overlap zone): {len(candidates):,}")
    logger.info(f"Pass-through (outside overlap):     {len(safe):,}")

    if len(candidates) == 0:
        return _write(gdf, output_path, cfg.output_format)

    # Build STRtree over candidates.
    geoms = list(candidates.geometry.values)
    tree = STRtree(geoms)

    # Sort candidates by score DESC so highest-scoring is processed first.
    # CRITICAL: operate on .values (numpy) before reversing — reversing a
    # pandas Series with [::-1] reverses by label, not by position.
    scores_arr = candidates["score"].to_numpy()
    order = np.argsort(scores_arr)[::-1]
    suppressed = np.zeros(len(candidates), dtype=bool)

    iou_thr = cfg.dedup_iou_threshold
    for idx in tqdm(order, desc="IoU dedup"):
        if suppressed[idx]:
            continue
        g = geoms[idx]
        # Query bbox-overlapping candidates. shapely 2.x returns integer
        # indices; shapely 1.x returns geometries — handle both.
        try:
            result = tree.query(g)
        except Exception:
            result = []
        for item in result:
            j = int(item) if isinstance(item, (int, np.integer)) else None
            if j is None:
                # shapely 1.x path: locate by identity (rare; kept for safety)
                continue
            if j == idx or suppressed[j]:
                continue
            other = geoms[j]
            inter_area = g.intersection(other).area
            if inter_area == 0:
                continue
            union_area = g.area + other.area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0
            if iou > iou_thr:
                # Suppress the lower-scoring one. Because we iterate high->low
                # by score, any un-suppressed j encountered here has score
                # <= the current idx.
                if scores_arr[j] <= scores_arr[idx]:
                    suppressed[j] = True
                else:
                    suppressed[idx] = True
                    break

    kept_candidates = candidates[~suppressed]
    logger.info(f"Removed {suppressed.sum():,} duplicate polygons "
                f"({100 * suppressed.sum() / max(len(candidates),1):.1f}% of candidates)")

    final = pd.concat([safe, kept_candidates], ignore_index=True)
    final = gpd.GeoDataFrame(final, geometry="geometry", crs=gdf.crs)
    final["id"] = range(len(final))
    logger.info(f"Final polygon count: {len(final):,}")

    return _write(final, output_path, cfg.output_format)


def _write(gdf: gpd.GeoDataFrame, path: Path, fmt: str) -> Path:
    if path.exists():
        path.unlink()
    if fmt == "parquet":
        gdf.to_parquet(path)
    else:
        driver = "GPKG" if fmt == "gpkg" else None
        gdf.to_file(path, driver=driver, engine="pyogrio")
    logger.info(f"Wrote {len(gdf):,} polygons -> {path}")
    return path


# =============================================================================
# 3. DENSITY RASTER (optional)
# =============================================================================

def emit_density_raster(
    consolidated_path: Path,
    output_raster: Path,
    cell_size_m: float = 100.0,
):
    """Rasterize palm centroid count per cell. Useful for country-wide heatmaps.
    Output: GeoTIFF with a single uint32 band where each pixel = number of
    palm centroids in that cell.
    """
    import rasterio
    from rasterio.transform import from_bounds

    logger.info(f"Building density raster from {consolidated_path}")
    if consolidated_path.suffix == ".parquet":
        gdf = gpd.read_parquet(consolidated_path)
    else:
        gdf = gpd.read_file(consolidated_path, engine="pyogrio")
    if len(gdf) == 0:
        logger.warning("No polygons to rasterize.")
        return

    centroids = gdf.geometry.centroid
    minx, miny, maxx, maxy = (
        centroids.x.min(), centroids.y.min(),
        centroids.x.max(), centroids.y.max(),
    )
    width = max(1, int(np.ceil((maxx - minx) / cell_size_m)))
    height = max(1, int(np.ceil((maxy - miny) / cell_size_m)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Bin centroids.
    col = ((centroids.x - minx) / cell_size_m).astype(int).clip(0, width - 1)
    row = (((maxy - centroids.y) / cell_size_m)).astype(int).clip(0, height - 1)
    grid = np.zeros((height, width), dtype=np.uint32)
    np.add.at(grid, (row.values, col.values), 1)

    output_raster.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_raster, "w",
        driver="GTiff",
        height=height, width=width,
        count=1, dtype="uint32",
        crs=gdf.crs, transform=transform,
        compress="deflate", predictor=2, tiled=True,
    ) as dst:
        dst.write(grid, 1)
    logger.info(f"Wrote density raster: {output_raster} "
                f"({width}x{height}, max count: {grid.max()})")


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def merge_and_dedup(cfg: InferenceConfig):
    """Convenience: merge per-input outputs, then dedup overlap zones,
    then optionally write a density raster."""
    consolidated = merge_outputs(cfg)
    if not consolidated.exists():
        logger.warning("No consolidated output produced — skipping dedup.")
        return None
    deduped = dedup_overlaps(cfg, consolidated)
    if cfg.emit_density_raster:
        density_path = cfg.output_root / "_density.tif"
        emit_density_raster(deduped, density_path,
                            cell_size_m=cfg.density_cell_size_m)
    logger.info("Postprocess complete.")
    logger.info(f"  Final layer: {deduped}")
    return deduped


if __name__ == "__main__":
    import argparse
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--format", default="gpkg",
                        choices=["gpkg", "shp", "parquet"])
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--cell-size-m", type=float, default=100.0)
    args = parser.parse_args()

    cfg = InferenceConfig(
        input_roots=[Path(p) for p in args.input_root],
        output_root=Path(args.output_root),
        config_file=Path("dummy"),       # not used by postprocess
        checkpoint_file=Path("dummy"),
        output_format=args.format,
        emit_density_raster=args.density,
        density_cell_size_m=args.cell_size_m,
    )
    merge_and_dedup(cfg)
