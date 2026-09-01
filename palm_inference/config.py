"""Configuration for country-scale date palm inference pipeline.

All settings live here. Pass an instance of `InferenceConfig` to the pipeline
entry point. Defaults are tuned for:
  - a 24 GB GPU + 64 GB RAM
  - Spatial-Mamba-S + Mask R-CNN, FP16 inference (the deployed model)
  - 0.15 m GSD (ground sample distance) imagery
  - Mixed input topology (folders of small tiles AND large mosaics)

The path defaults below are placeholders: the CLI (run_inference.py) requires
all of them explicitly, and programmatic callers must set them too.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class InferenceConfig:
    # ============================================================
    # PATHS
    # ============================================================
    # Root directories to scan recursively for GeoTIFFs.
    # Pass a list to handle multiple input trees in one run.
    input_roots: List[Path] = field(default_factory=list)

    # Where outputs go. One subdirectory per input_root will be created.
    output_root: Path = Path("palm_inference_output")

    # MMDetection model. The deployed configuration is
    # configs/Custom/5_deployment_finetune/maskrcnn_spatialmamba_s_deploy.py;
    # the trained checkpoint is not distributed (see WITHHELD.md).
    config_file: Path = Path(
        "configs/Custom/5_deployment_finetune/maskrcnn_spatialmamba_s_deploy.py"
    )
    checkpoint_file: Path = Path("checkpoints/deployed_checkpoint.pth")

    # Manifest (resumability) location.
    # If None, defaults to <output_root>/_manifest.parquet
    manifest_path: Optional[Path] = None

    # ============================================================
    # TILING
    # ============================================================
    # Tile size in pixels. 512 matches typical detection training resolution.
    tile_size: int = 1024

    # Pixel overlap between adjacent tiles. Tiles share `overlap` pixels with
    # each neighbour to avoid splitting palms across tile seams.
    # Rule of thumb: overlap >= max expected object diameter in pixels.
    # At 0.15m GSD a 12m palm = 80 px, so 128 px is comfortable.
    overlap: int = 256

    # Skip tiles where the read returns >this fraction of nodata pixels.
    # Useful for irregular mosaic boundaries.
    nodata_skip_threshold: float = 0.95

    # ============================================================
    # INFERENCE
    # ============================================================
    device: str = "cuda:0"
    batch_size: int = 6      # TITAN RTX 24GB safely handles 6 @ 1024x1024 FP16
    use_fp16: bool = True    # Tensor-Core acceleration; 1.5-2x speedup
    score_threshold: float = 0.35

    # DataLoader workers. Each opens its own rasterio handle.
    # 4 is optimal for a single large GeoTIFF; higher values cause page-cache
    # contention without throughput benefit. Confirmed empirically: at 8
    # workers the GPU-bound pipeline ran at ~0.35 tiles/s; dropping to 4 with
    # producer-consumer pipelining raises utilisation dramatically.
    num_workers: int = 4
    prefetch_factor: int = 2

    # Number of CPU threads used for per-tile vectorisation (contour extraction,
    # mask smoothing, polygon construction). Runs concurrently with GPU.
    # On a 16-thread CPU, 4 is a safe default; increase if the GPU is saturated
    # and polygon emission is the new bottleneck (visible as queue back-pressure
    # on the writer side).
    num_vec_threads: int = 4

    # NMS threshold overrides applied at inference time (optional).
    # Default None = use the value from the MMDetection config file, which
    # matches the validated training protocol (rcnn iou_threshold=0.7).
    # Set these only for deliberate false-positive suppression experiments;
    # aggressive values (e.g. 0.4) will suppress genuine adjacent palms in
    # dense plantation scenes and degrade recall.
    rcnn_nms_iou_threshold: Optional[float] = None  # e.g. 0.4 for FP suppression
    rpn_nms_iou_threshold:  Optional[float] = None  # e.g. 0.6 for RPN tightening

    # ============================================================
    # POSTPROCESSING
    # ============================================================
    # Secondary per-tile IoU suppression applied after score thresholding and
    # before vectorization. Removes duplicate proposals on the same palm crown
    # (two offset boxes for the same palm that survived the model's own NMS at
    # iou_threshold=0.7). Set to 1.0 to disable.
    # Rationale for 0.5: adjacent plantation palms in dense rows typically
    # have inter-crown bbox IoU of 0.1–0.3; duplicate proposals for the same
    # palm have bbox IoU of 0.5–0.9. 0.5 sits cleanly between these regimes.
    intra_tile_nms_iou: float = 0.5

    # Mask-pixel-IoU threshold for final mask-level duplicate suppression.
    # Applied after bbox-IoU NMS. Catches cases where two proposals have low
    # bbox IoU (e.g. one box is taller and offset) but their smoothed masks
    # overlap heavily because both cover the same palm crown pixels.
    # Lower than bbox threshold (0.3 vs 0.5): genuinely separate adjacent palms
    # have near-zero mask overlap even when bboxes overlap significantly.
    # Set to 1.0 to disable.
    mask_iou_dedup_thr: float = 0.3

    # Minimum polygon circularity (4π·area/perimeter²). Range [0, 1].
    # Date palm crowns viewed from nadir are approximately circular (0.6–1.0).
    # Elongated false positives (shadows, roads, irrigation channels, fence
    # posts) have circularity < 0.35. This is the primary filter for the
    # thin vertical/diagonal polygon artefacts visible in the output.
    min_circularity: float = 0.35

    # Maximum bounding-box aspect ratio (long_side / short_side).
    # Rejects detections whose bounding box is too elongated to be a palm crown.
    # At 0.15 m GSD a leaning palm has aspect ≤ 3.0; higher values are almost
    # always non-palm artefacts (shadows, walls, roads).
    max_aspect_ratio: float = 4.0

    # Polygon simplification tolerance in pixels (Douglas-Peucker).
    # 2 px @ 0.15m GSD = 0.3m ground tolerance: preserves crown shape,
    # reduces vertex count ~10x, dramatically smaller output files.
    simplify_tolerance_px: float = 2.0

    # Minimum polygon area in pixels. Filters detection noise.
    # At 0.15 m GSD: 1 px = 0.0225 m².
    # Real date palm crowns range from ~4 m diameter (young) to ~12 m (mature):
    #   4m  crown → radius 2m  → area ~12.6 m²  → ~560 px
    #   12m crown → radius 6m  → area ~113 m²   → ~5000 px
    # Setting 500 px (~11 m² = ~3.8 m diameter) as the lower bound filters out
    # noise detections (shadows, small shrubs) while retaining young palms.
    # The original value of 100 px (2.25 m²) was far too permissive and was
    # directly responsible for the small-circle false positives in the output.
    min_polygon_area_px: int = 500

    # Reject polygons whose mask has more components than this.
    # Real palm masks are 1-3 connected components; >5 is likely junk.
    max_mask_components: int = 5

    # Streaming write batch size. Polygons are accumulated in memory then
    # flushed in chunks. 5000 ~= 50 MB peak per chunk, safe for 64 GB RAM.
    write_chunk_size: int = 5000

    # ============================================================
    # CROSS-IMAGE DEDUPLICATION
    # ============================================================
    # IoU above which two polygons in different source images are treated
    # as the same palm (keep the higher-scoring one).
    dedup_iou_threshold: float = 0.5

    # Polygons whose bbox is within this distance (in CRS units, usually
    # metres) of any input-image boundary are candidates for cross-image
    # dedup. 5m comfortably covers a palm crown.
    dedup_boundary_buffer: float = 5.0

    # ============================================================
    # OUTPUT
    # ============================================================
    # Output format. GeoPackage is preferred (no 2GB limit, UTF-8 native).
    output_format: str = "gpkg"  # "gpkg" | "shp" | "parquet"

    # Compute and emit a per-100m density raster alongside vector output.
    emit_density_raster: bool = False
    density_cell_size_m: float = 100.0

    # ============================================================
    # OPERATIONAL
    # ============================================================
    # Log progress every N tile batches.
    log_interval: int = 50

    # If True, drop the OS file cache for each input after processing.
    # Recommended for country-scale runs to avoid memory pressure.
    drop_file_cache: bool = True

    # Skip files smaller than this many bytes (likely corrupt / empty).
    min_file_size_bytes: int = 100_000

    def __post_init__(self):
        # Coerce string paths to Path objects (config files often have strings).
        self.input_roots = [Path(p) for p in self.input_roots]
        self.output_root = Path(self.output_root)
        self.config_file = Path(self.config_file)
        self.checkpoint_file = Path(self.checkpoint_file)
        if self.manifest_path is not None:
            self.manifest_path = Path(self.manifest_path)
        else:
            self.manifest_path = self.output_root / "_manifest.parquet"

        if self.overlap >= self.tile_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be < tile_size ({self.tile_size})"
            )
        if self.output_format not in {"gpkg", "shp", "parquet"}:
            raise ValueError(f"Unsupported output_format: {self.output_format}")
