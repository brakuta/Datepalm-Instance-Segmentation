#!/usr/bin/env python3
"""Top-level runner: end-to-end UAE date palm mapping.

This script:
  1. Discovers all GeoTIFFs in one or more input directory trees.
  2. Runs MMDetection inference with FP16 batched GPU processing.
  3. Streams polygon output to per-input GeoPackage files.
  4. (Optional) Merges + deduplicates overlap regions across input images.
  5. (Optional) Emits a country-wide palm-density raster for QGIS.

Example: scan two input trees (one with mosaics, one with per-tile folders),
write outputs to a single tree, then merge+dedup at the end.

    python -m palm_inference.run_inference \\
        --input-root /path/to/mosaics \\
        --input-root /path/to/per_tile_folders \\
        --output-root /path/to/palm_output \\
        --config-file configs/Custom/5_deployment_finetune/maskrcnn_spatialmamba_s_deploy.py \\
        --checkpoint /path/to/deployed_checkpoint.pth \\
        --tile-size 1024 --overlap 256 \\
        --batch-size 6 --score-thr 0.30 \\
        --postprocess --density

The config path above is the DEPLOYMENT config and it exists in this tree.
Earlier examples in this file named config directories (`MambaVision/`,
`VMamba/`) that have never existed here, so both failed immediately with a
file-not-found that reads like a broken install rather than a stale
docstring. The trained checkpoint is not distributed with this repository
(see WITHHELD.md).

--score-thr is the PIPELINE threshold, not the model's. The model runs at an
internal score_thr of 0.05 regardless; this value filters what reaches the
GeoPackage. See the deploy config's header for how the two interact with
max_per_img.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from palm_inference import InferenceConfig, run_pipeline, merge_and_dedup


def setup_logging(level: str = "INFO", log_file: str | None = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Inputs
    parser.add_argument(
        "--input-root", action="append", required=True,
        help="Directory to scan recursively for GeoTIFFs. May be repeated."
    )
    parser.add_argument("--output-root", required=True)

    # Model
    parser.add_argument("--config-file", required=True,
                        help="MMDetection config .py")
    parser.add_argument("--checkpoint", required=True,
                        help="MMDetection .pth checkpoint")
    parser.add_argument("--device", default="cuda:0")

    # Inference knobs
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128,
                        help="Pixel overlap between adjacent tiles. >= max "
                             "expected palm crown diameter in pixels.")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--score-thr", type=float, default=0.35)
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable FP16 autocast (slower but exact).")
    parser.add_argument("--num-workers", type=int, default=4)

    # Postprocess
    parser.add_argument("--postprocess", action="store_true",
                        help="Run merge + cross-image dedup after inference.")
    parser.add_argument("--density", action="store_true",
                        help="Emit a per-100m palm-density raster.")
    parser.add_argument("--cell-size-m", type=float, default=100.0)

    # Output
    parser.add_argument("--format", default="gpkg",
                        choices=["gpkg", "shp", "parquet"])

    # Logging
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None,
                        help="Path to log file (in addition to stdout).")

    # Manifest control
    parser.add_argument("--manifest-path", default=None,
                        help="Override manifest location for resumability.")
    parser.add_argument("--reset-manifest", action="store_true",
                        help="Delete existing manifest before running.")

    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)
    log = logging.getLogger("run_inference")

    cfg = InferenceConfig(
        input_roots=[Path(p) for p in args.input_root],
        output_root=Path(args.output_root),
        config_file=Path(args.config_file),
        checkpoint_file=Path(args.checkpoint),
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        score_threshold=args.score_thr,
        use_fp16=not args.no_fp16,
        num_workers=args.num_workers,
        output_format=args.format,
        emit_density_raster=args.density,
        density_cell_size_m=args.cell_size_m,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
    )

    if args.reset_manifest and cfg.manifest_path.exists():
        log.warning(f"Deleting existing manifest: {cfg.manifest_path}")
        cfg.manifest_path.unlink()

    log.info("=" * 70)
    log.info("STAGE 1: Tiled inference")
    log.info("=" * 70)
    run_pipeline(cfg)

    if args.postprocess:
        log.info("=" * 70)
        log.info("STAGE 2: Merge + cross-image dedup")
        log.info("=" * 70)
        merge_and_dedup(cfg)

    log.info("All done.")


if __name__ == "__main__":
    main()
