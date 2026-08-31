# -*- coding: utf-8 -*-
"""Command-line interface and phase dispatch."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from . import __version__
from .analysis import phase_analyze, phase_extract
from .multiscale import phase_anatomy, phase_deep
from .config import AnalysisConfig, make_config
from .util import LOGGER, configure_logging
from .visualization import phase_composite, phase_visualize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="feature_analysis",
        description="Representation-level analysis and publication figures for "
                    "the date palm multi-resolution benchmark.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")
    sub = p.add_subparsers(dest="command", required=True)

    mc = sub.add_parser("make-config", help="Write an annotated example config.")
    mc.add_argument("--out", default="config_feature_analysis.json")

    for name, helptext in (
        ("extract", "GPU: build each model once and cache subsampled features."),
        ("analyze", "CPU: CKA, resolution-invariance, dimensionality, tables, figures."),
        ("deep", "CPU: crown-background separability, level activation share, "
                 "eigenspectrum decay, inter-level similarity."),
        ("visualize", "GPU: multiscale feature grids (families x pyramid levels)."),
        ("composite", "GPU: composite figure recipes (e.g. cross-resolution)."),
        ("anatomy", "GPU: scale-anatomy hero figure (input + GT, per-level "
                    "PCA-RGB with magnitude contours, activation share)."),
        ("run", "Run extract, analyze, deep, visualize, composite, anatomy in sequence."),
    ):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--config", required=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(verbose=not args.quiet)
    try:
        if args.command == "make-config":
            make_config(args.out)
            return 0
        cfg = AnalysisConfig.from_json(args.config)
        if args.command in ("extract", "run"):
            phase_extract(cfg)
        if args.command in ("analyze", "run"):
            phase_analyze(cfg)
        if args.command in ("deep", "run"):
            phase_deep(cfg)
        if args.command in ("visualize", "run"):
            phase_visualize(cfg)
        if args.command in ("composite", "run"):
            phase_composite(cfg)
        if args.command in ("anatomy", "run"):
            phase_anatomy(cfg)
        return 0
    except (FileNotFoundError, ValueError, IndexError, RuntimeError, KeyError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
