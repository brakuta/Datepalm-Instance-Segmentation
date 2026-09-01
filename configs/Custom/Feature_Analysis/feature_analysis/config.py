# -*- coding: utf-8 -*-
"""Configuration schema and (de)serialisation for the feature-analysis suite.

A single JSON document drives every phase. The schema is expressed as
dataclasses with explicit validation so that malformed configurations fail
early with actionable messages rather than mid-run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os.path as osp
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .util import LOGGER


@dataclass
class ModelSpec:
    label: str                 # unique display label, e.g. "SpatialMamba-S"
    family: str                # 'CNN' | 'Transformer' | 'Mamba'
    config: str                # MMDet config path
    checkpoint: str            # default trained checkpoint path
    # Optional per-tile-set override: {tile_set_name: checkpoint_path}.
    # Used by the extraction phase so that domain-specific best weights
    # (e.g. best_UAV_*, best_GE_*) feed the corresponding evaluation.
    # Tile sets absent from the mapping use `checkpoint`.
    #
    # METHODOLOGICAL NOTE: cross-resolution comparisons (the Bures
    # resolution-invariance headline and any metric contrasting two tile
    # sets) are only interpretable when both sides derive from the SAME
    # weight state. Populate `checkpoints` only for evaluation-aligned,
    # within-resolution analyses, and keep such runs in a separate
    # cache_dir/out_dir from the canonical single-checkpoint run.
    checkpoints: Optional[Dict[str, str]] = None

    def checkpoint_for(self, set_name: str) -> str:
        if self.checkpoints and set_name in self.checkpoints:
            return self.checkpoints[set_name]
        return self.checkpoint


@dataclass
class TileSetSpec:
    name: str                  # resolution group, e.g. 'UAV_5cm'
    img_dir: str               # directory of test tiles
    pattern: str = "*.png"     # glob pattern within img_dir
    coco_ann: Optional[str] = None   # optional COCO file -> palm-count stratification


@dataclass
class AnalysisConfig:
    out_dir: str
    cache_dir: str
    models: List[ModelSpec]
    tile_sets: List[TileSetSpec]
    device: str = "cuda:0"
    seed: int = 42
    fpn_levels: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    n_tiles: int = 40                       # sampled tiles per resolution group
    max_locations_per_tile: int = 2048      # spatial subsample per tile/level
    bootstrap: int = 1000
    rbf_cka: bool = False                   # also compute non-linear RBF CKA
    pca_tile_index: int = 0
    map_gap_csv: Optional[str] = None       # columns: label,map_gap (optional)

    # multiscale visualisation
    vis_source: str = "neck"                # 'neck' (FPN) or 'backbone' (stages)
    vis_levels: Optional[List[int]] = None  # defaults to fpn_levels
    vis_models: Optional[List[str]] = None  # subset of labels; default all
    showcase_tiles: Optional[Dict[str, List[str]]] = None

    # composite "figure recipe" specifications
    composite_figures: Optional[List[Dict]] = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_json(path: str) -> "AnalysisConfig":
        with open(path, "r") as f:
            raw = json.load(f)
        try:
            models = [ModelSpec(**m) for m in raw.pop("models")]
            tile_sets = [TileSetSpec(**t) for t in raw.pop("tile_sets")]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed 'models' or 'tile_sets' block: {exc}") from exc
        try:
            cfg = AnalysisConfig(models=models, tile_sets=tile_sets, **raw)
        except TypeError as exc:
            raise ValueError(f"Unknown or invalid configuration key: {exc}") from exc
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.models:
            raise ValueError("At least one model must be specified.")
        if not self.tile_sets:
            raise ValueError("At least one tile set must be specified.")
        labels = [m.label for m in self.models]
        if len(set(labels)) != len(labels):
            raise ValueError("Model labels must be unique.")
        names = [t.name for t in self.tile_sets]
        if len(set(names)) != len(names):
            raise ValueError("Tile-set names must be unique.")
        for m in self.models:
            if not osp.isfile(m.config):
                raise FileNotFoundError(f"Config not found for {m.label}: {m.config}")
            if not osp.isfile(m.checkpoint):
                raise FileNotFoundError(f"Checkpoint not found for {m.label}: {m.checkpoint}")
            if m.checkpoints:
                for set_name, ck in m.checkpoints.items():
                    if set_name not in names:
                        LOGGER.warning("%s: per-domain checkpoint mapped to "
                                       "unknown tile set '%s' (ignored).",
                                       m.label, set_name)
                    if not osp.isfile(ck):
                        raise FileNotFoundError(
                            f"Per-domain checkpoint not found for {m.label} "
                            f"[{set_name}]: {ck}")
                LOGGER.info("%s uses per-domain checkpoints for %s; "
                            "cross-resolution invariance metrics spanning "
                            "different weight states are not interpretable -- "
                            "keep this run's cache/out dirs separate from the "
                            "canonical single-checkpoint run.",
                            m.label, sorted(m.checkpoints))
            if m.family not in ("CNN", "Transformer", "Mamba"):
                LOGGER.warning("Unrecognised family '%s' for %s; colour coding "
                               "will default to grey.", m.family, m.label)
        for t in self.tile_sets:
            if not osp.isdir(t.img_dir):
                raise FileNotFoundError(f"Image directory not found for {t.name}: {t.img_dir}")
        if self.fpn_levels != sorted(set(self.fpn_levels)):
            raise ValueError("fpn_levels must be unique and ascending.")
        if self.vis_source not in ("neck", "backbone"):
            raise ValueError("vis_source must be 'neck' or 'backbone'.")

    def config_hash(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


# --------------------------------------------------------------------------- #
EXAMPLE_CONFIG = {
    "out_dir": "/workspace/mmdetection/work_dirs/feature_analysis/run01",
    "cache_dir": "/workspace/mmdetection/work_dirs/feature_analysis/run01/cache",
    "device": "cuda:0",
    "seed": 42,
    "fpn_levels": [0, 1, 2, 3],
    "n_tiles": 40,
    "max_locations_per_tile": 2048,
    "bootstrap": 1000,
    "rbf_cka": False,
    "map_gap_csv": "/workspace/mmdetection/work_dirs/feature_analysis/map_gap.csv",
    "vis_source": "neck",
    "vis_levels": [0, 1, 2, 3],
    "vis_models": ["ConvNeXt-T", "Swin-S", "SpatialMamba-S"],
    "showcase_tiles": {
        "Aerial_15cm": ["/workspace/datasets/COCO/Aerial_15cm/test/REPLACE_aerial_tile.png"]
    },
    "models": [
        {"label": "ResNet-50", "family": "CNN",
         "config": "configs/Custom/3_unified_multisource/maskrcnn_r50_stagec.py",
         "checkpoint": "/workspace/mmdetection/work_dirs/Stage_C/maskrcnn_r50_stagec/iter_80000.pth"},
        {"label": "ConvNeXt-T", "family": "CNN",
         "config": "configs/Custom/3_unified_multisource/maskrcnn_convnext_t_stagec.py",
         "checkpoint": "/workspace/mmdetection/work_dirs/Stage_C/maskrcnn_convnext_t_stagec/iter_80000.pth"},
        {"label": "Swin-S", "family": "Transformer",
         "config": "configs/Custom/3_unified_multisource/maskrcnn_swin_s_stagec.py",
         "checkpoint": "/workspace/mmdetection/work_dirs/Stage_C/maskrcnn_swin_s_stagec/iter_80000.pth"},
        {"label": "SpatialMamba-S", "family": "Mamba",
         "config": "configs/Custom/3_unified_multisource/maskrcnn_spatialmamba_s_stagec.py",
         "checkpoint": "/workspace/mmdetection/work_dirs/Stage_C/maskrcnn_spatialmamba_s_stagec/iter_80000.pth"}
    ],
    "tile_sets": [
        {"name": "UAV_5cm", "img_dir": "/workspace/datasets/COCO/UAV_5cm/test",
         "pattern": "*.png", "coco_ann": "/workspace/datasets/COCO/UAV_5cm/Annotations/test.json"},
        {"name": "Aerial_15cm", "img_dir": "/workspace/datasets/COCO/Aerial_15cm/test",
         "pattern": "*.png", "coco_ann": "/workspace/datasets/COCO/Aerial_15cm/Annotations/test.json"},
        {"name": "GE_15cm", "img_dir": "/workspace/datasets/COCO/GE_15cm/test",
         "pattern": "*.png", "coco_ann": "/workspace/datasets/COCO/GE_15cm/Annotations/test.json"},
        {"name": "WV3_30cm", "img_dir": "/workspace/datasets/COCO/Sat_30cm/test_sat",
         "pattern": "*.png", "coco_ann": "/workspace/datasets/COCO/Sat_30cm/Annotations/test_sat.json"}
    ],
    "composite_figures": [
        {
            "name": "fig3_cross_resolution_stageC",
            "models": ["ConvNeXt-T", "Swin-S", "SpatialMamba-S"],
            "level": 1, "source": "neck", "view": "both",
            "columns": [
                {"label": "UAV 5 cm (in-domain)",
                 "tile": "/workspace/datasets/COCO/UAV_5cm/test/REPLACE_uav_tile.png"},
                {"label": "Aerial 15 cm (in-domain)",
                 "tile": "/workspace/datasets/COCO/Aerial_15cm/test/REPLACE_aerial_tile.png"},
                {"label": "WV-3 30 cm (zero-shot)",
                 "tile": "/workspace/datasets/COCO/Sat_30cm/test_sat/REPLACE_wv3_tile.png"}
            ]
        },
        {
            "name": "fig3b_controlled_gsd",
            "models": ["ConvNeXt-T", "Swin-S", "SpatialMamba-S"],
            "level": 1, "source": "neck", "view": "pcargb",
            "source_tile": "/workspace/datasets/COCO/UAV_5cm/test/REPLACE_uav_tile.png",
            "degrade_factors": [1, 3, 6],
            "degrade_labels": ["5 cm", "15 cm (simulated)", "30 cm (simulated)"]
        }
    ]
}


def make_config(out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(EXAMPLE_CONFIG, f, indent=2)
    LOGGER.info("Wrote example configuration to %s", out_path)
