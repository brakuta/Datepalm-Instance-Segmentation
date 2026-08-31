# -*- coding: utf-8 -*-
"""Tile sampling (with optional palm-count stratification) and cache I/O.

The cache stores one spatially subsampled feature matrix per
(model, resolution, tile, level) as a ``.npy`` file, decoupling the GPU
extraction phase from the CPU analysis phase.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from glob import glob
from typing import Dict, List

import numpy as np

from .config import AnalysisConfig, TileSetSpec
from .util import stable_seed


def list_tiles(ts: TileSetSpec) -> List[str]:
    files = sorted(glob(osp.join(ts.img_dir, ts.pattern)))
    if not files:
        raise FileNotFoundError(f"No tiles matching {ts.pattern} in {ts.img_dir}")
    return files


def palm_counts(coco_ann: str) -> Dict[str, int]:
    """Map image file stem -> annotation count, from a COCO file."""
    with open(coco_ann, "r") as f:
        coco = json.load(f)
    id2stem = {im["id"]: osp.splitext(osp.basename(im["file_name"]))[0]
               for im in coco["images"]}
    counts: Dict[str, int] = {s: 0 for s in id2stem.values()}
    for ann in coco.get("annotations", []):
        stem = id2stem.get(ann["image_id"])
        if stem is not None:
            counts[stem] += 1
    return counts


def sample_tiles(ts: TileSetSpec, n: int, seed: int) -> List[str]:
    """Palm-count-stratified sample (quartiles) if a COCO file is supplied,
    otherwise a deterministic uniform sample."""
    files = list_tiles(ts)
    rng = np.random.default_rng(stable_seed(seed, ts.name, "tilesample"))
    if n >= len(files):
        return files
    if ts.coco_ann and osp.isfile(ts.coco_ann):
        counts = palm_counts(ts.coco_ann)
        keyed = [(f, counts.get(osp.splitext(osp.basename(f))[0], 0)) for f in files]
        vals = np.array([c for _, c in keyed], dtype=float)
        edges = np.quantile(vals, [0.25, 0.5, 0.75])
        strata: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}
        for f, c in keyed:
            b = int(np.searchsorted(edges, c, side="right"))
            strata[min(b, 3)].append(f)
        per = max(1, n // 4)
        chosen: List[str] = []
        for b in range(4):
            pool = strata[b]
            if not pool:
                continue
            take = min(per, len(pool))
            chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
        remaining = [f for f in files if f not in set(chosen)]
        if len(chosen) < n and remaining:
            extra = rng.choice(remaining, size=min(n - len(chosen), len(remaining)),
                               replace=False).tolist()
            chosen.extend(extra)
        return sorted(chosen[:n])
    return sorted(rng.choice(files, size=n, replace=False).tolist())


# --------------------------------------------------------------------------- #
# Cache layout
# --------------------------------------------------------------------------- #
def cache_path(cfg: AnalysisConfig, model_label: str, set_name: str,
               tile_stem: str, level: int) -> str:
    safe_model = model_label.replace(os.sep, "_").replace(" ", "_")
    d = osp.join(cfg.cache_dir, safe_model, set_name)
    os.makedirs(d, exist_ok=True)
    return osp.join(d, f"{tile_stem}__L{level}.npy")


def load_matrix(cfg: AnalysisConfig, model_label: str, set_name: str,
                tile_stem: str, level: int):
    p = cache_path(cfg, model_label, set_name, tile_stem, level)
    return np.load(p) if osp.isfile(p) else None


def load_sampled(cfg: AnalysisConfig) -> Dict[str, List[str]]:
    with open(osp.join(cfg.cache_dir, "_sampled_tiles.json"), "r") as f:
        sampled = json.load(f)
    return {k: [osp.splitext(osp.basename(t))[0] for t in v] for k, v in sampled.items()}
