# -*- coding: utf-8 -*-
"""Qualitative phases: multiscale spatial feature grids and composite
"figure recipes" that isolate a single experimental axis along the columns.

Two views are produced: PCA-to-RGB (spatial structure, with a per-row shared
basis for cross-level/column colour consistency) and activation-magnitude
overlay on the input tile (spatially comparable across rows).
"""

from __future__ import annotations

import os
import os.path as osp
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import AnalysisConfig, ModelSpec
from .extractor import FeatureExtractor, to_chw_numpy
from .metrics import channel_covariance  # noqa: F401  (kept for parity/use)
from .util import (LOGGER, OVERLAY_CMAP, FAMILY_COLORS, save_figure,
                   set_publication_style, style_image_axis)


# --------------------------------------------------------------------------- #
# Rendering primitives
# --------------------------------------------------------------------------- #
def pca_basis_from_levels(level_maps: List[np.ndarray], max_samples: int = 20000,
                          seed: int = 42) -> Optional[np.ndarray]:
    """Fit a shared three-component PCA basis over pooled locations of several
    feature maps that share a channel dimension. Returns a (3, C) basis, or
    None if channel dimensions are inconsistent (e.g. backbone stages)."""
    if not level_maps or len({m.shape[0] for m in level_maps}) != 1:
        return None
    rng = np.random.default_rng(seed)
    rows = []
    per = max(1, max_samples // len(level_maps))
    for m in level_maps:
        c, h, w = m.shape
        x = m.reshape(c, h * w).T
        if x.shape[0] > per:
            x = x[rng.choice(x.shape[0], per, replace=False)]
        rows.append(x)
    X = np.concatenate(rows, 0)
    X = X - X.mean(0, keepdims=True)
    cov = (X.T @ X) / max(X.shape[0] - 1, 1)
    w_eig, v_eig = np.linalg.eigh(cov)
    basis = v_eig[:, ::-1][:, :3].T
    signs = np.sign(basis.sum(axis=1)); signs[signs == 0] = 1.0
    return basis * signs[:, None]


def feature_to_pca_rgb(fmap: np.ndarray, basis: Optional[np.ndarray] = None,
                       norm: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """Project a (C,H,W) feature map to an RGB image via three principal
    components. A supplied `basis`/`norm` is reused for consistency across the
    levels or columns of one model row."""
    c, h, w = fmap.shape
    x = fmap.reshape(c, h * w).T
    x = x - x.mean(0, keepdims=True)
    if basis is None:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        basis = vt[:3]
        signs = np.sign(basis.sum(axis=1)); signs[signs == 0] = 1.0
        basis = basis * signs[:, None]
    comp = (x @ basis.T).reshape(h, w, 3)
    if norm is None:
        lo = np.percentile(comp, 2, axis=(0, 1), keepdims=True)
        hi = np.percentile(comp, 98, axis=(0, 1), keepdims=True)
    else:
        lo, hi = norm
    return np.clip((comp - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def activation_magnitude(fmap: np.ndarray) -> np.ndarray:
    """Per-location L2 norm across channels, robustly normalised to [0,1]."""
    mag = np.linalg.norm(fmap, axis=0)
    hi = np.percentile(mag, 99)
    return np.clip(mag / (hi + 1e-8), 0.0, 1.0)


def _row_basis_and_norm(maps: List[np.ndarray], source: str, seed: int):
    """Shared PCA basis and brightness normalisation for one model row."""
    valid = [m for m in maps if m is not None]
    if source != "neck" or not valid:
        return None, None
    basis = pca_basis_from_levels(valid, seed=seed)
    if basis is None:
        return None, None
    proj = []
    for fm in valid:
        c, h, w = fm.shape
        xx = fm.reshape(c, h * w).T
        xx = xx - xx.mean(0, keepdims=True)
        proj.append(xx @ basis.T)
    allp = np.concatenate(proj, 0)
    norm = (np.percentile(allp, 2, axis=0).reshape(1, 1, 3),
            np.percentile(allp, 98, axis=0).reshape(1, 1, 3))
    return basis, norm


def _load_rgb(img_path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0


def _resize_to(arr2d: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray((np.clip(arr2d, 0, 1) * 255).astype(np.uint8))
    im = im.resize((hw[1], hw[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _level_tag(source: str, level: int) -> str:
    return f"{'P' if source == 'neck' else 'C'}{level + 2}"


# --------------------------------------------------------------------------- #
# Phase 3: multiscale grids (rows = families, columns = pyramid levels)
# --------------------------------------------------------------------------- #
def select_vis_models(cfg: AnalysisConfig) -> List[ModelSpec]:
    if not cfg.vis_models:
        return cfg.models
    keep = set(cfg.vis_models)
    chosen = [m for m in cfg.models if m.label in keep]
    if not chosen:
        raise ValueError("vis_models matched no configured model labels.")
    return chosen


def resolve_showcase(cfg: AnalysisConfig) -> Dict[str, List[str]]:
    if cfg.showcase_tiles:
        return cfg.showcase_tiles
    out: Dict[str, List[str]] = {}
    for ts in cfg.tile_sets:
        files = sorted(glob(osp.join(ts.img_dir, ts.pattern)))
        if files:
            out[ts.name] = [files[0]]
    return out


def phase_visualize(cfg: AnalysisConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    set_publication_style()

    vis_dir = osp.join(cfg.out_dir, "figures", "multiscale")
    os.makedirs(vis_dir, exist_ok=True)
    levels = cfg.vis_levels if cfg.vis_levels is not None else cfg.fpn_levels
    source = cfg.vis_source
    fam_order = {"CNN": 0, "Transformer": 1, "Mamba": 2}
    models = sorted(select_vis_models(cfg), key=lambda m: (fam_order.get(m.family, 9), m.label))
    showcase = resolve_showcase(cfg)
    LOGGER.info("Visualising %d model(s) x %d level(s) over %d tile(s).",
                len(models), len(levels), sum(len(v) for v in showcase.values()))

    capture: Dict[str, Dict[str, Dict[str, Dict[int, np.ndarray]]]] = {}
    tile_paths: Dict[str, Dict[str, str]] = {}
    for m in models:
        ext = FeatureExtractor(m.config, m.checkpoint, cfg.device)
        try:
            for res, tiles in showcase.items():
                for tile in tiles:
                    stem = osp.splitext(osp.basename(tile))[0]
                    try:
                        store = ext.extract_all(tile)
                    except Exception as exc:
                        LOGGER.warning("Vis extraction failed (%s, %s): %s", m.label, stem, exc)
                        continue
                    feats = store[source]
                    per_level = {}
                    for lvl in levels:
                        if lvl >= len(feats):
                            raise IndexError(f"{source} has {len(feats)} levels; "
                                             f"requested {lvl}.")
                        per_level[lvl] = to_chw_numpy(feats[lvl])
                    capture.setdefault(res, {}).setdefault(stem, {})[m.label] = per_level
                    tile_paths.setdefault(res, {})[stem] = tile
        finally:
            ext.close()
        LOGGER.info("Captured visualisation features for %s.", m.label)

    for res, tiles in capture.items():
        for stem, by_model in tiles.items():
            labels = [m.label for m in models if m.label in by_model]
            fams = {m.label: m.family for m in models}
            if not labels:
                continue
            input_rgb = _load_rgb(tile_paths[res][stem])
            ncol = len(levels)

            # structure grid
            fig, axes = plt.subplots(len(labels), ncol,
                                     figsize=(2.6 * ncol, 2.6 * len(labels)), squeeze=False)
            for ri, lab in enumerate(labels):
                lvl_maps = [by_model[lab][lvl] for lvl in levels]
                basis, rownorm = _row_basis_and_norm(lvl_maps, source, cfg.seed)
                for ci, lvl in enumerate(levels):
                    ax = axes[ri][ci]
                    ax.imshow(feature_to_pca_rgb(by_model[lab][lvl], basis=basis, norm=rownorm))
                    style_image_axis(ax)
                    if ri == 0:
                        h, w = by_model[lab][lvl].shape[1:]
                        ax.set_title(f"{_level_tag(source, lvl)}\n{h}x{w}")
                    if ci == 0:
                        ax.set_ylabel(f"{lab}\n({fams[lab]})")
            fig.suptitle(f"Multiscale {source} features (PCA-to-RGB) - {res} / {stem}")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            save_figure(fig, osp.join(vis_dir, f"pcargb_{source}_{res}_{stem}"))

            # magnitude grid
            fig, axes = plt.subplots(len(labels), ncol,
                                     figsize=(2.6 * ncol, 2.6 * len(labels)), squeeze=False)
            ih, iw = input_rgb.shape[:2]
            for ri, lab in enumerate(labels):
                for ci, lvl in enumerate(levels):
                    ax = axes[ri][ci]
                    ax.imshow(input_rgb)
                    ax.imshow(_resize_to(activation_magnitude(by_model[lab][lvl]), (ih, iw)),
                              cmap=OVERLAY_CMAP, alpha=0.5)
                    style_image_axis(ax)
                    if ri == 0:
                        ax.set_title(_level_tag(source, lvl))
                    if ci == 0:
                        ax.set_ylabel(f"{lab}\n({fams[lab]})")
            fig.suptitle(f"Multiscale {source} activation magnitude - {res} / {stem}")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            save_figure(fig, osp.join(vis_dir, f"magnitude_{source}_{res}_{stem}"))
            LOGGER.info("Rendered multiscale grids for %s / %s.", res, stem)

    LOGGER.info("Visualisation complete. Panels at %s", vis_dir)


# --------------------------------------------------------------------------- #
# Phase 4: composite figure recipes (rows = models, columns = isolated axis)
# --------------------------------------------------------------------------- #
def degrade_image(src_path: str, factor: int, out_path: str) -> str:
    """Simulate a coarser ground sampling distance by anti-aliased downsampling
    by `factor` and resampling back to the original pixel grid, holding extent
    and pixel dimensions constant. factor=1 copies the image unchanged."""
    from PIL import Image
    im = Image.open(src_path).convert("RGB")
    if factor <= 1:
        im.save(out_path); return out_path
    w, h = im.size
    small = im.resize((max(1, w // factor), max(1, h // factor)), Image.LANCZOS)
    im.close()
    small.resize((w, h), Image.BILINEAR).save(out_path)
    return out_path


def _resolve_columns(cfg: AnalysisConfig, spec: Dict) -> List[Tuple[str, str]]:
    if spec.get("columns"):
        return [(c["label"], c["tile"]) for c in spec["columns"]]
    if "source_tile" in spec and "degrade_factors" in spec:
        deg_dir = osp.join(cfg.out_dir, "figures", "composite", "_degraded")
        os.makedirs(deg_dir, exist_ok=True)
        stem = osp.splitext(osp.basename(spec["source_tile"]))[0]
        factors = spec["degrade_factors"]
        labels = spec.get("degrade_labels", [f"x{f}" for f in factors])
        cols = []
        for f, lab in zip(factors, labels):
            out = osp.join(deg_dir, f"{stem}_f{f}.png")
            degrade_image(spec["source_tile"], int(f), out)
            cols.append((lab, out))
        return cols
    raise ValueError(f"Composite spec '{spec.get('name')}' must define either "
                     f"'columns' or ('source_tile' and 'degrade_factors').")


def phase_composite(cfg: AnalysisConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    set_publication_style()

    if not cfg.composite_figures:
        LOGGER.warning("No composite_figures defined; nothing to do.")
        return
    comp_dir = osp.join(cfg.out_dir, "figures", "composite")
    os.makedirs(comp_dir, exist_ok=True)
    label2spec = {m.label: m for m in cfg.models}
    fams = {m.label: m.family for m in cfg.models}

    resolved: List[Dict] = []
    needs: Dict[str, Dict[Tuple[str, int, str], None]] = {}
    for spec in cfg.composite_figures:
        cols = _resolve_columns(cfg, spec)
        level = int(spec.get("level", 1))
        source = spec.get("source", cfg.vis_source)
        for ml in spec["models"]:
            if ml not in label2spec:
                raise ValueError(f"Composite model '{ml}' not found in cfg.models.")
            for _, tile in cols:
                needs.setdefault(ml, {})[(tile, level, source)] = None
        resolved.append({"spec": spec, "cols": cols, "level": level,
                         "source": source, "models": spec["models"]})

    capture: Dict[str, Dict[Tuple[str, int, str], np.ndarray]] = {}
    for ml, reqs in needs.items():
        m = label2spec[ml]
        ext = FeatureExtractor(m.config, m.checkpoint, cfg.device)
        try:
            for (tile, level, source) in reqs:
                try:
                    store = ext.extract_all(tile)
                except Exception as exc:
                    LOGGER.warning("Composite extraction failed (%s, %s): %s",
                                   ml, osp.basename(tile), exc)
                    continue
                feats = store[source]
                if level >= len(feats):
                    raise IndexError(f"{source} has {len(feats)} levels; requested {level}.")
                capture.setdefault(ml, {})[(tile, level, source)] = to_chw_numpy(feats[level])
        finally:
            ext.close()
        LOGGER.info("Composite features captured for %s.", ml)

    n_written = 0
    for item in resolved:
        spec, cols, level, source = item["spec"], item["cols"], item["level"], item["source"]
        mdl_labels = [ml for ml in item["models"] if ml in capture]
        if not mdl_labels:
            LOGGER.warning("Spec '%s' has no capturable models; skipped.", spec.get("name"))
            continue
        name = spec.get("name", "composite")
        ncol = len(cols)
        views = ["pcargb", "magnitude"] if spec.get("view", "both") == "both" \
            else [spec.get("view", "both")]
        for vw in views:
            fig, axes = plt.subplots(len(mdl_labels), ncol,
                                     figsize=(2.7 * ncol, 2.7 * len(mdl_labels)), squeeze=False)
            for ri, ml in enumerate(mdl_labels):
                col_maps = [capture[ml].get((tile, level, source)) for _, tile in cols]
                if vw == "pcargb":
                    basis, rownorm = _row_basis_and_norm(col_maps, source, cfg.seed)
                for ci, (clab, tile) in enumerate(cols):
                    ax = axes[ri][ci]; style_image_axis(ax)
                    fm = col_maps[ci]
                    if fm is None:
                        ax.text(0.5, 0.5, "n/a", ha="center", va="center")
                    elif vw == "pcargb":
                        ax.imshow(feature_to_pca_rgb(fm, basis=basis, norm=rownorm))
                    else:
                        rgb = _load_rgb(tile)
                        ax.imshow(rgb)
                        ax.imshow(_resize_to(activation_magnitude(fm), rgb.shape[:2]),
                                  cmap=OVERLAY_CMAP, alpha=0.5)
                    if ri == 0:
                        ax.set_title(clab)
                    if ci == 0:
                        ax.set_ylabel(f"{ml}\n({fams.get(ml,'?')})")
            fig.suptitle(f"{name} - {vw} - {source} {_level_tag(source, level)}")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            save_figure(fig, osp.join(comp_dir, f"{name}_{vw}"))
            n_written += 1
            LOGGER.info("Wrote composite figure: %s (%s)", name, vw)

    LOGGER.info("Composite figures complete (%d figures). Output at %s", n_written, comp_dir)
