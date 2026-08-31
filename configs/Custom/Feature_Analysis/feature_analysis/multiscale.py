# -*- coding: utf-8 -*-
"""Deep multiscale analysis of the unified (Stage C) model and the
"scale-anatomy" composite figure.

This module extends the aggregate analyses (CKA / Bures invariance /
dimensionality) with four *scale-resolved* quantities that explain **where in
the feature pyramid** the crown signal lives, how it re-allocates as ground
sampling distance (GSD) coarsens, and how the backbone families differ in the
way they populate the pyramid:

1. **Crown-background separability** (``deep``, CPU). For every
   (model, resolution, FPN level), cached feature locations are labelled
   crown / background by rasterising the COCO ground-truth masks onto the
   feature grid, and class separation is quantified by the shrinkage-
   regularised two-class Mahalanobis distance d' = sqrt((mu1-mu0)^T
   Sigma_w^-1 (mu1-mu0)). Locations with partial crown coverage
   (0.05 < coverage < 0.5) are excluded to avoid boundary label noise.
   This converts "the model responds to palms" into a per-scale number and
   identifies the pyramid level at which crown evidence is linearly
   accessible at each GSD.

2. **Level activation share** (``deep``, CPU). The per-location mean squared
   feature norm of each FPN level, normalised across levels, i.e. the
   fraction of per-location activation energy the network allocates to each
   scale. The per-location (rather than total) statistic removes the 4x
   spatial-count imbalance between adjacent levels, making the profile
   comparable across the pyramid. Shifts of this profile with GSD reveal
   whether a backbone re-allocates computation toward coarser levels as
   objects shrink -- or fails to, which is the representational mechanism of
   zero-shot degradation.

3. **Eigenspectrum decay** (``deep``, CPU). The eigenvalue spectrum of the
   pooled channel covariance per (model, resolution, level), and the fitted
   power-law decay exponent alpha (log-log slope over ranks 5-100, after
   Stringer et al., 2019). Slow decay (small alpha) indicates a
   high-dimensional, detail-rich code; fast decay a compressed one. Read
   jointly with the Bures invariance ranking, this exposes the
   invariance-expressivity trade-off between families.

4. **Inter-level representational similarity** (``deep``, CPU).
   Trace-normalised Bures fidelity between the channel covariances of every
   pair of FPN levels within one model. High off-diagonal similarity means a
   redundant pyramid (levels encode near-identical statistics); low values
   mean genuinely scale-specialised encodings. Bures is used because CKA
   requires location-matched samples, which do not exist across strides.

5. **Scale-anatomy figure** (``anatomy``, GPU). For one showcase tile and the
   ``vis_models``: rows = backbones, columns = [input + GT crown outlines,
   P2...P5 PCA-to-RGB with activation-magnitude contours, level activation
   share]. A single panel that shows *what* each family encodes at each
   scale, *where* it concentrates activation, and *how* it budgets the
   pyramid -- against ground truth.

All quantitative results carry bootstrap confidence intervals over tiles and
are written as CSV tables plus publication-grade PDF/PNG figures. The
detector is never modified; the ``deep`` phase is CPU-only and consumes the
``extract`` cache.
"""

from __future__ import annotations

import csv
import json
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import AnalysisConfig
from .extractor import FeatureExtractor, to_chw_numpy
from .metrics import bures_fidelity, channel_covariance
from .sampling import load_matrix, load_sampled
from .util import (FAMILY_COLORS, HEATMAP_CMAP, LOGGER, bootstrap_ci,
                   save_figure, set_publication_style, stable_seed,
                   style_image_axis)
from .visualization import (_level_tag, _load_rgb, _row_basis_and_norm,
                            activation_magnitude, feature_to_pca_rgb,
                            resolve_showcase, select_vis_models)

# Line-style cycle to distinguish models within a family (colour = family).
_STYLE_CYCLE = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
_MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Separability estimation constants.
_MIN_PER_CLASS = 20          # minimum crown/background locations per tile
_COVER_POS = 0.50            # coverage >= : crown
_COVER_NEG = 0.05            # coverage <= : background (in-between ignored)
_SHRINKAGE = 1e-2            # ridge fraction of mean variance for Sigma_w


# --------------------------------------------------------------------------- #
# Ground-truth rasterisation
# --------------------------------------------------------------------------- #
def load_coco_index(coco_ann: str) -> Dict[str, Dict]:
    """Map image stem -> {'size': (H, W), 'segs': [segmentation, ...]}."""
    with open(coco_ann, "r") as f:
        coco = json.load(f)
    by_id: Dict[int, Dict] = {}
    for im in coco["images"]:
        stem = osp.splitext(osp.basename(im["file_name"]))[0]
        by_id[im["id"]] = {"stem": stem, "size": (im["height"], im["width"]),
                           "segs": []}
    for ann in coco.get("annotations", []):
        rec = by_id.get(ann["image_id"])
        if rec is not None and ann.get("segmentation"):
            rec["segs"].append(ann["segmentation"])
    return {rec["stem"]: {"size": rec["size"], "segs": rec["segs"]}
            for rec in by_id.values()}


def rasterize_union(size: Tuple[int, int], segs: Sequence) -> np.ndarray:
    """Binary union mask (H, W) of all instance segmentations.

    Polygon segmentations are filled with PIL; RLE segmentations are decoded
    with pycocotools when available and skipped (with a warning) otherwise.
    """
    from PIL import Image, ImageDraw
    H, W = size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    rle_skipped = False
    for seg in segs:
        if isinstance(seg, list):
            for poly in seg:
                if len(poly) >= 6:
                    draw.polygon([float(v) for v in poly], fill=1)
        elif isinstance(seg, dict):
            try:
                from pycocotools import mask as mutils
                rle = mutils.frPyObjects(seg, H, W) if isinstance(
                    seg.get("counts"), list) else seg
                dec = mutils.decode(rle)
                if dec.ndim == 3:
                    dec = dec.max(axis=-1)
                arr = np.asarray(mask)
                arr = np.maximum(arr, dec.astype(arr.dtype))
                mask = Image.fromarray(arr)
                draw = ImageDraw.Draw(mask)
            except Exception:
                rle_skipped = True
    if rle_skipped:
        LOGGER.warning("pycocotools unavailable; RLE segmentations skipped.")
    return np.asarray(mask, dtype=np.float32)


def coverage_at(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Area-averaged crown coverage of `mask` on an (h, w) feature grid."""
    from PIL import Image
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.resize((hw[1], hw[0]), Image.BOX)
    return np.asarray(im, dtype=np.float32) / 255.0


def _feature_shape(meta: Dict) -> Optional[Tuple[int, int]]:
    """Feature-grid shape from the location index; falls back to a square
    grid for caches written before `shape` was recorded."""
    if "shape" in meta:
        h, w = meta["shape"]
        return int(h), int(w)
    hw = int(meta["hw"])
    side = int(round(np.sqrt(hw)))
    if side * side == hw:
        return side, side
    return None


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def mahalanobis_dprime(pos: np.ndarray, neg: np.ndarray,
                       shrinkage: float = _SHRINKAGE) -> float:
    """Two-class separability d' = sqrt((mu1-mu0)^T Sigma_w^-1 (mu1-mu0))
    with ridge shrinkage of the pooled within-class covariance."""
    if pos.shape[0] < _MIN_PER_CLASS or neg.shape[0] < _MIN_PER_CLASS:
        return float("nan")
    d = pos.shape[1]
    diff = pos.mean(0) - neg.mean(0)
    n1, n0 = pos.shape[0], neg.shape[0]
    sw = (channel_covariance(pos) * (n1 - 1)
          + channel_covariance(neg) * (n0 - 1)) / max(n1 + n0 - 2, 1)
    sw = sw + shrinkage * (np.trace(sw) / d) * np.eye(d)
    try:
        sol = np.linalg.solve(sw, diff)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(np.sqrt(max(diff @ sol, 0.0)))


def spectral_alpha(eigs: np.ndarray, kmin: int = 5, kmax: int = 100) -> float:
    """Power-law decay exponent of the eigenspectrum: -slope of
    log(lambda_k) vs log(k) over ranks [kmin, kmax]."""
    ev = np.sort(np.asarray(eigs))[::-1]
    ev = ev[ev > 1e-12]
    kmax = min(kmax, ev.size)
    if kmax - kmin < 10:
        return float("nan")
    k = np.arange(kmin, kmax + 1, dtype=float)
    y = np.log(ev[kmin - 1:kmax])
    slope = np.polyfit(np.log(k), y, 1)[0]
    return float(-slope)


# --------------------------------------------------------------------------- #
# Phase: deep (CPU, from cache)
# --------------------------------------------------------------------------- #
def _label_cache(cfg: AnalysisConfig, sampled: Dict[str, List[str]],
                 loc_meta: Dict[str, Dict]) -> Dict[Tuple[str, str, int], np.ndarray]:
    """Per (resolution, stem, level): ternary labels (1 crown / 0 background /
    -1 ignore) at the cached subsampled locations."""
    ann_by_set = {ts.name: ts.coco_ann for ts in cfg.tile_sets}
    labels: Dict[Tuple[str, str, int], np.ndarray] = {}
    for set_name, stems in sampled.items():
        ann_path = ann_by_set.get(set_name)
        if not ann_path or not osp.isfile(ann_path):
            LOGGER.warning("No COCO annotations for %s; separability skipped "
                           "for this set.", set_name)
            continue
        index = load_coco_index(ann_path)
        mask_cache: Dict[str, np.ndarray] = {}
        for stem in stems:
            rec = index.get(stem)
            if rec is None:
                continue
            for level in cfg.fpn_levels:
                meta = loc_meta.get(f"{set_name}/{stem}/L{level}")
                if meta is None:
                    continue
                shape = _feature_shape(meta)
                if shape is None:
                    LOGGER.warning("Non-square feature grid without recorded "
                                   "shape for %s/%s L%d; skipped.",
                                   set_name, stem, level)
                    continue
                if stem not in mask_cache:
                    mask_cache[stem] = rasterize_union(rec["size"], rec["segs"])
                cover = coverage_at(mask_cache[stem], shape).reshape(-1)
                lab = np.full(cover.shape, -1, dtype=np.int8)
                lab[cover >= _COVER_POS] = 1
                lab[cover <= _COVER_NEG] = 0
                idx = np.asarray(meta["idx"], dtype=int)
                labels[(set_name, stem, level)] = lab[idx]
    return labels


def analyze_separability(cfg: AnalysisConfig, sampled: Dict[str, List[str]],
                         labels: Dict[Tuple[str, str, int], np.ndarray]) -> List[Dict]:
    rows: List[Dict] = []
    for m in cfg.models:
        for set_name, stems in sampled.items():
            for level in cfg.fpn_levels:
                vals: List[float] = []
                for stem in stems:
                    lab = labels.get((set_name, stem, level))
                    if lab is None:
                        continue
                    mat = load_matrix(cfg, m.label, set_name, stem, level)
                    if mat is None or mat.shape[0] != lab.shape[0]:
                        continue
                    vals.append(mahalanobis_dprime(mat[lab == 1], mat[lab == 0]))
                mean, lo, hi = bootstrap_ci(
                    vals, cfg.bootstrap,
                    stable_seed(cfg.seed, m.label, set_name, level, "sep"))
                rows.append({"model": m.label, "family": m.family,
                             "resolution": set_name, "level": level,
                             "dprime": mean, "ci_lo": lo, "ci_hi": hi,
                             "n_tiles": int(np.sum(np.isfinite(vals)))})
    return rows


def analyze_level_energy(cfg: AnalysisConfig,
                         sampled: Dict[str, List[str]]) -> List[Dict]:
    rows: List[Dict] = []
    for m in cfg.models:
        for set_name, stems in sampled.items():
            per_level_shares: Dict[int, List[float]] = {l: [] for l in cfg.fpn_levels}
            for stem in stems:
                energies: Dict[int, float] = {}
                for level in cfg.fpn_levels:
                    mat = load_matrix(cfg, m.label, set_name, stem, level)
                    if mat is not None:
                        energies[level] = float(np.mean(np.sum(mat ** 2, axis=1)))
                total = sum(energies.values())
                if len(energies) == len(cfg.fpn_levels) and total > 0:
                    for level, e in energies.items():
                        per_level_shares[level].append(e / total)
            for level in cfg.fpn_levels:
                mean, lo, hi = bootstrap_ci(
                    per_level_shares[level], cfg.bootstrap,
                    stable_seed(cfg.seed, m.label, set_name, level, "energy"))
                rows.append({"model": m.label, "family": m.family,
                             "resolution": set_name, "level": level,
                             "share": mean, "ci_lo": lo, "ci_hi": hi})
    return rows


def analyze_spectra(cfg: AnalysisConfig, sampled: Dict[str, List[str]]
                    ) -> Tuple[List[Dict], Dict[Tuple[str, str, int], np.ndarray]]:
    """Pooled (tile-averaged, trace-normalised) eigenspectra and decay
    exponents; also returns the pooled covariances for the inter-level
    similarity analysis."""
    rows: List[Dict] = []
    pooled: Dict[Tuple[str, str, int], np.ndarray] = {}
    for m in cfg.models:
        for set_name, stems in sampled.items():
            for level in cfg.fpn_levels:
                covs = []
                for stem in stems:
                    mat = load_matrix(cfg, m.label, set_name, stem, level)
                    if mat is None:
                        continue
                    cov = channel_covariance(mat)
                    tr = np.trace(cov)
                    if tr > 0:
                        covs.append(cov / tr)
                if not covs:
                    continue
                cov = np.mean(covs, axis=0)
                pooled[(m.label, set_name, level)] = cov
                ev = np.linalg.eigvalsh(cov)[::-1]
                rows.append({"model": m.label, "family": m.family,
                             "resolution": set_name, "level": level,
                             "alpha": spectral_alpha(ev),
                             "eigs": ev})
    return rows, pooled


def analyze_interlevel(cfg: AnalysisConfig,
                       pooled: Dict[Tuple[str, str, int], np.ndarray]) -> List[Dict]:
    rows: List[Dict] = []
    for m in cfg.models:
        for ts in cfg.tile_sets:
            for i, la in enumerate(cfg.fpn_levels):
                for lb in cfg.fpn_levels[i + 1:]:
                    A = pooled.get((m.label, ts.name, la))
                    B = pooled.get((m.label, ts.name, lb))
                    if A is None or B is None:
                        continue
                    rows.append({"model": m.label, "family": m.family,
                                 "resolution": ts.name, "level_a": la,
                                 "level_b": lb,
                                 "bures_fidelity": bures_fidelity(A, B)})
    return rows


# --------------------------------------------------------------------------- #
# Deep-phase figures
# --------------------------------------------------------------------------- #
def _model_styles(cfg: AnalysisConfig) -> Dict[str, Dict]:
    """Colour = family; line style / marker cycles within a family."""
    per_family_count: Dict[str, int] = {}
    styles: Dict[str, Dict] = {}
    for m in cfg.models:
        k = per_family_count.get(m.family, 0)
        per_family_count[m.family] = k + 1
        styles[m.label] = {
            "color": FAMILY_COLORS.get(m.family, "#888888"),
            "ls": _STYLE_CYCLE[k % len(_STYLE_CYCLE)],
            "marker": _MARKER_CYCLE[k % len(_MARKER_CYCLE)],
        }
    return styles


def _profile_panels(cfg: AnalysisConfig, rows: List[Dict], value_key: str,
                    ylabel: str, title: str, out_base: str,
                    logy: bool = False) -> None:
    """Small-multiple line plot: one panel per resolution, x = FPN level,
    one line per model (family colour, per-model style), CI bands."""
    import matplotlib.pyplot as plt
    res_names = [ts.name for ts in cfg.tile_sets]
    styles = _model_styles(cfg)
    levels = cfg.fpn_levels
    fig, axes = plt.subplots(1, len(res_names),
                             figsize=(3.0 * len(res_names) + 0.8, 3.3),
                             sharey=True, squeeze=False)
    for pi, res in enumerate(res_names):
        ax = axes[0][pi]
        for m in cfg.models:
            pts = {r["level"]: r for r in rows
                   if r["model"] == m.label and r["resolution"] == res}
            xs = [l for l in levels if l in pts and np.isfinite(pts[l][value_key])]
            if not xs:
                continue
            ys = [pts[l][value_key] for l in xs]
            lo = [pts[l]["ci_lo"] for l in xs]
            hi = [pts[l]["ci_hi"] for l in xs]
            st = styles[m.label]
            ax.plot(xs, ys, label=m.label, color=st["color"], ls=st["ls"],
                    marker=st["marker"], markersize=3.5, lw=1.2)
            if all(np.isfinite(lo)) and all(np.isfinite(hi)):
                ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.12, lw=0)
        ax.set_title(res.replace("_", " "))
        ax.set_xticks(levels)
        ax.set_xticklabels([_level_tag("neck", l) for l in levels])
        if logy:
            ax.set_yscale("log")
        if pi == 0:
            ax.set_ylabel(ylabel)
    for pi in range(len(res_names)):
        handles, labs = axes[0][pi].get_legend_handles_labels()
        if handles:
            axes[0][pi].legend(handles, labs, fontsize=6, ncol=1, loc="best")
            break
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_base)


def _spectra_figure(cfg: AnalysisConfig, spec_rows: List[Dict],
                    fig_dir: str) -> None:
    import matplotlib.pyplot as plt
    styles = _model_styles(cfg)
    for ts in cfg.tile_sets:
        sub = [r for r in spec_rows if r["resolution"] == ts.name]
        if not sub:
            continue
        levels = sorted({r["level"] for r in sub})
        fig, axes = plt.subplots(1, len(levels),
                                 figsize=(2.9 * len(levels) + 0.6, 3.1),
                                 sharey=True, squeeze=False)
        for pi, level in enumerate(levels):
            ax = axes[0][pi]
            for r in sub:
                if r["level"] != level:
                    continue
                ev = r["eigs"]
                k = np.arange(1, ev.size + 1)
                st = styles[r["model"]]
                a = r["alpha"]
                lab = (f"{r['model']} ($\\alpha$={a:.2f})"
                       if np.isfinite(a) else r["model"])
                ax.loglog(k, np.clip(ev, 1e-12, None), label=lab,
                          color=st["color"], ls=st["ls"], lw=1.1)
            ax.set_title(_level_tag("neck", level))
            ax.set_xlabel("rank $k$")
            if pi == 0:
                ax.set_ylabel("normalised eigenvalue $\\lambda_k$")
            ax.legend(fontsize=5.5, loc="lower left")
        fig.suptitle(f"Channel-covariance eigenspectra - {ts.name.replace('_', ' ')}")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        save_figure(fig, osp.join(fig_dir, f"eigenspectra_{ts.name}"))


def _interlevel_figure(cfg: AnalysisConfig, il_rows: List[Dict],
                       fig_dir: str) -> None:
    import matplotlib.pyplot as plt
    try:
        models = select_vis_models(cfg)
    except ValueError:
        LOGGER.warning("vis_models matched no configured labels; the "
                       "inter-level figure falls back to the first three "
                       "models (the CSV covers all models).")
        models = cfg.models[:3]
    res_names = [ts.name for ts in cfg.tile_sets]
    levels = cfg.fpn_levels
    n = len(levels)
    if n < 2 or not models:
        return
    lut: Dict[Tuple[str, str], np.ndarray] = {}
    for r in il_rows:
        key = (r["model"], r["resolution"])
        M = lut.setdefault(key, np.full((n, n), np.nan))
        ia, ib = levels.index(r["level_a"]), levels.index(r["level_b"])
        M[ia, ib] = M[ib, ia] = r["bures_fidelity"]
    fig, axes = plt.subplots(len(models), len(res_names),
                             figsize=(1.9 * len(res_names) + 1.2,
                                      1.9 * len(models) + 0.9), squeeze=False)
    for ri, m in enumerate(models):
        for ci, res in enumerate(res_names):
            ax = axes[ri][ci]
            M = lut.get((m.label, res), np.full((n, n), np.nan)).copy()
            np.fill_diagonal(M, 1.0)
            ax.imshow(np.nan_to_num(M, nan=0.0), vmin=0, vmax=1,
                      cmap=HEATMAP_CMAP)
            for i in range(n):
                for j in range(n):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center",
                                va="center", fontsize=5,
                                color="white" if M[i, j] < 0.55 else "black")
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            tags = [_level_tag("neck", l) for l in levels]
            ax.set_xticklabels(tags, fontsize=6)
            ax.set_yticklabels(tags, fontsize=6)
            if ri == 0:
                ax.set_title(res.replace("_", " "), fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{m.label}\n({m.family})", fontsize=7)
    fig.suptitle("Inter-level representational similarity (Bures fidelity)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, osp.join(fig_dir, "interlevel_similarity"))


def _write_csv(path: str, rows: List[Dict], drop: Sequence[str] = ()) -> None:
    if not rows:
        return
    fields = [k for k in rows[0].keys() if k not in drop]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _separability_summary(cfg: AnalysisConfig, sep_rows: List[Dict]) -> List[Dict]:
    """Per (model, resolution): the pyramid level with maximal d' and its
    value -- the quantity the manuscript reports as the 'peak evidence
    level' and its shift with GSD."""
    out: List[Dict] = []
    for m in cfg.models:
        for ts in cfg.tile_sets:
            cand = [r for r in sep_rows
                    if r["model"] == m.label and r["resolution"] == ts.name
                    and np.isfinite(r["dprime"])]
            if not cand:
                continue
            best = max(cand, key=lambda r: r["dprime"])
            out.append({"model": m.label, "family": m.family,
                        "resolution": ts.name,
                        "peak_level": best["level"],
                        "peak_tag": _level_tag("neck", best["level"]),
                        "peak_dprime": best["dprime"],
                        "ci_lo": best["ci_lo"], "ci_hi": best["ci_hi"]})
    return out


def phase_deep(cfg: AnalysisConfig) -> None:
    """CPU-only deep multiscale analysis over the extraction cache."""
    import matplotlib
    matplotlib.use("Agg")
    set_publication_style()

    fig_dir = osp.join(cfg.out_dir, "figures")
    tab_dir = osp.join(cfg.out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    for req, desc in ((osp.join(cfg.cache_dir, "_sampled_tiles.json"),
                       "sampled-tile manifest"),
                      (osp.join(cfg.cache_dir, "_locindex.json"),
                       "location index")):
        if not osp.isfile(req):
            raise FileNotFoundError(
                f"{desc.capitalize()} not found at {req}. The 'deep' phase "
                f"consumes the extraction cache; run the 'extract' phase first.")
    sampled = load_sampled(cfg)
    with open(osp.join(cfg.cache_dir, "_locindex.json"), "r") as f:
        loc_meta = json.load(f)

    LOGGER.info("Deep analysis 1/4: crown-background separability ...")
    labels = _label_cache(cfg, sampled, loc_meta)
    sep_rows = analyze_separability(cfg, sampled, labels)
    _write_csv(osp.join(tab_dir, "separability.csv"), sep_rows)
    _write_csv(osp.join(tab_dir, "separability_summary.csv"),
               _separability_summary(cfg, sep_rows))
    if any(np.isfinite(r["dprime"]) for r in sep_rows):
        _profile_panels(cfg, sep_rows,
                        "dprime", "crown-background $d'$",
                        "Linear crown-background separability by pyramid level",
                        osp.join(fig_dir, "separability_by_level"))

    LOGGER.info("Deep analysis 2/4: level activation share ...")
    en_rows = analyze_level_energy(cfg, sampled)
    _write_csv(osp.join(tab_dir, "level_energy.csv"), en_rows)
    _profile_panels(cfg, en_rows, "share", "activation share",
                    "Pyramid activation allocation by resolution",
                    osp.join(fig_dir, "level_energy_profile"))

    LOGGER.info("Deep analysis 3/4: eigenspectrum decay ...")
    spec_rows, pooled = analyze_spectra(cfg, sampled)
    _write_csv(osp.join(tab_dir, "spectral_decay.csv"), spec_rows, drop=("eigs",))
    _spectra_figure(cfg, spec_rows, fig_dir)

    LOGGER.info("Deep analysis 4/4: inter-level similarity ...")
    il_rows = analyze_interlevel(cfg, pooled)
    _write_csv(osp.join(tab_dir, "interlevel_bures.csv"), il_rows)
    _interlevel_figure(cfg, il_rows, fig_dir)

    import time
    from . import __version__
    meta = {"tool": "feature_analysis.deep", "version": __version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": cfg.seed, "bootstrap": cfg.bootstrap,
            "config_hash": cfg.config_hash(),
            "separability_thresholds": {"pos": _COVER_POS, "neg": _COVER_NEG,
                                        "min_per_class": _MIN_PER_CLASS,
                                        "shrinkage": _SHRINKAGE}}
    for mod in ("torch", "mmdet"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:
            pass
    with open(osp.join(cfg.out_dir, "deep_provenance.json"), "w") as f:
        json.dump(meta, f, indent=2)

    LOGGER.info("Deep analysis complete. Outputs at %s", cfg.out_dir)


# --------------------------------------------------------------------------- #
# Phase: anatomy (GPU) -- the multiscale hero figure
# --------------------------------------------------------------------------- #
def _draw_gt_outlines(ax, segs: Sequence, color: str = "#FFFFFF") -> None:
    for seg in segs:
        if not isinstance(seg, list):
            continue
        for poly in seg:
            if len(poly) < 6:
                continue
            xs = list(poly[0::2]) + [poly[0]]
            ys = list(poly[1::2]) + [poly[1]]
            ax.plot(xs, ys, color="black", lw=1.0, alpha=0.7, zorder=3)
            ax.plot(xs, ys, color=color, lw=0.5, alpha=0.95, zorder=4)


def phase_anatomy(cfg: AnalysisConfig) -> None:
    """Render the scale-anatomy composite: rows = vis_models; columns =
    [input + GT outlines, one PCA-to-RGB panel per FPN level with magnitude
    contours, level activation share]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    set_publication_style()

    out_dir = osp.join(cfg.out_dir, "figures", "anatomy")
    os.makedirs(out_dir, exist_ok=True)
    levels = cfg.vis_levels if cfg.vis_levels is not None else cfg.fpn_levels
    source = cfg.vis_source
    fam_order = {"CNN": 0, "Transformer": 1, "Mamba": 2}
    models = sorted(select_vis_models(cfg),
                    key=lambda m: (fam_order.get(m.family, 9), m.label))
    showcase = resolve_showcase(cfg)
    ann_by_set = {ts.name: ts.coco_ann for ts in cfg.tile_sets}

    # ---- capture (one detector resident at a time) ------------------------ #
    capture: Dict[Tuple[str, str], Dict[str, Dict[int, np.ndarray]]] = {}
    for m in models:
        ext = FeatureExtractor(m.config, m.checkpoint, cfg.device)
        try:
            for res, tiles in showcase.items():
                for tile in tiles:
                    stem = osp.splitext(osp.basename(tile))[0]
                    try:
                        store = ext.extract_all(tile)
                    except Exception as exc:
                        LOGGER.warning("Anatomy extraction failed (%s, %s): %s",
                                       m.label, stem, exc)
                        continue
                    feats = store[source]
                    per_level = {}
                    for lvl in levels:
                        if lvl >= len(feats):
                            raise IndexError(f"{source} has {len(feats)} levels;"
                                             f" requested {lvl}.")
                        per_level[lvl] = to_chw_numpy(feats[lvl])
                    capture.setdefault((res, tile), {})[m.label] = per_level
        finally:
            ext.close()
        LOGGER.info("Anatomy features captured for %s.", m.label)

    # ---- render ------------------------------------------------------------ #
    for (res, tile), by_model in capture.items():
        stem = osp.splitext(osp.basename(tile))[0]
        mdl_labels = [m.label for m in models if m.label in by_model]
        fams = {m.label: m.family for m in models}
        if not mdl_labels:
            continue
        input_rgb = _load_rgb(tile)

        segs: Sequence = []
        ann_path = ann_by_set.get(res)
        if ann_path and osp.isfile(ann_path):
            rec = load_coco_index(ann_path).get(stem)
            if rec is not None:
                segs = rec["segs"]

        ncol = 1 + len(levels) + 1
        fig, axes = plt.subplots(len(mdl_labels), ncol,
                                 figsize=(2.55 * ncol, 2.55 * len(mdl_labels)),
                                 squeeze=False)
        for ri, lab in enumerate(mdl_labels):
            lvl_maps = [by_model[lab][lvl] for lvl in levels]
            basis, rownorm = _row_basis_and_norm(lvl_maps, source, cfg.seed)

            # column 0: input + GT outlines
            ax = axes[ri][0]
            ax.imshow(input_rgb)
            _draw_gt_outlines(ax, segs)
            ax.set_xlim(0, input_rgb.shape[1]); ax.set_ylim(input_rgb.shape[0], 0)
            style_image_axis(ax)
            if ri == 0:
                ax.set_title("input + GT crowns")
            ax.set_ylabel(f"{lab}\n({fams[lab]})")

            # feature columns: PCA-to-RGB with magnitude contours
            energies = []
            for ci, lvl in enumerate(levels, start=1):
                fm = by_model[lab][lvl]
                energies.append(float(np.mean(np.sum(fm ** 2, axis=0))))
                ax = axes[ri][ci]
                ax.imshow(feature_to_pca_rgb(fm, basis=basis, norm=rownorm))
                mag = activation_magnitude(fm)
                ax.contour(mag, levels=[0.6, 0.85], colors="white",
                           linewidths=0.45, alpha=0.9)
                style_image_axis(ax)
                if ri == 0:
                    h, w = fm.shape[1:]
                    ax.set_title(f"{_level_tag(source, lvl)}\n{h}x{w}")

            # last column: level activation share
            ax = axes[ri][-1]
            total = sum(energies)
            shares = [e / total if total > 0 else 0.0 for e in energies]
            ypos = np.arange(len(levels))[::-1]
            ax.barh(ypos, shares, color=FAMILY_COLORS.get(fams[lab], "#888888"),
                    edgecolor="black", linewidth=0.4, height=0.65)
            ax.set_yticks(ypos)
            ax.set_yticklabels([_level_tag(source, l) for l in levels], fontsize=7)
            ax.set_xlim(0, max(0.05, max(shares) * 1.25))
            for y, s in zip(ypos, shares):
                ax.annotate(f"{s:.2f}", (s, y), xytext=(2, 0),
                            textcoords="offset points", va="center", fontsize=6)
            ax.tick_params(axis="x", labelsize=6)
            if ri == 0:
                ax.set_title("activation share")

        fig.suptitle(f"Scale anatomy - {source} - {res} / {stem}")
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        save_figure(fig, osp.join(out_dir, f"anatomy_{source}_{res}_{stem}"))
        LOGGER.info("Rendered scale-anatomy figure for %s / %s.", res, stem)

    LOGGER.info("Anatomy figures complete. Output at %s", out_dir)
