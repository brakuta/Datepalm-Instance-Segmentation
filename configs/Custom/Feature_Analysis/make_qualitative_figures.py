#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-grade qualitative feature figures with explicit tile control.

Two figure types, complementing the quantitative results:

1. ``channels`` -- the classic feature-map mosaic (rows = network depths,
   columns = individual channels, viridis), as in prior remote-sensing
   publications, with one decisive improvement: channels are RANKED BY
   CROWN SELECTIVITY (mean activation inside ground-truth crowns minus
   outside, in units of the map's standard deviation) rather than chosen
   arbitrarily. The figure illustrates hierarchical abstraction: textural
   responses at fine levels giving way to compact crown-level responses at
   coarse levels. State in the caption: "the K most crown-selective
   channels per level are shown."

2. ``ladder`` -- the multiscale x GSD magnitude matrix for ONE model:
   rows = input + pyramid levels P2..P5 (activation-magnitude overlays),
   columns = user-chosen tiles (e.g. UAV 5 cm, 15 cm sim, 30 cm sim,
   WV-3 30 cm real). This is the visual counterpart of the quantitative
   sensor-vs-resolution result: response patterns remain locked to crowns
   along the simulated-GSD columns and remain crown-localised on real
   WV-3 while the response statistics visibly shift.

Rendering improvements over the composite phase: bilinear upsampling of
small feature grids (no blocky nearest-neighbour artefacts on 30 cm tiles),
2-98 percentile contrast normalisation per panel, and an input reference row.

Usage (from the mmdetection root; one GPU model resident at a time):

  # A. previous-paper-style mosaic, best model, your chosen tile
  python configs/Custom/Feature_Analysis/make_qualitative_figures.py channels \\
      --config configs/Custom/Feature_Analysis/config_run03.json \\
      --model SpatialMamba-S --source neck --topk 8 \\
      --tile /workspace/datasets/COCO/UAV_5cm/test_UAV/JPEGImages/000000000003.jpg \\
      --out  /root/work_dirs/feature_analysis/manuscript_figs

  # B. multiscale x GSD ladder, your chosen same-scene tiles + a WV-3 tile
  python configs/Custom/Feature_Analysis/make_qualitative_figures.py ladder \\
      --config configs/Custom/Feature_Analysis/config_run03.json \\
      --model SpatialMamba-S \\
      --tiles UAV5=/.../test_UAV/JPEGImages/000000000003.jpg \\
              UAV15=/.../test_UAV_15cm/JPEGImages/000000000003.jpg \\
              UAV30=/.../test_UAV_30cm/JPEGImages/000000000003.jpg \\
              WV3=/.../test_sat/JPEGImages/WV3_Ajman_test_tile_000056.jpg \\
      --labels "UAV 5 cm (native)" "UAV 15 cm (sim.)" "UAV 30 cm (sim.)" \\
               "WV-3 30 cm (real)" \\
      --out /root/work_dirs/feature_analysis/manuscript_figs

Tiles are ALWAYS given explicitly -- you choose them visually in Explorer
and paste the paths; nothing is auto-selected.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = osp.dirname(osp.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from feature_analysis.config import AnalysisConfig  # noqa: E402
from feature_analysis.extractor import FeatureExtractor, to_chw_numpy  # noqa: E402
from feature_analysis.multiscale import (coverage_at, load_coco_index,  # noqa: E402
                                         rasterize_union)
from feature_analysis.util import (LOGGER, configure_logging,  # noqa: E402
                                   save_figure, set_publication_style,
                                   style_image_axis)
from feature_analysis.visualization import _level_tag, _load_rgb  # noqa: E402

DISPLAY = 512  # panel display size in pixels


# --------------------------------------------------------------------------- #
def _resize(arr: np.ndarray, size: int = DISPLAY) -> np.ndarray:
    """Bilinear resize of a 2-D float map to (size, size)."""
    from PIL import Image
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    norm = (arr - lo) / (hi - lo + 1e-12)
    im = Image.fromarray((norm * 255).astype(np.uint8))
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _stretch(m: np.ndarray, p_lo: float = 2, p_hi: float = 98) -> np.ndarray:
    lo, hi = np.percentile(m, [p_lo, p_hi])
    return np.clip((m - lo) / (hi - lo + 1e-12), 0, 1)


def _rgb_resized(path: str, size: int = DISPLAY) -> np.ndarray:
    from PIL import Image
    arr = _load_rgb(path)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
    return np.asarray(im)


def _find_ann(cfg: AnalysisConfig, tile: str) -> Optional[str]:
    """COCO annotation of the tile set whose img_dir best matches the tile."""
    parts = set(p for p in tile.split("/") if p)
    best, score = None, 0
    for ts in cfg.tile_sets:
        s = len(parts & set(p for p in ts.img_dir.split("/") if p))
        if s > score and ts.coco_ann and osp.isfile(ts.coco_ann):
            best, score = ts.coco_ann, s
    return best


def _crown_coverage(cfg: AnalysisConfig, tile: str,
                    hw: Tuple[int, int]) -> Optional[np.ndarray]:
    ann = _find_ann(cfg, tile)
    if ann is None:
        return None
    stem = osp.splitext(osp.basename(tile))[0]
    rec = load_coco_index(ann).get(stem)
    if rec is None or not rec["segs"]:
        return None
    return coverage_at(rasterize_union(rec["size"], rec["segs"]), hw)


def _gt_polys(cfg: AnalysisConfig, tile: str):
    """Ground-truth polygon list and native (H, W) for a tile, or (None, None)."""
    ann = _find_ann(cfg, tile)
    if ann is None:
        return None, None
    rec = load_coco_index(ann).get(osp.splitext(osp.basename(tile))[0])
    if rec is None or not rec["segs"]:
        return None, None
    return rec["segs"], rec["size"]


def _draw_outlines(ax, segs, native_hw, size: int = DISPLAY,
                   color: str = "#FFFF00") -> None:
    """Draw GT crown outlines scaled from native pixels to the display grid."""
    H, W = native_hw
    sx, sy = size / W, size / H
    for seg in segs:
        if not isinstance(seg, list):
            continue
        for poly in seg:
            if len(poly) < 6:
                continue
            xs = [v * sx for v in poly[0::2]] + [poly[0] * sx]
            ys = [v * sy for v in poly[1::2]] + [poly[1] * sy]
            ax.plot(xs, ys, color="black", lw=0.9, alpha=0.65, zorder=3)
            ax.plot(xs, ys, color=color, lw=0.45, alpha=0.95, zorder=4)


def _crown_ratio(mag, cover):
    """Mean activation magnitude inside crowns / outside (scale-free)."""
    if cover is None:
        return None
    inside, outside = cover >= 0.5, cover <= 0.05
    if inside.sum() < 4 or outside.sum() < 4:
        return None
    denom = float(mag[outside].mean())
    return float(mag[inside].mean()) / denom if denom > 0 else None


def _crown_auc(resp, cover):
    """AUC of the response for crown-vs-background discrimination: the
    probability that a random crown location has a higher response than a
    random background location. Bounded in [0, 1]; 0.5 = chance."""
    if cover is None:
        return None
    r = resp.reshape(-1)
    inside = cover.reshape(-1) >= 0.5
    outside = cover.reshape(-1) <= 0.05
    n1, n0 = int(inside.sum()), int(outside.sum())
    if n1 < 4 or n0 < 4:
        return None
    vals = np.concatenate([r[inside], r[outside]])
    ranks = np.argsort(np.argsort(vals)).astype(np.float64) + 1
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _gt_row_image(cfg, tile, rgb):
    """Dimmed image with the GT crown mask filled in yellow -- a dedicated
    ground-truth row the responses can be read against."""
    segs, native = _gt_polys(cfg, tile)
    if segs is None:
        return None
    mask = rasterize_union(native, segs)
    from PIL import Image
    m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                   .resize((DISPLAY, DISPLAY), Image.BILINEAR), float) / 255.
    out = (rgb.astype(float) * 0.45)
    yellow = np.array([255, 220, 0], float)
    out = out * (1 - 0.75 * m[..., None]) + yellow * (0.75 * m[..., None])
    return out.astype(np.uint8)


def _overlay_heat(ax, rgb, mag01, max_alpha: float = 0.85) -> None:
    """Alpha-ramped heat overlay: activation controls opacity, so weakly
    activated regions keep the underlying imagery visible."""
    import matplotlib
    rgba = matplotlib.colormaps["inferno"](mag01)
    rgba[..., 3] = np.clip(mag01 ** 1.2, 0, 1) * max_alpha
    ax.imshow(rgb)
    ax.imshow(rgba)


def _channel_weights(fm, cover):
    """Per-channel crown-selectivity weights from one reference tile/level.

    score_c = (mean activation inside GT crowns - mean outside) / std_c,
    clipped to non-negative and L1-normalised. Falls back to uniform
    weights (equivalent to plain magnitude) if no ground truth is found.
    """
    c = fm.shape[0]
    flat = fm.reshape(c, -1)
    if cover is None:
        return np.ones(c) / c
    inside = cover.reshape(-1) >= 0.5
    outside = cover.reshape(-1) <= 0.05
    if not inside.any() or not outside.any():
        return np.ones(c) / c
    sd = flat.std(axis=1) + 1e-6
    score = (flat[:, inside].mean(1) - flat[:, outside].mean(1)) / sd
    score = np.clip(score, 0, None)
    total = score.sum()
    return score / total if total > 0 else np.ones(c) / c


def _selective_response(fm, weights):
    """Weighted channel combination emphasising crown-selective channels,
    instead of the L2 norm over all channels (diluted by the majority of
    channels that respond to non-crown structure)."""
    return np.tensordot(weights, fm, axes=(0, 0))


def _model_spec(cfg: AnalysisConfig, label: str):
    for m in cfg.models:
        if m.label == label:
            return m
    raise ValueError(f"Model '{label}' not in config; available: "
                     f"{[m.label for m in cfg.models]}")


def _capture(cfg: AnalysisConfig, label: str, tiles: Sequence[str],
             source: str) -> Dict[str, List[np.ndarray]]:
    spec = _model_spec(cfg, label)
    ext = FeatureExtractor(spec.config, spec.checkpoint, cfg.device)
    out: Dict[str, List[np.ndarray]] = {}
    try:
        for t in tiles:
            feats = ext.extract_all(t)[source]
            out[t] = [to_chw_numpy(f) for f in feats]
            LOGGER.info("Captured %s: %s (%d levels).",
                        label, osp.basename(t), len(out[t]))
    finally:
        ext.close()
    return out


# --------------------------------------------------------------------------- #
# Figure 1: channel mosaic (previous-paper style, selectivity-ranked)
# --------------------------------------------------------------------------- #
def _select_channels(fm, cover):
    """Top-channel ordering by crown selectivity (falls back to variance)."""
    c = fm.shape[0]
    flat = fm.reshape(c, -1)
    if cover is not None:
        inside = cover.reshape(-1) >= 0.5
        outside = cover.reshape(-1) <= 0.05
        if inside.any() and outside.any():
            sd = flat.std(axis=1) + 1e-6
            score = (flat[:, inside].mean(1) - flat[:, outside].mean(1)) / sd
            return score
    return flat.var(axis=1)


def _channel_block(cfg, fig, gs0, model_label, tile, source, levels, topk,
                   panel_tag, block_title=None, transpose=False,
                   cmap="viridis") -> None:
    """One block: input panel + label spacer + (levels x topk) channel grid.

    A dedicated narrow spacer column carries the level tags (C2..C5 / P2..P5)
    so they can never collide with the input thumbnail.
    transpose=True puts levels as columns (compact) with tags as titles.
    """
    maps = _capture(cfg, model_label, [tile], source)[tile]
    if max(levels) >= len(maps):
        raise IndexError(f"{source} has {len(maps)} levels.")
    segs, native = _gt_polys(cfg, tile)

    ninner = len(levels) if transpose else topk
    nrows_in = topk if transpose else len(levels)
    widths = ([1.25, 0.02] if transpose else [1.25, 0.16]) + [1] * ninner
    gs = gs0.subgridspec(nrows_in, ninner + 2, wspace=0.03, hspace=0.03,
                         width_ratios=widths)

    ax_in = fig.add_subplot(gs[:, 0])
    ax_in.imshow(_rgb_resized(tile))
    if segs is not None:
        _draw_outlines(ax_in, segs, native)
        ax_in.set_xlim(0, DISPLAY); ax_in.set_ylim(DISPLAY, 0)
    style_image_axis(ax_in)
    tag = f"({panel_tag}) " if panel_tag else ""
    head = block_title if block_title is not None else model_label
    ax_in.set_ylabel(f"{tag}{head}", fontsize=9, labelpad=4)

    per_level = {}
    for lvl in levels:
        fm = maps[lvl]
        cover = _crown_coverage(cfg, tile, fm.shape[1:])
        score = _select_channels(fm, cover)
        per_level[lvl] = (fm, score, np.argsort(score)[::-1][:topk])

    for li, lvl in enumerate(levels):
        fm, score, top = per_level[lvl]
        if not transpose:
            ax_lab = fig.add_subplot(gs[li, 1])
            ax_lab.axis("off")
            ax_lab.text(0.5, 0.5, _level_tag(source, lvl), rotation=90,
                        ha="center", va="center", fontsize=8,
                        transform=ax_lab.transAxes)
        for ki, ch in enumerate(top):
            r, cpos = (ki, li) if transpose else (li, ki)
            ax = fig.add_subplot(gs[r, cpos + 2])
            ax.imshow(_stretch(_resize(fm[ch])), cmap=cmap)
            style_image_axis(ax)
            ax.annotate(f"{ch}", (0.04, 0.96), xycoords="axes fraction",
                        va="top", fontsize=5, color="white",
                        bbox=dict(fc="black", alpha=0.4, pad=0.8, lw=0))
            if transpose and ki == 0:
                ax.set_title(_level_tag(source, lvl), fontsize=7)


def fig_channels(cfg: AnalysisConfig, args) -> None:
    import matplotlib.pyplot as plt
    levels, topk = args.levels, args.topk

    model_list = args.model
    if len(model_list) == 1 and model_list[0].lower() == "all":
        model_list = [m.label for m in cfg.models]

    if getattr(args, "tiles", None):
        # one figure PER MODEL, each stacking the given tiles as (a)/(b)/(c)
        tile_entries = [(kv.split("=", 1)[0], kv.split("=", 1)[1])
                        for kv in args.tiles]
        labels = args.labels if args.labels else [n for n, _ in tile_entries]
        if len(labels) != len(tile_entries):
            raise ValueError("--labels count must match --tiles count.")
        for mdl in model_list:
            _render_channels_figure(
                cfg, args, mdl,
                [(mdl, t, lab) for (_, t), lab in zip(tile_entries, labels)],
                stub=f"{mdl}_multisensor")
        return

    # single tile, one or more models stacked in ONE figure
    entries = [(m, args.tile, m) for m in model_list]
    _render_channels_figure(cfg, args, None, entries,
                            stub="_".join(model_list) if len(model_list) <= 3
                            else f"{len(model_list)}models")


def _render_channels_figure(cfg, args, mdl_for_title, entries, stub) -> None:
    import matplotlib.pyplot as plt
    levels, topk = args.levels, args.topk
    nblk = len(entries)
    rows_per = (topk if args.transpose else len(levels))
    cols_per = (len(levels) if args.transpose else topk)
    unit = 1.05
    fig = plt.figure(figsize=(unit * (cols_per + 1.5),
                              unit * rows_per * nblk + 0.28 * nblk))
    outer = fig.add_gridspec(nblk, 1, hspace=0.035,
                             top=0.965, bottom=0.01, left=0.055, right=0.99)
    tags = list("abcdefgh")
    for bi, (mdl, tile, title) in enumerate(entries):
        _channel_block(cfg, fig, outer[bi], mdl, tile, args.source, levels,
                       topk, tags[bi] if nblk > 1 else None,
                       block_title=title, transpose=args.transpose,
                       cmap=args.cmap)
    head = (f"{mdl_for_title} - " if mdl_for_title else "")
    fig.suptitle(f"{head}Top-{topk} crown-selective channels per level",
                 fontsize=10, y=0.998)
    base = osp.join(args.out, f"channels_{stub}_{args.source}"
                    + ("_T" if args.transpose else ""))
    os.makedirs(args.out, exist_ok=True)
    save_figure(fig, base)
    LOGGER.info("Wrote %s.{pdf,png}", base)


# --------------------------------------------------------------------------- #
def _ladder_block(cfg, axes, mdl, tiles, labels, levels, metric,
                  weight_tile_idx, row_offset=0, panel_tag=None,
                  style="pure", cmap="viridis"):
    """Rows: Image, Ground truth, then one response row per pyramid level.

    style='pure' (default): response maps rendered directly in `cmap`
    (consistent with the channel-mosaic figure); style='overlay': alpha-ramp
    heat overlay on the imagery. Each response panel is annotated with the
    crown-vs-background AUC (0.5 = chance, 1.0 = perfect ranking).
    """
    caps = _capture(cfg, mdl, tiles, "neck")

    weights_by_level = None
    if metric == "selective":
        ref_tile = tiles[weight_tile_idx]
        ref_maps = caps[ref_tile]
        weights_by_level = {}
        for lvl in levels:
            fm = ref_maps[lvl]
            cover = _crown_coverage(cfg, ref_tile, fm.shape[1:])
            weights_by_level[lvl] = _channel_weights(fm, cover)

    for ci, (t, lab) in enumerate(zip(tiles, labels)):
        rgb = _rgb_resized(t)
        # row 0: image
        ax = axes[row_offset][ci]
        ax.imshow(rgb)
        style_image_axis(ax)
        if row_offset == 0:
            ax.set_title(lab, fontsize=9)
        if ci == 0:
            lbl = "Image"
            if panel_tag:
                lbl = f"({panel_tag}) {mdl}\n{lbl}"
            ax.set_ylabel(lbl, fontsize=9)
        # row 1: ground truth
        ax = axes[row_offset + 1][ci]
        gt_img = _gt_row_image(cfg, t, rgb)
        if gt_img is not None:
            ax.imshow(gt_img)
        else:
            ax.imshow(rgb)
            ax.annotate("no GT", (0.5, 0.5), xycoords="axes fraction",
                        ha="center", fontsize=8, color="white",
                        bbox=dict(fc="black", alpha=0.5, pad=2, lw=0))
        style_image_axis(ax)
        if ci == 0:
            ax.set_ylabel("Ground truth", fontsize=9)
        # response rows
        for ri, lvl in enumerate(levels, start=2):
            fm = caps[t][lvl]
            if metric == "selective":
                resp = _selective_response(fm, weights_by_level[lvl])
            else:
                resp = np.linalg.norm(fm, axis=0)
            cover = _crown_coverage(cfg, t, resp.shape)
            auc = _crown_auc(resp, cover)
            disp = _stretch(_resize(resp))
            ax = axes[row_offset + ri][ci]
            if style == "overlay":
                _overlay_heat(ax, rgb, disp)
            else:
                ax.imshow(disp, cmap=cmap)
            style_image_axis(ax)
            if auc is not None:
                ax.annotate(f"AUC = {auc:.2f}", (0.03, 0.97),
                            xycoords="axes fraction", va="top",
                            fontsize=6.5, color="white",
                            bbox=dict(fc="black", alpha=0.55, pad=1.5, lw=0))
            if ci == 0:
                h, w = fm.shape[1:]
                tag = "" if panel_tag is None else f"({panel_tag}) "
                level_tag = _level_tag("neck", lvl)
                ax.set_ylabel(f"{tag}{level_tag}\n{h}x{w}", fontsize=8)


def fig_ladder(cfg: AnalysisConfig, args) -> None:
    import matplotlib.pyplot as plt
    tiles = [kv.split("=", 1)[1] for kv in args.tiles]
    labels = args.labels if args.labels else \
        [kv.split("=", 1)[0] for kv in args.tiles]
    if len(labels) != len(tiles):
        raise ValueError("--labels count must match --tiles count.")
    levels = args.levels
    metric = args.metric
    wti = args.weight_tile if args.weight_tile is not None else 0
    if not (0 <= wti < len(tiles)):
        raise ValueError(f"--weight-tile index {wti} out of range for "
                         f"{len(tiles)} tiles.")
    rows_per = 2 + len(levels)
    metric_txt = ("crown-selectivity-weighted response" if metric == "selective"
                 else "activation magnitude")

    model_list = args.model
    if len(model_list) == 1 and model_list[0].lower() == "all":
        model_list = [m.label for m in cfg.models]

    if args.combine and len(model_list) > 1:
        nblk = len(model_list)
        nrow, ncol = nblk * rows_per, len(tiles)
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(2.15 * ncol, 2.05 * nrow), squeeze=False)
        tags = list("abcdefgh")
        for bi, mdl in enumerate(model_list):
            _ladder_block(cfg, axes, mdl, tiles, labels, levels, metric, wti,
                         row_offset=bi * rows_per, panel_tag=tags[bi],
                         style=args.style, cmap=args.cmap)
        fig.suptitle(f"{metric_txt.capitalize()} across pyramid levels, "
                     f"sensors, and architecture", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        base = osp.join(args.out, f"ladder_{'_'.join(model_list)}_combined")
        os.makedirs(args.out, exist_ok=True)
        save_figure(fig, base)
        LOGGER.info("Wrote %s.{pdf,png}", base)
        return

    for mdl in model_list:
        nrow, ncol = rows_per, len(tiles)
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(2.15 * ncol, 2.15 * nrow), squeeze=False)
        _ladder_block(cfg, axes, mdl, tiles, labels, levels, metric, wti,
                      style=args.style, cmap=args.cmap)
        fig.suptitle(f"{mdl} - {metric_txt} across pyramid levels and "
                     f"sensors / ground sampling distances", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        base = osp.join(args.out, f"ladder_{mdl}_{metric}_{args.style}")
        os.makedirs(args.out, exist_ok=True)
        save_figure(fig, base)
        LOGGER.info("Wrote %s.{pdf,png}", base)

# --------------------------------------------------------------------------- #
def _gt_mask_display(cfg, tile):
    """Binary GT crown mask resized to the display grid, or None."""
    segs, native = _gt_polys(cfg, tile)
    if segs is None:
        return None
    mask = rasterize_union(native, segs)
    from PIL import Image
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.resize((DISPLAY, DISPLAY), Image.NEAREST)
    return np.asarray(im, dtype=np.float32) / 255.0


def _concentration(resp, cover):
    """Bounded response-concentration statistic: the fraction of total
    positive response mass falling inside GT crowns, reported against the
    crown area fraction. Unlike a mean ratio, it cannot explode when the
    background response approaches zero."""
    if cover is None:
        return None, None
    pos = np.clip(resp, 0, None)
    total = pos.sum()
    if total <= 0:
        return None, None
    inside = cover >= 0.5
    return float(pos[inside].sum() / total), float(inside.mean())


def fig_sensors(cfg: AnalysisConfig, args) -> None:
    """Rows = sensors/tiles; columns = [input, GT mask, one response panel
    per pyramid level]. Responses use the crown-selectivity weighting (fit
    on --weight-tile) and are rendered as standalone heatmaps for maximum
    contrast, not as overlays."""
    import matplotlib.pyplot as plt
    tiles = [kv.split("=", 1)[1] for kv in args.tiles]
    names = [kv.split("=", 1)[0] for kv in args.tiles]
    labels = args.labels if args.labels else names
    if len(labels) != len(tiles):
        raise ValueError("--labels count must match --tiles count.")
    levels = args.levels
    wti = args.weight_tile if args.weight_tile is not None else 0

    model_list = args.model
    if len(model_list) == 1 and model_list[0].lower() == "all":
        model_list = [m.label for m in cfg.models]

    for mdl in model_list:
        caps = _capture(cfg, mdl, tiles, "neck")
        # selectivity weights from the reference tile, per level
        wmaps = {}
        for lvl in levels:
            fm = caps[tiles[wti]][lvl]
            cover = _crown_coverage(cfg, tiles[wti], fm.shape[1:])
            wmaps[lvl] = _channel_weights(fm, cover)

        nrow, ncol = len(tiles), 2 + len(levels)
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(1.95 * ncol, 1.95 * nrow + 0.35),
                                 squeeze=False,
                                 gridspec_kw=dict(wspace=0.03, hspace=0.05,
                                                  left=0.055, right=0.995,
                                                  top=0.90, bottom=0.01))
        for ri, (t, lab) in enumerate(zip(tiles, labels)):
            ax = axes[ri][0]
            ax.imshow(_rgb_resized(t))
            style_image_axis(ax)
            ax.set_ylabel(f"({list('abcdefgh')[ri]}) {lab}", fontsize=9)
            if ri == 0:
                ax.set_title("Image", fontsize=9)

            gt = _gt_mask_display(cfg, t)
            ax = axes[ri][1]
            if gt is not None:
                ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(np.zeros((DISPLAY, DISPLAY)), cmap="gray",
                          vmin=0, vmax=1)
                ax.text(0.5, 0.5, "no GT", color="white", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8)
            style_image_axis(ax)
            if ri == 0:
                ax.set_title("Ground truth", fontsize=9)

            for li, lvl in enumerate(levels):
                fm = caps[t][lvl]
                resp = _selective_response(fm, wmaps[lvl])
                cover = _crown_coverage(cfg, t, resp.shape)
                frac, area = _concentration(resp, cover)
                ax = axes[ri][2 + li]
                ax.imshow(_stretch(_resize(np.clip(resp, 0, None))),
                          cmap=args.cmap)
                style_image_axis(ax)
                if frac is not None:
                    ax.annotate(f"{frac*100:.0f}% in {area*100:.0f}% area",
                                (0.03, 0.97), xycoords="axes fraction",
                                va="top", fontsize=6.5, color="white",
                                bbox=dict(fc="black", alpha=0.55,
                                          pad=1.5, lw=0))
                if ri == 0:
                    h, w = fm.shape[1:]
                    ax.set_title(f"{_level_tag('neck', lvl)} response "
                                 f"({h}x{w})", fontsize=9)
        fig.suptitle(f"{mdl}: crown-selective response across sensors "
                     f"(weights fit on {labels[wti]}, applied unchanged "
                     f"to all sensors)", fontsize=10)
        base = osp.join(args.out, f"sensors_{mdl}")
        os.makedirs(args.out, exist_ok=True)
        save_figure(fig, base)
        LOGGER.info("Wrote %s.{pdf,png}", base)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("channels", help="previous-paper-style channel mosaic")
    c.add_argument("--config", required=True)
    c.add_argument("--model", nargs="+", required=True, metavar="LABEL",
                   # 'all' expands to every model in the config
                   help="one or more models; several stack as (a)/(b) panels")
    c.add_argument("--tile", default=None,
                   help="single tile (multi-model mode)")
    c.add_argument("--tiles", nargs="+", default=None, metavar="NAME=PATH",
                   help="one model across several tiles/sensors, stacked as "
                        "(a)/(b)/(c) blocks (e.g. UAV=..., GE=..., Aerial=...)")
    c.add_argument("--labels", nargs="+", default=None,
                   help="block titles for --tiles (default: the NAME parts)")
    c.add_argument("--source", default="backbone", choices=["backbone", "neck"])
    c.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    c.add_argument("--topk", type=int, default=8)
    c.add_argument("--cmap", default="viridis",
                   help="matplotlib colormap for channel maps (default "
                        "viridis -- perceptually uniform and the de facto "
                        "standard; alternatives: magma, cividis, gray).")
    c.add_argument("--transpose", action="store_true",
                   help="levels as COLUMNS, channels as ROWS -- the most "
                        "compact layout for a manuscript column.")
    c.add_argument("--out", required=True)

    l = sub.add_parser("ladder", help="multiscale x GSD/sensor response matrix")
    l.add_argument("--config", required=True)
    l.add_argument("--model", nargs="+", required=True, metavar="LABEL",
                   help="one or more models; one figure per model unless "
                        "--combine is given")
    l.add_argument("--tiles", nargs="+", required=True,
                   metavar="NAME=PATH", help="ordered column tiles -- may be "
                   "the same scene at different GSD, or different scenes "
                   "from different sensors (e.g. UAV, GE, Aerial)")
    l.add_argument("--labels", nargs="+", default=None,
                   help="column titles (defaults to the NAME parts)")
    l.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    l.add_argument("--metric", choices=["selective", "magnitude"],
                   default="selective",
                   help="'selective' (default): per-channel crown-selectivity "
                        "weights fit on --weight-tile and applied to every "
                        "column -- shows whether a crown probe built on one "
                        "image still fires on the others. 'magnitude': plain "
                        "L2 norm over all channels.")
    l.add_argument("--weight-tile", type=int, default=None,
                   help="index into --tiles for the reference scene used by "
                        "--metric selective (default: 0).")
    l.add_argument("--style", choices=["pure", "overlay"], default="pure",
                   help="'pure' (default): response maps rendered directly in "
                        "--cmap, matching the channel-mosaic style; "
                        "'overlay': heat overlay on the imagery.")
    l.add_argument("--cmap", default="viridis",
                   help="colormap for --style pure (default viridis).")
    l.add_argument("--combine", action="store_true",
                   help="stack multiple --model values as (a)/(b)/(c) blocks "
                        "in ONE figure instead of one figure per model.")
    l.add_argument("--out", required=True)

    s = sub.add_parser("sensors", help="Image | GT | per-level responses, "
                                       "one row per sensor")
    s.add_argument("--config", required=True)
    s.add_argument("--model", nargs="+", required=True, metavar="LABEL",
                   help="one or more models ('all' = every configured model); "
                        "one figure per model")
    s.add_argument("--tiles", nargs="+", required=True, metavar="NAME=PATH")
    s.add_argument("--labels", nargs="+", default=None)
    s.add_argument("--levels", type=int, nargs="+", default=[0, 1],
                   help="pyramid levels to show as response columns "
                        "(default P2 P3 -- the levels where crowns resolve)")
    s.add_argument("--weight-tile", type=int, default=None,
                   help="index into --tiles for fitting the selectivity "
                        "weights (default 0).")
    s.add_argument("--cmap", default="magma",
                   help="colormap for the response panels (default magma).")
    s.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    configure_logging()
    import matplotlib
    matplotlib.use("Agg")
    set_publication_style()
    cfg = AnalysisConfig.from_json(args.config)
    if args.cmd == "sensors":
        fig_sensors(cfg, args)
        return 0
    if args.cmd == "channels":
        if not args.tile and not args.tiles:
            ap.error("channels requires either --tile or --tiles.")
        fig_channels(cfg, args)
    else:
        fig_ladder(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
