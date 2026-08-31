"""
make_qualitative_grid.py
=========================================================================
Publication qualitative-comparison grid for the date-palm instance
segmentation benchmark (Stage C unified multi-source model).

Layout (default): 9 rows x 7 columns
    Columns : Image | Ground truth | ResNet-50 | ConvNeXt-S | Swin-S |
              MambaVision-S | Spatial-Mamba-S
    Rows    : 3 tiles per source group -- UAV (3-5 cm), Aerial (15 cm),
              Google Earth (15 cm) -- with row letters (a)-(i) and a
              rotated group label spanning each 3-row block.

Reuses the rendering primitives (mask decoding, background dimming,
per-instance golden-ratio colouring, fills, contours) from the companion
script `visualize_predictions_for_publication.py`, so panel styling is
identical to the single-model visualisations. Ground truth is rendered
through the SAME primitives from the COCO annotation JSON, so the only
visual difference between the GT column and the model columns is mask
quality itself.

=========================================================================
STEP 0 -- PREREQUISITE PKLs (per-sensor diagonal protocol, Stage C)
=========================================================================
One pkl per (backbone x test set). Under the Stage C diagonal protocol:
    best_UAV checkpoint --> UAV test set
    best_GE  checkpoint --> GE and Aerial test sets

    python tools/test.py CONFIG best_UAV.pth --out results/qual/convnext_s_stageC_UAV.pkl
    python tools/test.py CONFIG best_GE.pth  --out results/qual/convnext_s_stageC_GE.pkl
    python tools/test.py CONFIG best_GE.pth  --out results/qual/convnext_s_stageC_Aerial.pkl
(repeat for each backbone; the test dataloader in CONFIG must point at the
matching test split each time)

=========================================================================
STEP 1 -- EDIT `CONFIG` BELOW (paths, tile picks, thresholds)
STEP 2 -- RUN
=========================================================================
    python make_qualitative_grid.py                 # explicit tiles in CONFIG
    python make_qualitative_grid.py --suggest-tiles # print density-stratified
                                                    # tile candidates per source,
                                                    # then exit (no rendering)
Outputs <out-stem>.png (600 dpi) and <out-stem>.pdf into --out-dir.

REQUIREMENTS
    pip install opencv-python-headless pycocotools numpy matplotlib tqdm
    `visualize_predictions_for_publication.py` in the same directory (or
    set VIZ_SCRIPT_PATH below).
"""

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgb, to_hex
from pycocotools.coco import COCO

# ------------------------------------------------------------------------
# Import the shared rendering primitives from the existing script
# ------------------------------------------------------------------------
VIZ_SCRIPT_PATH = Path(__file__).parent / "visualize_predictions_for_publication.py"

def _load_viz_module(path):
    if not Path(path).exists():
        sys.exit(f"[FATAL] Renderer not found: {path}\n"
                 f"Place visualize_predictions_for_publication.py next to this "
                 f"script or edit VIZ_SCRIPT_PATH.")
    spec = importlib.util.spec_from_file_location("viz", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

viz = _load_viz_module(VIZ_SCRIPT_PATH)


# ------------------------------------------------------------------------
# Typography and family palette -- PORTED VERBATIM from the figure notebook
# (stage_ABC_figures_v11.ipynb) so this grid is typographically identical to
# the leaderboard/heatmap/Pareto figures in the same paper.
#   SANS  : the notebook's resolved sans family (used by every theme's rc)
#   _BASE : the notebook's family palette, keyed CNN / Transformer / Mamba
# ------------------------------------------------------------------------
def _font(*cands):
    installed = {f.name for f in fm.fontManager.ttflist}
    for c in cands:
        if c in installed:
            return c
    return "DejaVu Sans"

SANS  = _font("Source Sans Pro", "Source Sans 3", "Helvetica Neue",
              "Arial", "Liberation Sans", "DejaVu Sans")
SERIF = _font("Source Serif 4", "Source Serif Pro", "Charter", "Georgia",
              "Liberation Serif", "DejaVu Serif")

# Notebook's _BASE (the "tidy" theme uses it undesaturated).
_BASE = {"CNN": "#35617F", "Transformer": "#C0823C", "Mamba": "#2E8B7F"}


def _tint(color, amount):
    """Notebook's tint(): blend toward white by `amount` (0-1)."""
    r, g, b = to_rgb(color)
    return to_hex((r + (1 - r) * amount,
                   g + (1 - g) * amount,
                   b + (1 - b) * amount))

# ==========================================================================
# CONFIG -- edit this block only
# ==========================================================================
CONFIG = {
    # Column order of the model panels. Keys must match the per-source
    # "preds" dicts below. The second line of each header states the family
    # (visual support for the family-convergence claim); set to None to
    # suppress.
    # Keys must match the pkl names written by make_stagec_pkls.sh
    # (<Key>_stageC_<Set>.pkl). ConvNeXt-T, not -S: the Stage C cohort's
    # CNN pair is ResNet-50 + ConvNeXt-T.
    "models": [
        ("ResNet-50",      "CNN"),
        ("ConvNeXt-T",     "CNN"),
        ("Swin-S",         "Transformer"),
        ("MambaVision-S",  "SSM hybrid"),
        ("SpatialMamba-S", "SSM"),
    ],

    # One entry per source group (3 rows each, in this order).
    # "tiles": exactly 3 file names from the test split. Leave [] to let
    # --suggest-tiles propose density-stratified candidates.
    "sources": [
        {
            "name":     "UAV (5 cm)",
            # UAV tiles here are 5 cm (confirmed) -- matches the UAV_5cm
            # corpus and the "UAV (5 cm)" axis label in the leaderboards.
            # "gsd_cm" also accepts a list of 3 values (one per tile) for a
            # source whose GSD varies tile to tile.
            "gsd_cm":   5,
            "scalebar_m": 10,
            "img_dir":  "/workspace/datasets/COCO/UAV_5cm/test_UAV/",
            "ann_json": "/workspace/datasets/COCO/UAV_5cm/Annotations/test_UAV.json",
            "tiles":    ["JPEGImages/000000006774.jpg",
                         "JPEGImages/000000006087.jpg",
                         "JPEGImages/000000007042.jpg"],
            "preds": {
                "ResNet-50":      "results/qual/ResNet-50_stageC_UAV.pkl",
                "ConvNeXt-T":     "results/qual/ConvNeXt-T_stageC_UAV.pkl",
                "Swin-S":         "results/qual/Swin-S_stageC_UAV.pkl",
                "MambaVision-S":  "results/qual/MambaVision-S_stageC_UAV.pkl",
                "SpatialMamba-S": "results/qual/SpatialMamba-S_stageC_UAV.pkl",
            },
        },
        {
            "name":     "Aerial (15 cm)",
            "gsd_cm":   15,
            "scalebar_m": 30,
            "img_dir":  "/workspace/datasets/COCO/Aerial_15cm/test_aerial/",
            "ann_json": "/workspace/datasets/COCO/Aerial_15cm/Annotations/test_aerial.json",
            "tiles":    ["JPEGImages/test_000000000180.jpg",
                         "JPEGImages/test_000000000284.jpg",
                         "JPEGImages/test_000000000185.jpg"],
            "preds": {
                "ResNet-50":      "results/qual/ResNet-50_stageC_Aerial.pkl",
                "ConvNeXt-T":     "results/qual/ConvNeXt-T_stageC_Aerial.pkl",
                "Swin-S":         "results/qual/Swin-S_stageC_Aerial.pkl",
                "MambaVision-S":  "results/qual/MambaVision-S_stageC_Aerial.pkl",
                "SpatialMamba-S": "results/qual/SpatialMamba-S_stageC_Aerial.pkl",
            },
        },
        {
            "name":     "GE (15 cm)",
            "gsd_cm":   15,
            "scalebar_m": 30,
            "img_dir":  "/workspace/datasets/COCO/GE_15cm/test_GE/",
            "ann_json": "/workspace/datasets/COCO/GE_15cm/Annotations/test_GE.json",
            "tiles":    ["JPEGImages/RAK_1_GE_2025_15cm_test_tile_000104.jpg",
                         "JPEGImages/AlAin_1_GE_2025_15cm_test_tile_000076.jpg",
                         "JPEGImages/Grid_R8_C1_test_tile_000018.jpg"],
            "preds": {
                "ResNet-50":      "results/qual/ResNet-50_stageC_GE.pkl",
                "ConvNeXt-T":     "results/qual/ConvNeXt-T_stageC_GE.pkl",
                "Swin-S":         "results/qual/Swin-S_stageC_GE.pkl",
                "MambaVision-S":  "results/qual/MambaVision-S_stageC_GE.pkl",
                "SpatialMamba-S": "results/qual/SpatialMamba-S_stageC_GE.pkl",
            },
        },
    ],

    # Confidence threshold applied to every model panel. Use the
    # val-selected operating threshold reported in the paper; per-model
    # overrides allowed, e.g. {"default": 0.45, "Swin-S": 0.50}.
    "score_thr": {"default": 0.45},

    # Panel styling (passed to the shared primitives). Chips/boxes are off:
    # at ~24 mm printed panel width, per-instance labels are illegible and
    # occlude the masks. Mild dim/desat keeps context readable while masks
    # carry the figure.
    "dim_factor":        0.70,   # 1.0 = natural background (no dimming)
    "desat_factor":      0.55,   # 1.0 = natural colours
    "mask_alpha":        0.55,
    "contour_thickness": 2,
    # Mask palette: golden-ratio hues at controlled saturation/value
    # (0.58/0.88 reads as saturated-but-print-friendly; raise saturation
    # toward 0.9 to recover the previous vivid look).
    "mask_saturation":   0.58,
    "mask_value":        0.88,
    "contour_darken":    0.72,

    # Figure geometry: full double-column width (~180 mm = 7.09 in).
    "fig_width_in":  7.09,
    "panel_gap":     0.015,   # wspace/hspace as fraction of panel size
    "group_gap":     0.042,   # extra vertical gap between source groups
    "header_fs":     8.0,     # column header font size (pt)
    "family_fs":     6.5,     # family sub-line font size (pt)
    "label_fs":      8.0,     # (a)-(i) row letters
    "group_fs":      8.0,     # rotated group labels
    # Font: resolved exactly as the notebook does (SANS), so this figure
    # matches the leaderboard/heatmap/Pareto figures. Set to SERIF only if
    # the notebook's "editorial" theme (serif titles) is the target.
    "font_family":   SANS,
    # Family accents = the notebook's _BASE palette, so a colour means the
    # same family here as in every quantitative figure.
    "family_colors": {"CNN":         _BASE["CNN"],
                      "Transformer": _BASE["Transformer"],
                      "SSM hybrid":  _BASE["Mamba"],
                      "SSM":         _BASE["Mamba"]},
    "sep_col_w":     0.12,    # spacer column between (Image, GT) and models

    # Instance-count chips. The GT panel carries the reference count; each
    # model panel carries its own count at the operating threshold, and
    # (optionally) the signed difference from GT. Counting crowns by eye is
    # infeasible at this panel size, so the chip is what makes the columns
    # comparable rather than merely similar-looking.
    # CAVEAT for the caption: a matching count is NOT proof of correctness --
    # false positives and false negatives can cancel. The chip reports how
    # many instances each model returned, not how many it got right; the
    # per-tile F1 in the divergence figure is the accuracy statement.
    "show_counts":   True,
    "count_delta":   True,    # model panels also show (+n / -n) vs GT
    "count_fs":      5.6,     # chip font size (pt)
    "count_loc":     "lower left",   # or "lower right"
}
# ==========================================================================


# --------------------------------------------------------------------------
# Rendering helpers (all styling delegated to the shared primitives)
# --------------------------------------------------------------------------

import colorsys

def _soft_color(rank, cfg):
    """Golden-ratio hue sequence with controlled saturation/value: distinct
    but not neon. Returns (fill_bgr, contour_bgr); contour is a darkened
    version of the fill for crisp edges without a second hue."""
    hue = (rank * 0.61803398875) % 1.0
    sat = cfg.get("mask_saturation", 0.58)
    val = cfg.get("mask_value", 0.88)
    dk  = cfg.get("contour_darken", 0.72)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    fill = (int(b * 255), int(g * 255), int(r * 255))
    r2, g2, b2 = colorsys.hsv_to_rgb(hue, min(1.0, sat * 1.15), val * dk)
    contour = (int(b2 * 255), int(g2 * 255), int(r2 * 255))
    return fill, contour


def render_panel(img_bgr, masks, cfg):
    """Dim background outside the mask union, then per-instance fill+contour,
    largest to smallest, using the shared golden-ratio palette."""
    H, W = img_bgr.shape[:2]
    resized = []
    for m in masks:
        if m is not None and m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        resized.append(m)

    canvas = viz.dim_background(img_bgr, resized,
                                dim_factor=cfg["dim_factor"],
                                desat_factor=cfg["desat_factor"])

    areas = [int(m.sum()) if m is not None else 0 for m in resized]
    order = np.argsort(-np.asarray(areas))
    for rank, i in enumerate(order):
        m = resized[int(i)]
        if m is None or areas[int(i)] == 0:
            continue
        fill, contour = _soft_color(rank, cfg)
        viz.draw_mask_fill(canvas, m, fill, cfg["mask_alpha"])
        viz.draw_mask_contour(canvas, m, contour, cfg["contour_thickness"])
    return canvas


def count_chip(ax, n, delta, cfg):
    """Dark rounded chip with the instance count; on model panels, append the
    signed difference from the ground-truth count."""
    txt = f"n = {n}"
    if delta is not None:
        sign = "+" if delta > 0 else ("\u2212" if delta < 0 else "\u00b1")
        txt += f"  ({sign}{abs(delta)})"
    right = cfg.get("count_loc", "lower left") == "lower right"
    ax.text(0.965 if right else 0.035, 0.035, txt,
            transform=ax.transAxes, ha="right" if right else "left",
            va="bottom", fontsize=cfg["count_fs"], color="white", zorder=6,
            bbox=dict(boxstyle="round,pad=0.24", fc="#1A1A1A", ec="none",
                      alpha=0.62))


def gt_masks_for_image(coco, img_id, H, W):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))
    return [viz.decode_single_mask(a["segmentation"], H, W) for a in anns]


def pred_masks_for_image(pkl_index, img_id, score_thr):
    """Return the thresholded predicted masks for one image from an indexed pkl."""
    result = pkl_index.get(img_id)
    if result is None:
        return None
    (_, _, bboxes, labels, scores, masks, img_shape) = viz.unpack_result(result)
    keep = np.where(scores >= score_thr)[0]
    return [masks[int(i)] for i in keep]


def load_pkl_index(path):
    with open(path, "rb") as f:
        results = pickle.load(f)
    return {r["img_id"]: r for r in results}


# --------------------------------------------------------------------------
# Tile selection helper
# --------------------------------------------------------------------------

def read_display_bgr(img_path, bands=None, stretch=None):
    """Read one tile as a 3-channel BGR array for rendering.

    bands=None            -> ordinary 3-channel image via cv2 (RGB sources).
    bands=[R, G, B]       -> 1-based TILE channel indices to place in the
                             display red, green and blue guns, read with
                             rasterio. For the WorldView-3 MS build the tile
                             channel order is
                                 1 R  2 G  3 B  4 Coastal
                                 5 Yellow  6 RedEdge  7 NIR1  8 NIR2
                             so [7, 2, 3] is the NIR-G-B false colour in
                             which healthy palm crowns read bright red, and
                             [1, 2, 3] is true colour.

    stretch=(lo, hi)      -> optional per-tile percentile stretch, applied
                             per channel. Off by default: the tiler already
                             wrote 8-bit stretched pixels, and a second
                             per-tile stretch makes panels in different rows
                             radiometrically incomparable. Enable only if the
                             false-colour composite prints too flat, and say
                             so in the caption if you do.
    """
    if bands is None:
        img = cv2.imread(str(img_path))
        if img is None:
            sys.exit(f"[FATAL] Cannot read image: {img_path}")
        return img

    try:
        import rasterio
    except ImportError:
        sys.exit("[FATAL] reading multiband tiles needs rasterio")

    if len(bands) != 3:
        sys.exit(f"[FATAL] 'bands' must name exactly 3 channels, got {bands}")

    with rasterio.open(str(img_path)) as ds:
        if max(bands) > ds.count:
            sys.exit(f"[FATAL] band {max(bands)} requested but {img_path} "
                     f"has {ds.count} channels")
        arr = ds.read(indexes=list(bands)).astype(np.float32)  # (3, H, W)

    if stretch is not None:
        lo_p, hi_p = stretch
        for i in range(3):
            v = arr[i][arr[i] > 0]
            if v.size == 0:
                continue
            lo, hi = np.percentile(v, lo_p), np.percentile(v, hi_p)
            if hi - lo < 1e-6:
                hi = lo + 1.0
            arr[i] = np.clip((arr[i] - lo) / (hi - lo) * 255.0, 0, 255)
    elif arr.max() > 255:
        # uint16 or float source with no stretch requested: scale rather than
        # clip, so nothing is silently blown out.
        arr = arr / arr.max() * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    # arr is ordered (R, G, B) for display; the renderer works in BGR.
    return np.stack([arr[2], arr[1], arr[0]], axis=-1)


def suggest_tiles(cfg, k_per_stratum=4, group_re=None,
                  percentiles=(20, 50, 90), min_valid_frac=0.0):
    """Print density-stratified candidates (sparse / median / dense) per
    source: the 20th, 50th and 90th percentile of GT instance count. Pick one
    per stratum so each row block spans the density range the error analysis
    stratifies over.

    group_re, if given, is a regular expression with one capturing group
    applied to the file name; candidates are then stratified WITHIN each
    captured group as well as overall. Use it when the tiles come from
    several sites whose density distributions differ, since a single global
    stratification will otherwise draw every sparse candidate from one site
    and every dense one from another, and the resulting figure confounds
    density with site.

    percentiles selects the strata. The default (20, 50, 90) gives the
    sparse/median/dense triple; pass more values when the grid has more rows
    to fill, so each row sits at a distinct point of the density range rather
    than duplicating a stratum.

    min_valid_frac, if above zero, excludes tiles whose imagery is largely
    nodata. Mosaic seams and block edges leave wide black bands, and a tile
    like that makes a poor figure row whatever its crown count.
    """
    for src in cfg["sources"]:
        coco = COCO(src["ann_json"])
        counts = []
        for img in coco.dataset["images"]:
            n = len(coco.getAnnIds(imgIds=img["id"], iscrowd=None))
            if n > 0:
                counts.append((n, img["file_name"]))
        counts.sort()

        # Drop tiles whose imagery is largely nodata before stratifying.
        # Mosaic seams and block edges leave wide black bands, and such a
        # tile makes a poor figure row however well its crown count fits a
        # stratum: the reader sees black, and the model was working from a
        # fraction of the frame. Checked on a decimated read, so the cost is
        # a fraction of a second per tile.
        if min_valid_frac > 0.0:
            try:
                import rasterio
                from rasterio.enums import Resampling
            except ImportError:
                sys.exit("[FATAL] --min-valid needs rasterio")
            kept, dropped = [], 0
            for n, fname in counts:
                path = Path(src["img_dir"]) / fname
                try:
                    with rasterio.open(str(path)) as ds:
                        h = max(1, ds.height // 8)
                        w = max(1, ds.width // 8)
                        a = ds.read(out_shape=(ds.count, h, w),
                                    resampling=Resampling.nearest)
                except Exception:
                    kept.append((n, fname))       # unreadable here, let the
                    continue                       # renderer report it
                valid = float((a.max(axis=0) > 0).mean())
                if valid >= min_valid_frac:
                    kept.append((n, fname))
                else:
                    dropped += 1
            if dropped:
                print(f"  [filter] dropped {dropped} tile(s) below "
                      f"{min_valid_frac:.0%} valid pixels")
            counts = kept
            if not counts:
                sys.exit(f"[FATAL] every tile in '{src['name']}' fell below "
                         f"--min-valid {min_valid_frac}")

        groups = {"": counts}
        if group_re:
            import re as _re
            from collections import defaultdict
            grouped = defaultdict(list)
            for n, fname in counts:
                m = _re.search(group_re, fname)
                grouped[m.group(1) if m else "(unmatched)"].append((n, fname))
            groups = dict(sorted(grouped.items()))

        for gname, gcounts in groups.items():
            n_arr = np.array([c[0] for c in gcounts])
            head = src["name"] + (f" / {gname}" if gname else "")
            print(f"\n=== {head} -- {len(gcounts)} non-empty tiles, "
                  f"instance count min/med/max = "
                  f"{n_arr.min()}/{int(np.median(n_arr))}/{n_arr.max()}")
            for q in percentiles:
                label = f"P{int(q):<3d}"
                target = np.percentile(n_arr, q)
                nearest = np.argsort(np.abs(n_arr - target))[:k_per_stratum]
                picks = ", ".join(f"{gcounts[i][1]} ({gcounts[i][0]})"
                                  for i in sorted(nearest))
                print(f"  {label}: {picks}")


# --------------------------------------------------------------------------
# Grid assembly
# --------------------------------------------------------------------------

def build_grid(cfg, out_dir, out_stem):
    # Mirror the notebook's rc for the "tidy" theme: same family, same ink.
    plt.rcParams.update({
        "font.family":   cfg["font_family"],
        "text.color":    "#2A2A2A",
        "axes.labelcolor": "#2A2A2A",
        "pdf.fonttype":  42,
        "ps.fonttype":   42,
    })
    models = cfg["models"]
    n_cols = 2 + len(models)
    n_src = len(cfg["sources"])
    # Rows per source group. Three by default (the Stage C figure's
    # sparse/median/dense triple), but the Stage D single-source grid uses
    # more, so it is configurable. Every source group must supply this many
    # tiles: the group label spans exactly this many rows and the row-letter
    # sequence is derived from it.
    # Rows per source group. Three by default; a source may instead declare
    # its own "rows" so blocks of different height can share a figure (the
    # Stage D grid splits by acquisition site, and the sites need not
    # contribute equally).
    _default_rows = int(cfg.get("rows_per_src", 3))
    rows_list = [int(src.get("rows", _default_rows)) for src in cfg["sources"]]
    n_img_rows = sum(rows_list)
    if not 1 <= n_img_rows <= 26:
        sys.exit(f"[FATAL] {n_img_rows} image rows requested; row letters run "
                 f"(a)-(z), so 26 is the ceiling.")
    # First image-row index of each source block, for the group labels and for
    # mapping a row back to the source it belongs to.
    row_offset = []
    _acc = 0
    for r in rows_list:
        row_offset.append(_acc)
        _acc += r

    def _src_of_row(i):
        """Source index owning image row i."""
        for s_ in range(n_src - 1, -1, -1):
            if i >= row_offset[s_]:
                return s_
        return 0

    # ---- Gather all panels first (fail early on missing files) ------------
    thr_map = cfg["score_thr"]
    grid = []          # list of rows; each row = list of RGB arrays
    counts = []        # parallel to grid: [None(Image), n_gt, n_m1, ...]
    for src in cfg["sources"]:
        coco = COCO(src["ann_json"])
        name_to_img = {img["file_name"]: img for img in coco.dataset["images"]}
        tiles = src["tiles"]
        rows_this = rows_list[cfg["sources"].index(src)]
        if len(tiles) != rows_this:
            sys.exit(f"[FATAL] Source '{src['name']}' needs exactly "
                     f"{rows_this} tiles in CONFIG (got {len(tiles)}). "
                     f"Run --suggest-tiles for candidates.")
        pkl_indexes = {m: load_pkl_index(src["preds"][m]) for m, _ in models}

        for fname in tiles:
            if fname not in name_to_img:
                sys.exit(f"[FATAL] Tile '{fname}' not in {src['ann_json']}")
            info = name_to_img[fname]
            img_id, H, W = info["id"], info["height"], info["width"]
            img_path = Path(src["img_dir"]) / fname
            img_bgr = read_display_bgr(img_path,
                                       bands=src.get("bands"),
                                       stretch=src.get("display_stretch"))

            row_panels = [cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)]

            gt_masks = gt_masks_for_image(coco, img_id, H, W)
            gt = render_panel(img_bgr, gt_masks, cfg)
            row_panels.append(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
            row_counts = [None, len(gt_masks)]

            for m, _fam in models:
                thr = thr_map.get(m, thr_map["default"])
                masks = pred_masks_for_image(pkl_indexes[m], img_id, thr)
                if masks is None:
                    sys.exit(f"[FATAL] img_id {img_id} ('{fname}') missing from "
                             f"{src['preds'][m]}")
                panel = render_panel(img_bgr, masks, cfg)
                row_panels.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
                row_counts.append(len(masks))
            grid.append(row_panels)
            counts.append(row_counts)
            print(f"[OK] rendered row: {src['name']} / {fname}"
                  + (f"  (GT n={row_counts[1]}; "
                     + ", ".join(f"{m}={c}" for (m, _), c
                                 in zip(models, row_counts[2:])) + ")"
                     if cfg.get("show_counts") else ""))

    # ---- Figure geometry ---------------------------------------------------
    # Square panels; spacer rows between source groups.
    # Top strip for the column headers. Specified in INCHES, not as a
    # fraction of figure height: the headers are two lines of fixed-point
    # text, so they need a fixed absolute strip. Expressing it as a fraction
    # made the strip shrink with the figure, and a grid with few rows -- the
    # Stage D single-source case -- clipped the backbone names off the top.
    header_in = float(cfg.get("header_in", 0.42))
    hr = []
    spacer = cfg["group_gap"] * 4               # spacer height in panel units
    for s in range(n_src):
        hr.extend([1.0] * rows_list[s])
        if s < n_src - 1:
            hr.append(spacer)
    n_grid_rows = len(hr)

    # Column layout: Image | GT | <spacer> | model_1..model_k. The spacer
    # visually separates the reference columns from the model comparison.
    wr = [1.0, 1.0, cfg["sep_col_w"]] + [1.0] * len(models)
    n_grid_cols = len(wr)
    _left = 0.075 if any(s_["name"] for s_ in cfg["sources"]) else 0.035
    panel_w_frac = (0.990 - _left) / sum(wr)      # width of ONE unit panel
    fig_w = cfg["fig_width_in"]
    panel_in = fig_w * panel_w_frac
    # Solve fig_h for a header of fixed absolute height:
    #   fig_h = panel_in*sum(hr) / (1 - f - 0.005)   with   f = header_in/fig_h
    # gives fig_h = (panel_in*sum(hr) + header_in) / 0.995
    fig_h = (panel_in * sum(hr) + header_in) / 0.995
    header_h_frac = header_in / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Left margin holds the rotated group label plus the row letters. With no
    # group label only the letters need room.
    left_margin = 0.075 if any(s_["name"] for s_ in cfg["sources"]) else 0.035
    gs = GridSpec(n_grid_rows, n_grid_cols, figure=fig,
                  left=left_margin, right=0.990,
                  top=1.0 - header_h_frac, bottom=0.004,
                  wspace=cfg["panel_gap"], hspace=cfg["panel_gap"],
                  height_ratios=hr, width_ratios=wr)

    col_titles = ["Image", "Ground truth"] + [m for m, _ in models]
    col_fams   = [None, None] + [f for _, f in models]
    # grid-column index for each content column (skipping the spacer)
    col_slots  = [0, 1] + list(range(3, 3 + len(models)))

    letters = [f"({chr(ord('a') + i)})" for i in range(n_img_rows)]
    axes_first_col = []
    axes_first_row = []
    img_row = 0
    for gr in range(n_grid_rows):
        if hr[gr] != 1.0:
            continue                             # spacer row
        for c in range(n_cols):
            ax = fig.add_subplot(gs[gr, col_slots[c]])
            ax.imshow(grid[img_row][c], interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("0.25")
            # count chip: GT gets the reference count, model panels get their
            # own count (+ signed difference from GT); the Image panel gets none
            if cfg.get("show_counts") and counts[img_row][c] is not None:
                n_here = counts[img_row][c]
                d = (n_here - counts[img_row][1]
                     if (cfg.get("count_delta") and c >= 2) else None)
                count_chip(ax, n_here, d, cfg)
            if img_row == 0:                     # column headers on top row
                axes_first_row.append(ax)
                title = col_titles[c]
                if col_fams[c]:
                    ax.set_title(f"{title}\n{col_fams[c]}",
                                 fontsize=cfg["header_fs"], pad=3.0,
                                 linespacing=1.15)
                    # shrink the family line
                    t = ax.title
                    t.set_fontsize(cfg["header_fs"])
                else:
                    ax.set_title(title, fontsize=cfg["header_fs"], pad=3.0)
            if c == 0:
                ax.set_ylabel(letters[img_row], fontsize=cfg["label_fs"],
                              rotation=0, ha="right", va="center", labelpad=4)
                axes_first_col.append(ax)
                # scale bar (Image column only, if the source declares a GSD)
                src_i = _src_of_row(img_row)
                s_cfg = cfg["sources"][src_i]
                # GSD may be a scalar (uniform source) or a list of 3 values
                # (one per tile, for sources whose GSD varies tile to tile).
                _g = s_cfg.get("gsd_cm")
                if isinstance(_g, (list, tuple)):
                    _g = _g[img_row - row_offset[src_i]]
                s_cfg = dict(s_cfg, gsd_cm=_g)
                if s_cfg.get("gsd_cm") and s_cfg.get("scalebar_m"):
                    W_px = grid[img_row][0].shape[1]
                    bar_px = s_cfg["scalebar_m"] * 100.0 / s_cfg["gsd_cm"]
                    x0, y0 = 0.05 * W_px, 0.955 * grid[img_row][0].shape[0]
                    ax.plot([x0, x0 + bar_px], [y0, y0], color="w",
                            lw=1.6, solid_capstyle="butt",
                            path_effects=[pe.withStroke(linewidth=2.6,
                                                        foreground="0.15")])
                    ax.text(x0 + bar_px / 2, y0 - 0.018 * W_px,
                            f"{s_cfg['scalebar_m']} m", color="w",
                            ha="center", va="bottom", fontsize=5.5,
                            path_effects=[pe.withStroke(linewidth=1.6,
                                                        foreground="0.15")])
        img_row += 1

    # Two-size header: model name (serif), family sub-line in the family's
    # accent colour, and a short underline bar in the same colour -- ties the
    # grid to the leaderboard figures' family palette.
    fig.canvas.draw()
    for c in range(n_cols):
        ax = axes_first_row[c]
        if col_fams[c]:
            fam = col_fams[c]
            fcol = cfg["family_colors"].get(fam, "0.35")
            # Model name in the theme's ink; family as a small tinted chip
            # (rounded swatch + word) rather than a rule -- the rule read as
            # a dull underline at print size and competed with the panel
            # borders directly beneath it.
            ax.set_title(col_titles[c], fontsize=cfg["header_fs"],
                         color="#2A2A2A", pad=13.0)
            ax.annotate(f"  {fam}  ", xy=(0.5, 1.030),
                        xycoords="axes fraction", ha="center", va="bottom",
                        fontsize=cfg["family_fs"], color=fcol,
                        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.5",
                                  facecolor=_tint(fcol, 0.86),
                                  edgecolor="none"))
        else:
            ax.set_title(col_titles[c], fontsize=cfg["header_fs"],
                         color="#2A2A2A", pad=3.0)

    # Rotated source-group labels spanning each row block. A source with an
    # empty name draws none: with a single source the label repeats what the
    # caption already says and costs width the panels can use.
    fig.canvas.draw()
    for s, src in enumerate(cfg["sources"]):
        if not src["name"]:
            continue
        top_ax = axes_first_col[row_offset[s]]
        bot_ax = axes_first_col[row_offset[s] + rows_list[s] - 1]
        y0 = bot_ax.get_position().y0
        y1 = top_ax.get_position().y1
        fig.text(0.010, (y0 + y1) / 2.0, src["name"],
                 rotation=90, ha="left", va="center",
                 fontsize=cfg["group_fs"])

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{out_stem}.png"
    pdf = out_dir / f"{out_stem}.pdf"
    fig.savefig(png, dpi=600)
    fig.savefig(pdf, dpi=600)
    plt.close(fig)
    print(f"\n[DONE] {png}\n       {pdf}")
    return png, pdf


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out-dir",  default="results/qual/figures")
    p.add_argument("--out-stem", default="fig_qualitative_stageC")
    p.add_argument("--suggest-tiles", action="store_true",
                   help="Print density-stratified tile candidates per source "
                        "(P20/P50/P90 of GT instance count), then exit.")
    args = p.parse_args()

    if args.suggest_tiles:
        suggest_tiles(CONFIG)
        return
    build_grid(CONFIG, args.out_dir, args.out_stem)


if __name__ == "__main__":
    main()
