# -*- coding: utf-8 -*-
"""Shared utilities: logging, reproducible seeding, bootstrap confidence
intervals, publication figure styling, and figure saving.

This module has no dependency on the other package modules, so it can be
imported freely without risk of circular imports.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from typing import Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("feature_analysis")

# Okabe-Ito colourblind-safe palette, mapped to backbone families.
FAMILY_COLORS = {
    "CNN": "#0072B2",          # blue
    "Transformer": "#E69F00",  # orange
    "Mamba": "#009E73",        # bluish green
}
# Perceptually uniform, colourblind-safe sequential map for similarity matrices.
HEATMAP_CMAP = "cividis"
OVERLAY_CMAP = "inferno"


# --------------------------------------------------------------------------- #
def configure_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                            datefmt="%H:%M:%S"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def set_publication_style() -> None:
    """Configure Matplotlib for publication-quality vector output. Text is
    embedded as editable TrueType (fonttype 42) so figures remain editable in
    vector editors; spines are minimised and resolution is set for raster
    fallbacks."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_image_axis(ax) -> None:
    """Strip ticks and draw a thin uniform border, appropriate for image
    panels in a feature grid."""
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#444444")
        spine.set_linewidth(0.6)


def save_figure(fig, base_path: str) -> None:
    """Write a figure as both vector PDF (manuscript) and 300-dpi PNG."""
    import matplotlib.pyplot as plt
    fig.savefig(base_path + ".pdf")
    fig.savefig(base_path + ".png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def stable_seed(*parts) -> int:
    """Reproducible 32-bit seed from arbitrary parts, free of Python's
    per-process string-hash salt."""
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def location_indices(seed: int, tile_stem: str, level: int, hw: int, k: int) -> np.ndarray:
    """Deterministic subsample of k spatial-location indices out of hw, shared
    across all models so that cross-architecture similarity is location-matched.
    """
    rng = np.random.default_rng(stable_seed(seed, tile_stem, level))
    if k >= hw:
        return np.arange(hw)
    return np.sort(rng.choice(hw, size=k, replace=False))


def bootstrap_ci(values: Sequence[float], reps: int, seed: int,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Mean and percentile confidence interval of a per-tile statistic,
    resampling tiles with replacement."""
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, size=arr.size, replace=True).mean()
                      for _ in range(reps)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi
