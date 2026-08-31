# -*- coding: utf-8 -*-
"""Quantitative phases: feature extraction/caching and the aggregate analyses
(cross-architecture CKA, representational dimensionality, and the headline
cross-resolution Bures invariance), with publication-grade tables and figures.
"""

from __future__ import annotations

import csv
import json
import os
import os.path as osp
import platform
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import AnalysisConfig
from .extractor import FeatureExtractor, to_chw_numpy
from .metrics import (bures_fidelity, channel_covariance, debiased_linear_cka,
                      debiased_rbf_cka, effective_rank, participation_ratio)
from .sampling import (cache_path, load_matrix, load_sampled, sample_tiles)
from .util import (FAMILY_COLORS, HEATMAP_CMAP, LOGGER, bootstrap_ci,
                   location_indices, save_figure, set_publication_style,
                   stable_seed)


# --------------------------------------------------------------------------- #
# Phase 1: extraction and caching
# --------------------------------------------------------------------------- #
def phase_extract(cfg: AnalysisConfig) -> None:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    sampled: Dict[str, List[str]] = {}
    for ts in cfg.tile_sets:
        sampled[ts.name] = sample_tiles(ts, cfg.n_tiles, cfg.seed)
        LOGGER.info("Tile set %s: %d tiles sampled.", ts.name, len(sampled[ts.name]))
    with open(osp.join(cfg.cache_dir, "_sampled_tiles.json"), "w") as f:
        json.dump(sampled, f, indent=2)

    loc_meta: Dict[str, Dict] = {}
    ckpt_manifest: Dict[str, Dict[str, str]] = {}
    for spec in cfg.models:
        # Group tile sets by resolved checkpoint so each distinct weight
        # state is built exactly once (one detector resident at a time).
        groups: Dict[str, List] = {}
        for ts in cfg.tile_sets:
            ck = (spec.checkpoint_for(ts.name)
                  if hasattr(spec, "checkpoint_for") else spec.checkpoint)
            groups.setdefault(ck, []).append(ts)
            ckpt_manifest.setdefault(spec.label, {})[ts.name] = ck
        if len(groups) > 1:
            LOGGER.info("%s: %d distinct checkpoints across tile sets "
                        "(per-domain mode).", spec.label, len(groups))
        for ck, ts_list in groups.items():
            LOGGER.info("Building model: %s (%s) [%s]",
                        spec.label, spec.family, osp.basename(ck))
            ext = FeatureExtractor(spec.config, ck, cfg.device)
            try:
                for ts in ts_list:
                    for tile in sampled[ts.name]:
                        stem = osp.splitext(osp.basename(tile))[0]
                        try:
                            neck = ext.extract(tile)
                        except Exception as exc:
                            LOGGER.warning("Extraction failed for %s on %s: %s",
                                           spec.label, stem, exc)
                            continue
                        for level in cfg.fpn_levels:
                            if level >= len(neck):
                                raise IndexError(f"Requested FPN level {level} but neck "
                                                 f"has {len(neck)} levels.")
                            fmap = neck[level][0]
                            c, h, w = fmap.shape
                            hw = h * w
                            key = f"{ts.name}/{stem}/L{level}"
                            if key not in loc_meta:
                                idx = location_indices(cfg.seed, stem, level, hw,
                                                       cfg.max_locations_per_tile)
                                loc_meta[key] = {"hw": hw, "idx": idx.tolist(),
                                                 "channels": int(c),
                                                 "shape": [int(h), int(w)]}
                            else:
                                if loc_meta[key]["hw"] != hw:
                                    LOGGER.warning("Spatial-size mismatch for %s at %s; "
                                                   "this model is excluded from CKA for "
                                                   "this tile/level. Verify identical "
                                                   "test pipelines across configs.",
                                                   spec.label, key)
                                    continue
                                idx = np.array(loc_meta[key]["idx"])
                            mat = fmap.reshape(c, hw).T.numpy()[idx]
                            np.save(cache_path(cfg, spec.label, ts.name, stem, level),
                                    mat.astype(np.float32))
            finally:
                ext.close()
            LOGGER.info("Finished and released model: %s [%s]",
                        spec.label, osp.basename(ck))

    with open(osp.join(cfg.cache_dir, "_locindex.json"), "w") as f:
        json.dump(loc_meta, f)
    with open(osp.join(cfg.cache_dir, "_checkpoints.json"), "w") as f:
        json.dump(ckpt_manifest, f, indent=2)
    LOGGER.info("Extraction complete. Cache at %s", cfg.cache_dir)

# --------------------------------------------------------------------------- #
# Phase 2: aggregate analyses
# --------------------------------------------------------------------------- #
def analyze_cka(cfg: AnalysisConfig, sampled: Dict[str, List[str]]) -> Dict:
    out: Dict = {}
    labels = [m.label for m in cfg.models]
    for set_name, stems in sampled.items():
        for level in cfg.fpn_levels:
            per_pair_lin: Dict[Tuple[int, int], List[float]] = {}
            per_pair_rbf: Dict[Tuple[int, int], List[float]] = {}
            for stem in stems:
                mats = [load_matrix(cfg, lab, set_name, stem, level) for lab in labels]
                valid = [k for k, m in enumerate(mats) if m is not None]
                for a in range(len(valid)):
                    for b in range(a + 1, len(valid)):
                        i, j = valid[a], valid[b]
                        if mats[i].shape[0] != mats[j].shape[0]:
                            continue
                        per_pair_lin.setdefault((i, j), []).append(
                            debiased_linear_cka(mats[i], mats[j]))
                        if cfg.rbf_cka:
                            per_pair_rbf.setdefault((i, j), []).append(
                                debiased_rbf_cka(mats[i], mats[j]))
            n = len(labels)
            lin = np.full((n, n), np.nan); lin_lo = lin.copy(); lin_hi = lin.copy()
            np.fill_diagonal(lin, 1.0)
            for (i, j), vals in per_pair_lin.items():
                m, lo, hi = bootstrap_ci(vals, cfg.bootstrap, stable_seed(cfg.seed, i, j))
                lin[i, j] = lin[j, i] = m
                lin_lo[i, j] = lin_lo[j, i] = lo
                lin_hi[i, j] = lin_hi[j, i] = hi
            entry = {"labels": labels, "linear_cka": lin.tolist(),
                     "linear_cka_lo": lin_lo.tolist(), "linear_cka_hi": lin_hi.tolist()}
            if cfg.rbf_cka:
                rbf = np.full((n, n), np.nan); np.fill_diagonal(rbf, 1.0)
                for (i, j), vals in per_pair_rbf.items():
                    m, _, _ = bootstrap_ci(vals, cfg.bootstrap,
                                           stable_seed(cfg.seed, i, j, "rbf"))
                    rbf[i, j] = rbf[j, i] = m
                entry["rbf_cka"] = rbf.tolist()
            out[f"{set_name}__L{level}"] = entry
    return out


def analyze_dimensionality(cfg: AnalysisConfig, sampled: Dict[str, List[str]]) -> List[Dict]:
    rows: List[Dict] = []
    for m in cfg.models:
        for set_name, stems in sampled.items():
            for level in cfg.fpn_levels:
                er_vals, pr_vals = [], []
                for stem in stems:
                    mat = load_matrix(cfg, m.label, set_name, stem, level)
                    if mat is None:
                        continue
                    cov = channel_covariance(mat)
                    er_vals.append(effective_rank(cov))
                    pr_vals.append(participation_ratio(cov))
                er = bootstrap_ci(er_vals, cfg.bootstrap,
                                  stable_seed(cfg.seed, m.label, set_name, level, "er"))
                pr = bootstrap_ci(pr_vals, cfg.bootstrap,
                                  stable_seed(cfg.seed, m.label, set_name, level, "pr"))
                rows.append({"model": m.label, "family": m.family, "resolution": set_name,
                             "level": level, "eff_rank": er[0], "eff_rank_lo": er[1],
                             "eff_rank_hi": er[2], "part_ratio": pr[0],
                             "part_ratio_lo": pr[1], "part_ratio_hi": pr[2]})
    return rows


def analyze_resolution_invariance(cfg: AnalysisConfig,
                                  sampled: Dict[str, List[str]]) -> Tuple[List[Dict], Dict[str, float]]:
    """Cross-resolution channel-covariance Bures fidelity per backbone.

    Sufficient-statistics implementation: the covariance of any concatenation
    of per-tile matrices is computed exactly from per-tile Gram matrices,
    column sums, and row counts, so each bootstrap resample costs one weighted
    sum of (k, 256, 256) statistics plus one Bures evaluation, instead of
    re-concatenating ~80k feature rows and recomputing the covariance from
    raw data. The estimator and the bootstrap draw sequence are identical to
    the reference implementation; statistics are accumulated in float64.
    """
    res_names = list(sampled.keys())
    rows: List[Dict] = []
    score: Dict[str, float] = {}
    n_combo = len(cfg.models) * len(cfg.fpn_levels)
    done = 0

    def cov_of(S, s, n, w=None):
        if w is None:
            Ssum, ssum, N = S.sum(axis=0), s.sum(axis=0), float(n.sum())
        else:
            Ssum = np.tensordot(w, S, axes=1)
            ssum = w @ s
            N = float(w @ n)
        if N <= 1.0:
            return None
        return (Ssum - np.outer(ssum, ssum) / N) / (N - 1.0)

    for m in cfg.models:
        pair_means: List[float] = []
        for level in cfg.fpn_levels:
            stats: Dict[str, Optional[tuple]] = {}
            for r in res_names:
                mats = [load_matrix(cfg, m.label, r, s, level) for s in sampled[r]]
                mats = [x.astype(np.float64) for x in mats if x is not None]
                if mats:
                    S = np.stack([x.T @ x for x in mats])
                    sv = np.stack([x.sum(axis=0) for x in mats])
                    nv = np.array([x.shape[0] for x in mats], dtype=np.float64)
                    stats[r] = (S, sv, nv)
                else:
                    stats[r] = None
            for a in range(len(res_names)):
                for b in range(a + 1, len(res_names)):
                    ra, rb = res_names[a], res_names[b]
                    if stats[ra] is None or stats[rb] is None:
                        continue
                    Sa, sa, na_ = stats[ra]
                    Sb, sb, nb_ = stats[rb]
                    point = bures_fidelity(cov_of(Sa, sa, na_), cov_of(Sb, sb, nb_))
                    rng = np.random.default_rng(stable_seed(cfg.seed, m.label, level, ra, rb))
                    ka, kb = len(na_), len(nb_)
                    boot = []
                    for _ in range(cfg.bootstrap):
                        wa = np.bincount(rng.integers(0, ka, ka),
                                         minlength=ka).astype(np.float64)
                        wb = np.bincount(rng.integers(0, kb, kb),
                                         minlength=kb).astype(np.float64)
                        boot.append(bures_fidelity(cov_of(Sa, sa, na_, wa),
                                                   cov_of(Sb, sb, nb_, wb)))
                    finite = [v for v in boot if np.isfinite(v)]
                    lo = float(np.percentile(finite, 2.5)) if finite else float("nan")
                    hi = float(np.percentile(finite, 97.5)) if finite else float("nan")
                    rows.append({"model": m.label, "family": m.family, "level": level,
                                 "res_a": ra, "res_b": rb, "bures_fidelity": point,
                                 "ci_lo": lo, "ci_hi": hi})
                    if np.isfinite(point):
                        pair_means.append(point)
            done += 1
            LOGGER.info("Invariance: %s L%d complete (%d/%d combinations).",
                        m.label, level, done, n_combo)
        score[m.label] = float(np.mean(pair_means)) if pair_means else float("nan")
    return rows, score


def correlate_with_map_gap(cfg: AnalysisConfig, inv_score: Dict[str, float]) -> Optional[Dict]:
    if not cfg.map_gap_csv or not osp.isfile(cfg.map_gap_csv):
        return None
    from scipy.stats import spearmanr
    gap: Dict[str, float] = {}
    with open(cfg.map_gap_csv, newline="") as f:
        for row in csv.DictReader(f):
            gap[row["label"]] = float(row["map_gap"])
    common = [l for l in inv_score if l in gap and np.isfinite(inv_score[l])]
    if len(common) < 3:
        LOGGER.warning("Fewer than 3 matched models; skipping correlation.")
        return None
    x = [inv_score[l] for l in common]
    y = [gap[l] for l in common]
    rho, p = spearmanr(x, y)
    return {"labels": common, "invariance": x, "map_gap": y,
            "spearman_rho": float(rho), "p_value": float(p)}


# --------------------------------------------------------------------------- #
# Tables and publication figures
# --------------------------------------------------------------------------- #
def _family_color(fam: str) -> str:
    return FAMILY_COLORS.get(fam, "#888888")


def write_tables_and_figures(cfg: AnalysisConfig, cka: Dict, dim_rows: List[Dict],
                             inv_rows: List[Dict], inv_score: Dict[str, float],
                             corr: Optional[Dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    set_publication_style()

    fig_dir = osp.join(cfg.out_dir, "figures")
    tab_dir = osp.join(cfg.out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # ---- CKA matrices ----------------------------------------------------- #
    for key, e in cka.items():
        labels = e["labels"]
        mat = np.array(e["linear_cka"])
        with open(osp.join(tab_dir, f"cka_{key}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow([""] + labels)
            for i, lab in enumerate(labels):
                w.writerow([lab] + [f"{v:.4f}" for v in mat[i]])
        fig, ax = plt.subplots(figsize=(0.62 * len(labels) + 2.6, 0.62 * len(labels) + 2.6))
        im = ax.imshow(np.nan_to_num(mat, nan=0.0), vmin=0, vmax=1, cmap=HEATMAP_CMAP)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            color="white" if mat[i, j] < 0.55 else "black", fontsize=6)
        ax.set_title(f"Debiased linear CKA - {key}")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("CKA")
        save_figure(fig, osp.join(fig_dir, f"cka_{key}"))

    # ---- Dimensionality table --------------------------------------------- #
    if dim_rows:
        with open(osp.join(tab_dir, "dimensionality.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dim_rows[0].keys()))
            w.writeheader()
            for r in dim_rows:
                w.writerow(r)

    # ---- Resolution-invariance table -------------------------------------- #
    if inv_rows:
        with open(osp.join(tab_dir, "resolution_invariance.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(inv_rows[0].keys()))
            w.writeheader()
            for r in inv_rows:
                w.writerow(r)

    # ---- Headline: per-backbone resolution-invariance --------------------- #
    labels = [m.label for m in cfg.models if np.isfinite(inv_score.get(m.label, np.nan))]
    fams = {m.label: m.family for m in cfg.models}
    if labels:
        per_model: Dict[str, List[float]] = {l: [] for l in labels}
        for r in inv_rows:
            if r["model"] in per_model and np.isfinite(r["bures_fidelity"]):
                per_model[r["model"]].append(r["bures_fidelity"])
        means, los, his = [], [], []
        for l in labels:
            m, lo, hi = bootstrap_ci(per_model[l], cfg.bootstrap, stable_seed(cfg.seed, l, "score"))
            means.append(m); los.append(m - lo); his.append(hi - m)
        order = np.argsort(means)[::-1]
        colors = [_family_color(fams[labels[i]]) for i in order]
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(labels)), 4.2))
        ax.bar(range(len(labels)), [means[i] for i in order],
               yerr=[[los[i] for i in order], [his[i] for i in order]],
               color=colors, capsize=3, edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([labels[i] for i in order], rotation=45, ha="right")
        ax.set_ylabel("Cross-resolution Bures fidelity")
        ax.set_title("Representational resolution-invariance by backbone")
        ax.set_ylim(0, 1)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
        ax.legend(handles, list(FAMILY_COLORS.keys()), loc="upper right", ncol=3)
        save_figure(fig, osp.join(fig_dir, "resolution_invariance"))

    # ---- Explanatory: invariance vs mAP gap ------------------------------- #
    if corr is not None:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        cols = [_family_color(fams.get(l, "")) for l in corr["labels"]]
        ax.scatter(corr["invariance"], corr["map_gap"], c=cols, s=55,
                   edgecolor="black", linewidth=0.4, zorder=3)
        for l, xx, yy in zip(corr["labels"], corr["invariance"], corr["map_gap"]):
            ax.annotate(l, (xx, yy), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel("Resolution-invariance score (Bures fidelity)")
        ax.set_ylabel("Cross-resolution mAP gap")
        ax.set_title(f"Spearman \u03c1 = {corr['spearman_rho']:.2f} "
                     f"(p = {corr['p_value']:.3f})")
        save_figure(fig, osp.join(fig_dir, "invariance_vs_mapgap"))


def write_provenance(cfg: AnalysisConfig, extra: Dict) -> None:
    import dataclasses
    from . import __version__
    prov = {"tool": "feature_analysis", "version": __version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": platform.python_version(),
            "config_hash": cfg.config_hash(),
            "config": dataclasses.asdict(cfg)}
    for mod in ("torch", "mmdet"):
        try:
            prov[mod] = __import__(mod).__version__
        except Exception:
            pass
    prov.update(extra)
    with open(osp.join(cfg.out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2, default=str)


def _cka_hash(cfg: AnalysisConfig) -> str:
    import hashlib as _hl
    key = repr([(m.label, m.checkpoint, sorted((m.checkpoints or {}).items()))
                for m in cfg.models]) + \
          repr([(t.name, t.img_dir, t.pattern, t.coco_ann) for t in cfg.tile_sets]) + \
          repr((cfg.seed, cfg.n_tiles, cfg.max_locations_per_tile,
                cfg.fpn_levels, cfg.bootstrap, cfg.rbf_cka))
    return _hl.sha256(key.encode()).hexdigest()[:12]


def phase_analyze(cfg: AnalysisConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    sampled = load_sampled(cfg)
    cka_cache = osp.join(cfg.out_dir, "cka_cache.json")
    cka = None
    if osp.isfile(cka_cache):
        with open(cka_cache, "r") as f:
            payload = json.load(f)
        if payload.get("hash") == _cka_hash(cfg):
            LOGGER.info("Reusing cached CKA results (%s).", cka_cache)
            cka = payload["cka"]
        else:
            LOGGER.info("CKA cache stale (configuration changed); recomputing.")
    if cka is None:
        LOGGER.info("Computing cross-architecture CKA ...")
        cka = analyze_cka(cfg, sampled)
        with open(cka_cache, "w") as f:
            json.dump({"hash": _cka_hash(cfg), "cka": cka}, f)
    LOGGER.info("Computing representational dimensionality ...")
    dim_rows = analyze_dimensionality(cfg, sampled)
    LOGGER.info("Computing resolution-invariance (headline) ...")
    inv_rows, inv_score = analyze_resolution_invariance(cfg, sampled)
    corr = correlate_with_map_gap(cfg, inv_score)
    LOGGER.info("Writing tables and figures ...")
    write_tables_and_figures(cfg, cka, dim_rows, inv_rows, inv_score, corr)
    write_provenance(cfg, {"resolution_invariance_score": inv_score, "spearman": corr})
    LOGGER.info("Analysis complete. Outputs at %s", cfg.out_dir)
