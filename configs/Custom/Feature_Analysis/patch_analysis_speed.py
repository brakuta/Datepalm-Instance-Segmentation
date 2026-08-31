#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot performance patch for feature_analysis/analysis.py.

1. Replaces analyze_resolution_invariance with an exact sufficient-statistics
   implementation (identical estimator and bootstrap draw sequence; the
   bootstrap no longer re-concatenates raw features per resample), reducing
   the stage from days to minutes, with progress logging.
2. Adds CKA result caching to phase_analyze (out_dir/cka_cache.json, guarded
   by a hash of the fields CKA depends on), so re-running analyze skips the
   ~80-minute CKA stage when nothing relevant changed.

Usage:
    python patch_analysis_speed.py \
        configs/Custom/Feature_Analysis/feature_analysis/analysis.py

Idempotent; writes analysis.py.bak2; verifies syntax after patching.
"""
import ast, shutil, sys

OLD_FUNC = 'def analyze_resolution_invariance(cfg: AnalysisConfig,\n                                  sampled: Dict[str, List[str]]) -> Tuple[List[Dict], Dict[str, float]]:\n    """Cross-resolution channel-covariance Bures fidelity per backbone."""\n    res_names = list(sampled.keys())\n    rows: List[Dict] = []\n    score: Dict[str, float] = {}\n    for m in cfg.models:\n        pair_means: List[float] = []\n        for level in cfg.fpn_levels:\n            pooled: Dict[str, Optional[np.ndarray]] = {}\n            tilewise: Dict[str, List[np.ndarray]] = {}\n            for r in res_names:\n                mats = [load_matrix(cfg, m.label, r, s, level) for s in sampled[r]]\n                mats = [x for x in mats if x is not None]\n                tilewise[r] = mats\n                pooled[r] = channel_covariance(np.concatenate(mats, 0)) if mats else None\n            for a in range(len(res_names)):\n                for b in range(a + 1, len(res_names)):\n                    ra, rb = res_names[a], res_names[b]\n                    if pooled[ra] is None or pooled[rb] is None:\n                        continue\n                    point = bures_fidelity(pooled[ra], pooled[rb])\n                    rng = np.random.default_rng(stable_seed(cfg.seed, m.label, level, ra, rb))\n                    boot = []\n                    na, nb = len(tilewise[ra]), len(tilewise[rb])\n                    if na and nb:\n                        for _ in range(cfg.bootstrap):\n                            ia = rng.integers(0, na, na)\n                            ib = rng.integers(0, nb, nb)\n                            ca = channel_covariance(np.concatenate([tilewise[ra][k] for k in ia], 0))\n                            cb = channel_covariance(np.concatenate([tilewise[rb][k] for k in ib], 0))\n                            boot.append(bures_fidelity(ca, cb))\n                        lo = float(np.percentile(boot, 2.5))\n                        hi = float(np.percentile(boot, 97.5))\n                    else:\n                        lo = hi = float("nan")\n                    rows.append({"model": m.label, "family": m.family, "level": level,\n                                 "res_a": ra, "res_b": rb, "bures_fidelity": point,\n                                 "ci_lo": lo, "ci_hi": hi})\n                    if np.isfinite(point):\n                        pair_means.append(point)\n        score[m.label] = float(np.mean(pair_means)) if pair_means else float("nan")\n    return rows, score'
NEW_FUNC = 'def analyze_resolution_invariance(cfg: AnalysisConfig,\n                                  sampled: Dict[str, List[str]]) -> Tuple[List[Dict], Dict[str, float]]:\n    """Cross-resolution channel-covariance Bures fidelity per backbone.\n\n    Sufficient-statistics implementation: the covariance of any concatenation\n    of per-tile matrices is computed exactly from per-tile Gram matrices,\n    column sums, and row counts, so each bootstrap resample costs one weighted\n    sum of (k, 256, 256) statistics plus one Bures evaluation, instead of\n    re-concatenating ~80k feature rows and recomputing the covariance from\n    raw data. The estimator and the bootstrap draw sequence are identical to\n    the reference implementation; statistics are accumulated in float64.\n    """\n    res_names = list(sampled.keys())\n    rows: List[Dict] = []\n    score: Dict[str, float] = {}\n    n_combo = len(cfg.models) * len(cfg.fpn_levels)\n    done = 0\n\n    def cov_of(S, s, n, w=None):\n        if w is None:\n            Ssum, ssum, N = S.sum(axis=0), s.sum(axis=0), float(n.sum())\n        else:\n            Ssum = np.tensordot(w, S, axes=1)\n            ssum = w @ s\n            N = float(w @ n)\n        if N <= 1.0:\n            return None\n        return (Ssum - np.outer(ssum, ssum) / N) / (N - 1.0)\n\n    for m in cfg.models:\n        pair_means: List[float] = []\n        for level in cfg.fpn_levels:\n            stats: Dict[str, Optional[tuple]] = {}\n            for r in res_names:\n                mats = [load_matrix(cfg, m.label, r, s, level) for s in sampled[r]]\n                mats = [x.astype(np.float64) for x in mats if x is not None]\n                if mats:\n                    S = np.stack([x.T @ x for x in mats])\n                    sv = np.stack([x.sum(axis=0) for x in mats])\n                    nv = np.array([x.shape[0] for x in mats], dtype=np.float64)\n                    stats[r] = (S, sv, nv)\n                else:\n                    stats[r] = None\n            for a in range(len(res_names)):\n                for b in range(a + 1, len(res_names)):\n                    ra, rb = res_names[a], res_names[b]\n                    if stats[ra] is None or stats[rb] is None:\n                        continue\n                    Sa, sa, na_ = stats[ra]\n                    Sb, sb, nb_ = stats[rb]\n                    point = bures_fidelity(cov_of(Sa, sa, na_), cov_of(Sb, sb, nb_))\n                    rng = np.random.default_rng(stable_seed(cfg.seed, m.label, level, ra, rb))\n                    ka, kb = len(na_), len(nb_)\n                    boot = []\n                    for _ in range(cfg.bootstrap):\n                        wa = np.bincount(rng.integers(0, ka, ka),\n                                         minlength=ka).astype(np.float64)\n                        wb = np.bincount(rng.integers(0, kb, kb),\n                                         minlength=kb).astype(np.float64)\n                        boot.append(bures_fidelity(cov_of(Sa, sa, na_, wa),\n                                                   cov_of(Sb, sb, nb_, wb)))\n                    finite = [v for v in boot if np.isfinite(v)]\n                    lo = float(np.percentile(finite, 2.5)) if finite else float("nan")\n                    hi = float(np.percentile(finite, 97.5)) if finite else float("nan")\n                    rows.append({"model": m.label, "family": m.family, "level": level,\n                                 "res_a": ra, "res_b": rb, "bures_fidelity": point,\n                                 "ci_lo": lo, "ci_hi": hi})\n                    if np.isfinite(point):\n                        pair_means.append(point)\n            done += 1\n            LOGGER.info("Invariance: %s L%d complete (%d/%d combinations).",\n                        m.label, level, done, n_combo)\n        score[m.label] = float(np.mean(pair_means)) if pair_means else float("nan")\n    return rows, score'
OLD_PHASE = 'def phase_analyze(cfg: AnalysisConfig) -> None:\n    os.makedirs(cfg.out_dir, exist_ok=True)\n    sampled = load_sampled(cfg)\n    LOGGER.info("Computing cross-architecture CKA ...")\n    cka = analyze_cka(cfg, sampled)\n    LOGGER.info("Computing representational dimensionality ...")'
NEW_PHASE = 'def _cka_hash(cfg: AnalysisConfig) -> str:\n    import hashlib as _hl\n    key = repr([(m.label, m.checkpoint, sorted((m.checkpoints or {}).items()))\n                for m in cfg.models]) + \\\n          repr([(t.name, t.img_dir, t.pattern, t.coco_ann) for t in cfg.tile_sets]) + \\\n          repr((cfg.seed, cfg.n_tiles, cfg.max_locations_per_tile,\n                cfg.fpn_levels, cfg.bootstrap, cfg.rbf_cka))\n    return _hl.sha256(key.encode()).hexdigest()[:12]\n\n\ndef phase_analyze(cfg: AnalysisConfig) -> None:\n    os.makedirs(cfg.out_dir, exist_ok=True)\n    sampled = load_sampled(cfg)\n    cka_cache = osp.join(cfg.out_dir, "cka_cache.json")\n    cka = None\n    if osp.isfile(cka_cache):\n        with open(cka_cache, "r") as f:\n            payload = json.load(f)\n        if payload.get("hash") == _cka_hash(cfg):\n            LOGGER.info("Reusing cached CKA results (%s).", cka_cache)\n            cka = payload["cka"]\n        else:\n            LOGGER.info("CKA cache stale (configuration changed); recomputing.")\n    if cka is None:\n        LOGGER.info("Computing cross-architecture CKA ...")\n        cka = analyze_cka(cfg, sampled)\n        with open(cka_cache, "w") as f:\n            json.dump({"hash": _cka_hash(cfg), "cka": cka}, f)\n    LOGGER.info("Computing representational dimensionality ...")'  # noqa: leakscan (literal \n in a source string, not a path)

def main():
    if len(sys.argv) != 2:
        print(__doc__); return 2
    path = sys.argv[1]
    src = open(path).read()
    if "sufficient-statistics implementation" in src.lower() or "cka_cache" in src:
        print("Already patched; nothing to do."); return 0
    n = 0
    if OLD_FUNC in src:
        src = src.replace(OLD_FUNC, NEW_FUNC); n += 1
    else:
        print("ERROR: analyze_resolution_invariance did not match verbatim; "
              "no changes written."); return 1
    if OLD_PHASE in src:
        src = src.replace(OLD_PHASE, NEW_PHASE); n += 1
    else:
        print("WARNING: phase_analyze header did not match; CKA caching "
              "skipped (invariance fix still applied).")
    ast.parse(src)
    shutil.copyfile(path, path + ".bak2")
    open(path, "w").write(src)
    print(f"Applied {n} patch section(s); backup at {path}.bak2; syntax OK.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
