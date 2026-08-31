#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot patcher for feature_analysis/analysis.py.

Applies two changes and nothing else:

1. Records the feature-grid shape in the location index (required by the
   `deep` phase to map COCO ground truth onto cached feature locations).
2. Replaces `phase_extract` with a version that resolves the checkpoint per
   tile set via `ModelSpec.checkpoint_for`, so per-domain best weights
   (best_UAV_*, best_GE_*) feed the corresponding tile sets. Tile sets are
   grouped by resolved checkpoint so each distinct weight state is built
   exactly once, preserving the one-detector-resident memory constraint.
   The mapping actually used is recorded to `<cache_dir>/_checkpoints.json`
   for provenance.

Usage (from the mmdetection root):
    python configs/Custom/Feature_Analysis/patch_analysis.py \
        configs/Custom/Feature_Analysis/feature_analysis/analysis.py

A backup is written alongside as analysis.py.bak. The patch is idempotent:
re-running on an already-patched file is a no-op.
"""

from __future__ import annotations

import shutil
import sys

NEW_FUNC = '''def phase_extract(cfg: AnalysisConfig) -> None:
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
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    src = open(path, "r").read()

    if "_checkpoints.json" in src:
        print("Already patched; nothing to do.")
        return 0

    start = src.find("def phase_extract(")
    if start < 0:
        print("ERROR: phase_extract not found; apply the replacement manually "
              "using the function body embedded in this script.")
        return 1
    # The function ends at the next top-level comment banner or def.
    tail = src[start:]
    end_markers = ["\n# ----", "\ndef analyze_cka("]
    ends = [tail.find(m) for m in end_markers if tail.find(m) > 0]
    if not ends:
        print("ERROR: could not delimit phase_extract; patch manually.")
        return 1
    end = start + min(ends)

    shutil.copyfile(path, path + ".bak")
    patched = src[:start] + NEW_FUNC + "\n" + src[end:].lstrip("\n")
    open(path, "w").write(patched)

    import ast
    ast.parse(patched)  # fail loudly if the result is not valid Python
    print(f"Patched {path} (backup at {path}.bak); syntax verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
