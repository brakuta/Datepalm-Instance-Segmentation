#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Showcase- and composite-tile selection for the feature-analysis suite.

Representative tiles for the qualitative figures (multiscale grids, composite
recipes, scale-anatomy) should exhibit moderate crown density and typical
scene structure. This utility ranks the tiles of every configured tile set by
proximity to the MEDIAN per-tile palm count (computed from the COCO
annotations) and either lists the candidates (default) or rewrites the
configuration in place (--apply), replacing:

  * every REPLACE_* entry in `showcase_tiles`,
  * every REPLACE_* `tile` in `composite_figures[*].columns`
    (the tile set is inferred from the placeholder path),
  * every REPLACE_* `source_tile` in degrade-type composite recipes.

Median-density selection is a deterministic, defensible criterion; the ranked
listing permits manual substitution where visual inspection suggests a better
exemplar (e.g. to avoid shadowed or cloud-affected scenes).

Usage (from the repository root):
    # list the 5 median-density candidates per tile set
    python configs/Custom/Feature_Analysis/pick_showcase_tiles.py \
        --config configs/Custom/Feature_Analysis/config_feature_analysis.json

    # fill all placeholders with the top candidate per tile set (writes .bak)
    python ... --config <same> --apply
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import shutil
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple


def palm_counts(coco_ann: str) -> Dict[str, int]:
    with open(coco_ann, "r") as f:
        coco = json.load(f)
    id2stem = {im["id"]: osp.splitext(osp.basename(im["file_name"]))[0]
               for im in coco["images"]}
    counts = {s: 0 for s in id2stem.values()}
    for ann in coco.get("annotations", []):
        stem = id2stem.get(ann["image_id"])
        if stem is not None:
            counts[stem] += 1
    return counts


def ranked_tiles(ts: Dict) -> List[Tuple[str, int, float]]:
    """[(path, count, |count - median|)] sorted by distance to the median."""
    files = sorted(glob(osp.join(ts["img_dir"], ts.get("pattern", "*.png"))))
    if not files:
        print(f"[warn ] {ts['name']}: no tiles in {ts['img_dir']}")
        return []
    ann = ts.get("coco_ann")
    if not ann or not osp.isfile(ann):
        print(f"[warn ] {ts['name']}: no COCO annotations; ranking by name only.")
        return [(f, -1, 0.0) for f in files]
    counts = palm_counts(ann)
    keyed = [(f, counts.get(osp.splitext(osp.basename(f))[0], 0)) for f in files]
    vals = sorted(c for _, c in keyed)
    median = vals[len(vals) // 2]
    ranked = sorted(((f, c, abs(c - median)) for f, c in keyed),
                    key=lambda t: (t[2], t[0]))
    return ranked


def infer_set(path: str, tile_sets: List[Dict]) -> Optional[Dict]:
    """Match a REPLACE_* placeholder path to a tile set by shared path
    components (dataset folder names are the discriminating components)."""
    parts = set(p for p in path.split("/") if p)
    best, score = None, 0
    for ts in tile_sets:
        s = len(parts & set(p for p in ts["img_dir"].split("/") if p))
        if s > score:
            best, score = ts, s
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Rewrite the config, filling all REPLACE_* entries "
                         "with the top median-density candidate.")
    ap.add_argument("--top", type=int, default=5,
                    help="Number of candidates to list per tile set.")
    args = ap.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = json.load(f)
    tile_sets = cfg.get("tile_sets", [])
    choice: Dict[str, str] = {}

    for ts in tile_sets:
        ranked = ranked_tiles(ts)
        if not ranked:
            continue
        choice[ts["name"]] = ranked[0][0]
        print(f"\n{ts['name']} — median-density candidates:")
        for f, c, _ in ranked[:args.top]:
            print(f"    {osp.basename(f):<40s} palms={c}")

    if not args.apply:
        print("\n(Preview only; re-run with --apply to fill the placeholders.)")
        return 0

    changed = 0
    sc = cfg.get("showcase_tiles") or {}
    for set_name, tiles in sc.items():
        for i, t in enumerate(tiles):
            if "REPLACE" in t and set_name in choice:
                sc[set_name][i] = choice[set_name]; changed += 1
    for rec in cfg.get("composite_figures") or []:
        for col in rec.get("columns") or []:
            if "REPLACE" in col.get("tile", ""):
                ts = infer_set(col["tile"], tile_sets)
                if ts and ts["name"] in choice:
                    col["tile"] = choice[ts["name"]]; changed += 1
                else:
                    print(f"[warn ] could not infer tile set for column "
                          f"'{col.get('label')}' in recipe '{rec.get('name')}'.")
        st = rec.get("source_tile", "")
        if "REPLACE" in st:
            ts = infer_set(st, tile_sets)
            if ts and ts["name"] in choice:
                rec["source_tile"] = choice[ts["name"]]; changed += 1
            else:
                print(f"[warn ] could not infer tile set for source_tile "
                      f"in recipe '{rec.get('name')}'.")

    shutil.copyfile(args.config, args.config + ".bak")
    with open(args.config, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nFilled {changed} placeholder(s); backup at {args.config}.bak")
    if changed == 0:
        print("No REPLACE_* placeholders remained; nothing changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
