#!/usr/bin/env python3
# ==========================================================================
# tools_staged/build_budget_manifests.py   (Stage D, v1)
# --------------------------------------------------------------------------
# Builds the NESTED, density-stratified annotation-budget subsets for the
# Stage D WV-3 transfer experiment and writes them as standalone COCO files:
#
#   train_sat_b05.json  (5 %)   subset of
#   train_sat_b10.json  (10 %)  subset of
#   train_sat_b25.json  (25 %)  subset of
#   train_sat_b50.json  (50 %)  subset of
#   train_sat.json      (100 %, untouched original)
#
# Stratification: tiles are ranked by per-tile palm count and split into
# QUINTILES; each budget samples (budget x stratum size) tiles per stratum,
# so every budget spans the full planting-density range. Nesting is enforced
# top-down (b50 sampled from full, b25 from b50, ...), so a larger budget
# strictly contains every smaller one -- the budget curve is monotone in
# information.
#
# DETERMINISTIC: seed fixed (42). Run ONCE, commit the manifest JSON, and
# never regenerate after the first fine-tuning run has used the subsets.
#
# Usage:
#   python tools_staged/build_budget_manifests.py \
#       --ann /workspace/datasets/COCO/Sat_30cm/Annotations/train_sat.json
# ==========================================================================
import argparse
import copy
import json
import os
import random
from collections import defaultdict

BUDGETS = [0.50, 0.25, 0.10, 0.05]          # built top-down for nesting
TAGS    = {0.50: 'b50', 0.25: 'b25', 0.10: 'b10', 0.05: 'b05'}
N_STRATA = 5
SEED = 42


def stratify(images, counts, n_strata):
    """Rank tiles by palm count; return list of strata (lists of image ids)."""
    ranked = sorted(images, key=lambda im: counts[im['id']])
    strata, k = [], len(ranked) // n_strata
    for i in range(n_strata):
        lo = i * k
        hi = (i + 1) * k if i < n_strata - 1 else len(ranked)
        strata.append([im['id'] for im in ranked[lo:hi]])
    return strata


def sample_nested(parent_ids_by_stratum, frac_of_full, full_sizes, rng):
    """Sample per stratum from the parent so global size = frac x full."""
    picked = {}
    for s, parent in parent_ids_by_stratum.items():
        target = max(1, round(frac_of_full * full_sizes[s]))
        target = min(target, len(parent))
        picked[s] = sorted(rng.sample(parent, target))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True,
                    help='Path to the full train_sat.json')
    args = ap.parse_args()

    rng = random.Random(SEED)
    coco = json.load(open(args.ann))
    out_dir = os.path.dirname(args.ann)

    counts = defaultdict(int)
    for a in coco['annotations']:
        counts[a['image_id']] += 1
    for im in coco['images']:
        counts.setdefault(im['id'], 0)

    strata = stratify(coco['images'], counts, N_STRATA)
    full_sizes = {s: len(ids) for s, ids in enumerate(strata)}
    parent = {s: list(ids) for s, ids in enumerate(strata)}

    manifest = {
        'seed': SEED, 'n_strata': N_STRATA,
        'source': os.path.basename(args.ann),
        'n_images_full': len(coco['images']),
        'stratum_palm_count_ranges': [
            [min(counts[i] for i in ids), max(counts[i] for i in ids)]
            for ids in strata],
        'budgets': {},
    }

    for frac in BUDGETS:                      # 0.50 -> 0.25 -> 0.10 -> 0.05
        picked = sample_nested(parent, frac, full_sizes, rng)
        ids = sorted(i for s in picked.values() for i in s)
        idset = set(ids)

        sub = copy.deepcopy({k: v for k, v in coco.items()
                             if k not in ('images', 'annotations')})
        sub['images'] = [im for im in coco['images'] if im['id'] in idset]
        sub['annotations'] = [a for a in coco['annotations']
                              if a['image_id'] in idset]

        tag = TAGS[frac]
        out = os.path.join(out_dir, f'train_sat_{tag}.json')
        json.dump(sub, open(out, 'w'))
        manifest['budgets'][tag] = {
            'fraction': frac,
            'n_images': len(sub['images']),
            'n_annotations': len(sub['annotations']),
            'image_ids': ids,
        }
        print(f'{tag}: {len(sub["images"]):4d} tiles, '
              f'{len(sub["annotations"]):6d} palms -> {out}')

        parent = picked                       # next (smaller) budget nests here

    mpath = os.path.join(out_dir, 'budget_manifest_stageD.json')
    json.dump(manifest, open(mpath, 'w'), indent=2)
    print(f'manifest -> {mpath}')

    # nesting check
    prev = None
    for tag in ['b05', 'b10', 'b25', 'b50']:
        cur = set(manifest['budgets'][tag]['image_ids'])
        assert prev is None or prev <= cur, f'nesting violated at {tag}'
        prev = cur
    print('nesting verified: b05 < b10 < b25 < b50 < full')


if __name__ == '__main__':
    main()
