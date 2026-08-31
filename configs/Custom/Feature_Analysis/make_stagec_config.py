#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic `models`-block generation for the feature-analysis suite from a
Stage C `work_dirs` tree.

Stage C training (EarlyStopping on per-dataset ``segm_mAP_50``) leaves several
checkpoint types in each run directory::

    best_UAV_segm_mAP_50_iter_85001.pth   # best on the UAV validation set
    best_GE_segm_mAP_50_iter_85001.pth    # best on the GE validation set
    iter_110000.pth, iter_115000.pth, ... # periodic iterates
    last_checkpoint                        # text pointer to the latest iterate

This script scans ``--work-root``, identifies each backbone from the run
directory name, resolves the appropriate checkpoint(s) under an explicit
``--policy``, and rewrites ONLY the ``models`` block of the target
configuration JSON (all other fields -- tile sets, showcase tiles, composite
recipes -- are preserved).

Checkpoint policies
-------------------
auto        If every ``best_*`` checkpoint of a run points to the SAME
            iteration (the common case; e.g. both bests at iter_85001), that
            single file is used. Otherwise the run falls back to
            ``last_checkpoint`` and a NOTE is printed, because no single
            "best" is defined. This is the default and the recommended
            policy for the representation analysis, which requires one
            canonical weight state per model.
per-domain  As ``auto`` for the default ``checkpoint`` field, plus a
            ``checkpoints`` mapping {tile_set -> best_<TAG> file} built from
            ``--map`` entries (e.g. ``--map UAV_5cm=UAV --map GE_15cm=GE``).
            Tile sets without a mapped tag fall back to the default. Use for
            evaluation-aligned, WITHIN-resolution analyses only (see the
            methodological note in the integration guide): cross-resolution
            invariance comparisons are only interpretable within a single
            checkpoint.
best:<TAG>  Always the ``best_<TAG>_segm_mAP_50_iter_*.pth`` file
            (e.g. ``best:UAV``); errors if absent.
last        Always the ``last_checkpoint`` pointer target.
iter:<N>    Always ``iter_<N>.pth``; errors if absent.

Usage (from the mmdetection root)
---------------------------------
    python configs/Custom/Feature_Analysis/make_stagec_config.py \
        --work-root  /root/work_dirs/Stage_C \
        --config-root configs/Custom/maskrcnn_palm_stagec \
        --base configs/Custom/Feature_Analysis/config_feature_analysis.json \
        --out  configs/Custom/Feature_Analysis/config_feature_analysis.json \
        --policy auto

    # evaluation-aligned variant (separate cache/out dirs advised):
    python ... --policy per-domain --map UAV_5cm=UAV --map GE_15cm=GE
"""

from __future__ import annotations

import argparse
import glob
import json
import os.path as osp
import re
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Backbone registry: work_dir stem -> (display label, family).
# Labels follow the locked manuscript conventions (SSM cohort grouped under
# 'Mamba' for the family palette; MambaOut is reported within that cohort as
# the SSM-ablation control). Edit here if the reported grouping changes.
# ---------------------------------------------------------------------------
REGISTRY: Dict[str, Tuple[str, str]] = {
    "r50":             ("ResNet-50",       "CNN"),
    "r101":            ("ResNet-101",      "CNN"),
    "convnext_t":      ("ConvNeXt-T",      "CNN"),
    "swin_s":          ("Swin-S",          "Transformer"),
    "pvtv2_b2":        ("PVTv2-B2",        "Transformer"),
    "vmamba_s":        ("VMamba-S",        "Mamba"),
    "spatialmamba_s":  ("SpatialMamba-S",  "Mamba"),
    "groupmamba_s":    ("GroupMamba-S",    "Mamba"),
    "efficientvmamba_b": ("EfficientVMamba-B", "Mamba"),
    "mambavision_s":   ("MambaVision-S",   "Mamba"),
    "mambaout_s":      ("MambaOut-S",      "Mamba"),
}

_DIR_RE = re.compile(r"^maskrcnn_(?P<stem>.+)_stagec$")
_BEST_RE = re.compile(r"^best_(?P<tag>.+?)_segm_mAP_50_iter_(?P<it>\d+)\.pth$")
_ITER_RE = re.compile(r"^iter_(?P<it>\d+)\.pth$")


def _guess(stem: str) -> Tuple[str, str]:
    """Fallback label/family for run directories absent from the registry."""
    label = stem.replace("_", "-").title().replace("-S", "-S").replace("-B", "-B")
    low = stem.lower()
    if "mamba" in low or "vim" in low or "ssm" in low:
        family = "Mamba"
    elif any(k in low for k in ("swin", "pvt", "vit", "deit", "transformer")):
        family = "Transformer"
    else:
        family = "CNN"
    return label, family


def scan_run(run_dir: str) -> Dict:
    """Inventory one run directory: best_* files, periodic iterates, last."""
    best: Dict[str, Tuple[int, str]] = {}
    iters: Dict[int, str] = {}
    for p in glob.glob(osp.join(run_dir, "*.pth")):
        name = osp.basename(p)
        mb = _BEST_RE.match(name)
        if mb:
            best[mb.group("tag")] = (int(mb.group("it")), p)
            continue
        mi = _ITER_RE.match(name)
        if mi:
            iters[int(mi.group("it"))] = p
    last: Optional[str] = None
    lp = osp.join(run_dir, "last_checkpoint")
    if osp.isfile(lp):
        with open(lp, "r") as f:
            target = f.read().strip()
        # The pointer stores the absolute path from TRAINING time, which may
        # reference a different root (e.g. /root/work_dirs vs the workspace
        # tree). Always prefer the copy inside THIS run directory; use the
        # absolute pointer only as a fallback, with a warning.
        local = osp.join(run_dir, osp.basename(target))
        if osp.isfile(local):
            last = local
        elif osp.isabs(target) and osp.isfile(target):
            last = target
            print(f"[warn ] {osp.basename(run_dir)}: last_checkpoint resolves "
                  f"OUTSIDE the run directory ({target}); no local copy found.")
    if last is None and iters:
        last = iters[max(iters)]
    return {"best": best, "iters": iters, "last": last}


def resolve_default(inv: Dict, policy: str, run_dir: str) -> Tuple[str, str]:
    """Return (checkpoint_path, note) for the default `checkpoint` field."""
    best, last = inv["best"], inv["last"]
    if policy.startswith("best:"):
        tag = policy.split(":", 1)[1]
        if tag not in best:
            raise FileNotFoundError(f"{run_dir}: no best_{tag}_segm_mAP_50 checkpoint.")
        return best[tag][1], f"best:{tag} (iter {best[tag][0]})"
    if policy.startswith("iter:"):
        it = int(policy.split(":", 1)[1])
        if it not in inv["iters"]:
            raise FileNotFoundError(f"{run_dir}: iter_{it}.pth not found.")
        return inv["iters"][it], f"iter {it}"
    if policy == "last":
        if last is None:
            raise FileNotFoundError(f"{run_dir}: no last_checkpoint or iter_*.pth.")
        return last, "last"
    if policy in ("best-latest", "per-domain"):
        # Only best_* checkpoints are admissible; periodic iterates and the
        # last_checkpoint pointer are never selected under these policies.
        if not best:
            raise FileNotFoundError(f"{run_dir}: no best_* checkpoints found "
                                    f"(required by policy '{policy}').")
        tag, (it, path) = max(best.items(), key=lambda kv: (kv[1][0], kv[0]))
        return path, f"best-latest ({tag} @ iter {it})"
    # 'auto' and the default half of 'per-domain'
    if best:
        its = {it for it, _ in best.values()}
        if len(its) == 1:
            it = next(iter(its))
            path = next(p for i, p in best.values() if i == it)
            tags = "/".join(sorted(best))
            return path, f"consensus best ({tags}) @ iter {it}"
        if last is None:
            raise FileNotFoundError(f"{run_dir}: divergent bests and no last checkpoint.")
        detail = ", ".join(f"{t}@{i}" for t, (i, _) in sorted(best.items()))
        return last, f"NOTE divergent bests ({detail}) -> last"
    if last is None:
        raise FileNotFoundError(f"{run_dir}: no checkpoints found.")
    return last, "no best_* found -> last"


def build_models(work_root: str, config_root: str, policy: str,
                 domain_map: Dict[str, str],
                 overrides: Optional[Dict[str, str]] = None) -> List[Dict]:
    overrides = dict(overrides or {})
    models: List[Dict] = []
    run_dirs = sorted(d for d in glob.glob(osp.join(work_root, "*"))
                      if osp.isdir(d))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories under {work_root}.")
    for run_dir in run_dirs:
        name = osp.basename(run_dir)
        m = _DIR_RE.match(name)
        if not m:
            print(f"[skip ] {name}: does not match maskrcnn_*_stagec.")
            continue
        stem = m.group("stem")
        label, family = REGISTRY.get(stem, (None, None))
        if label is None:
            label, family = _guess(stem)
            print(f"[warn ] {name}: '{stem}' not in registry; "
                  f"using label='{label}', family='{family}' (verify).")
        config = osp.join(config_root, f"{name}.py")
        if not osp.isfile(config):
            print(f"[warn ] {label}: config not found at {config}; "
                  f"entry emitted anyway (validation will fail until fixed).")
        inv = scan_run(run_dir)
        if label in overrides:
            ckpt = overrides.pop(label)
            if not osp.isfile(ckpt):
                raise FileNotFoundError(f"--override checkpoint not found: {ckpt}")
            note = "manual override"
        else:
            try:
                ckpt, note = resolve_default(inv, policy, run_dir)
            except FileNotFoundError as exc:
                print(f"[skip ] {label}: {exc}")
                continue
        entry: Dict = {"label": label, "family": family,
                       "config": config, "checkpoint": ckpt}
        if policy == "per-domain" and domain_map:
            mapping: Dict[str, str] = {}
            for set_name, tag in domain_map.items():
                if tag in inv["best"]:
                    mapping[set_name] = inv["best"][tag][1]
                else:
                    print(f"[warn ] {label}: no best_{tag} checkpoint; "
                          f"{set_name} falls back to the default.")
            if mapping:
                entry["checkpoints"] = mapping
        models.append(entry)
        extra = (f" + per-domain {sorted(entry.get('checkpoints', {}))}"
                 if "checkpoints" in entry else "")
        print(f"[model] {label:<20s} <- {note}{extra}")
    for missed in overrides:
        print(f"[warn ] --override label '{missed}' matched no model.")
    if not models:
        raise RuntimeError("No usable Stage C runs were found.")
    return models


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-root", default="/root/work_dirs/Stage_C")
    ap.add_argument("--config-root",
                    default="configs/Custom/maskrcnn_palm_stagec")
    ap.add_argument("--base", required=True,
                    help="Existing config JSON; all non-model fields are kept.")
    ap.add_argument("--out", required=True,
                    help="Output path (may equal --base to update in place).")
    ap.add_argument("--policy", default="auto",
                    help="best-latest | per-domain | auto | best:<TAG> | last | iter:<N>")
    ap.add_argument("--map", action="append", default=[],
                    metavar="TILESET=TAG",
                    help="per-domain mapping, e.g. UAV_5cm=UAV (repeatable).")
    ap.add_argument("--override", action="append", default=[],
                    metavar="LABEL=CHECKPOINT",
                    help="force a specific default checkpoint for one model, "
                         "e.g. --override 'MambaVision-S=/path/to/best.pth' "
                         "(repeatable; applied after policy resolution).")
    args = ap.parse_args(argv)

    domain_map: Dict[str, str] = {}
    for item in args.map:
        if "=" not in item:
            ap.error(f"--map expects TILESET=TAG, got '{item}'.")
        k, v = item.split("=", 1)
        domain_map[k.strip()] = v.strip()
    if args.policy == "per-domain" and not domain_map:
        ap.error("--policy per-domain requires at least one --map entry.")

    overrides: Dict[str, str] = {}
    for item in args.override:
        if "=" not in item:
            ap.error(f"--override expects LABEL=CHECKPOINT, got '{item}'.")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()

    with open(args.base, "r") as f:
        cfg = json.load(f)
    models = build_models(args.work_root, args.config_root,
                          args.policy, domain_map, overrides)
    known_sets = {t["name"] for t in cfg.get("tile_sets", [])}
    for mdl in models:
        for set_name in mdl.get("checkpoints", {}):
            if set_name not in known_sets:
                print(f"[warn ] {mdl['label']}: mapped tile set '{set_name}' "
                      f"is not defined in tile_sets.")
    # sanity: flag any checkpoint resolving outside the scanned work root
    wr = osp.abspath(args.work_root)
    for mdl in models:
        paths = [mdl["checkpoint"]] + list(mdl.get("checkpoints", {}).values())
        for pth in paths:
            if not osp.abspath(pth).startswith(wr):
                print(f"[warn ] {mdl['label']}: checkpoint outside --work-root: {pth}")

    cfg["models"] = models
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nWrote {len(models)} model entries to {args.out}")
    print("Policy:", args.policy,
          "| Non-model fields preserved from", args.base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
