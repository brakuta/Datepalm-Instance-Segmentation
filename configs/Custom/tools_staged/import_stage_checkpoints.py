#!/usr/bin/env python3
# ==========================================================================
# tools_staged/import_stage_checkpoints.py   (Stage D, v3 checklist #3)
# --------------------------------------------------------------------------
# Persist the Stage C real-data priors and the GE-30sim simulation priors
# from the EPHEMERAL container overlay (/root/work_dirs) onto the persistent
# Windows-backed mount (/workspace). /root is reclaimed when the container is
# recreated -- these checkpoints are weeks of GPU time and must not live only
# there. Run this BEFORE any container teardown.
#
# WHAT IT COPIES (discovery-based, no hard-coded iteration numbers):
#   Stage C priors   : <src>/Stage_C/maskrcnn_*/best_*.pth        (best_GE + best_UAV)
#   GE-30sim priors  : <src>/Stage_D/*_ge30sim_stage1/best_*.pth  (best_coco_*)
# The bulky full iter_*.pth (optimiser state) are NOT copied: only the
# selection-best checkpoints, which are what `load_from` needs as a prior.
#
# HOW IT COPIES:
#   * Follows symlinks (materialises the real file, == cp -L).
#   * Run-prefixed flat names so provenance is unambiguous:
#       stage_c/maskrcnn_pvtv2_b2_stagec__best_GE_segm_mAP_50_iter_65001.pth
#       ge30sim/spatialmamba_ge30sim_stage1__best_coco_segm_mAP_50_iter_34000.pth
#   * Idempotent: skips a file already present at the destination with a
#     matching SHA-256 (re-run freely; only new/changed files are copied).
#   * Writes checkpoints_manifest.json (SHA-256 + size + source + mtime) so
#     you can prove exactly which weight state each reported number used.
#
# USAGE (from inside the container, mmdetection root):
#   python tools_staged/import_stage_checkpoints.py            # copy + manifest
#   python tools_staged/import_stage_checkpoints.py --dry-run  # preview only
#   python tools_staged/import_stage_checkpoints.py \
#       --src-root /root/work_dirs \
#       --dest /workspace/mmdetection/checkpoints_imported
# ==========================================================================

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import os.path as osp
import shutil
import sys

# (label, category subdir, glob relative to src-root) discovery rules.
DISCOVERY = [
    ("Stage C prior",  "stage_c", "Stage_C/maskrcnn_*/best_*.pth"),
    ("GE-30sim prior", "ge30sim", "Stage_D/*_ge30sim_stage1/best_*.pth"),
]

CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def human(nbytes: int) -> str:
    x = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if x < 1024 or unit == "GiB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} GiB"


def run_name(pth: str) -> str:
    """The immediate parent dir name is the run identity."""
    return osp.basename(osp.dirname(pth))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-root", default="/root/work_dirs",
                    help="Root that holds Stage_C/ and Stage_D/ (default: /root/work_dirs).")
    ap.add_argument("--dest", default="/workspace/mmdetection/checkpoints_imported",
                    help="Persistent destination directory.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be copied; write nothing.")
    args = ap.parse_args()

    if not osp.isdir(args.src_root):
        print(f"ERROR: source root not found: {args.src_root}", file=sys.stderr)
        return 2

    # Discover.
    found = []  # (label, category, src)
    for label, category, pattern in DISCOVERY:
        for src in sorted(glob.glob(osp.join(args.src_root, pattern))):
            if osp.isfile(src):
                found.append((label, category, src))

    if not found:
        print(f"No checkpoints matched under {args.src_root}. Nothing to do.")
        return 0

    # Guard against silently importing from a stale corpus.
    for _, _, src in found:
        real = osp.realpath(src)
        if "STALE" in real.upper():
            print(f"WARNING: source resolves into a STALE path -> {real}",
                  file=sys.stderr)

    print(f"Discovered {len(found)} checkpoint(s) under {args.src_root}")
    print(f"Destination: {args.dest}{'  (DRY RUN)' if args.dry_run else ''}\n")

    manifest = []
    copied = skipped = 0
    total_copied_bytes = 0

    for label, category, src in found:
        real = osp.realpath(src)
        size = os.stat(real).st_size
        mtime = _dt.datetime.utcfromtimestamp(
            os.stat(real).st_mtime).strftime("%Y-%m-%dT%H:%M:%SZ")
        dest_name = f"{run_name(src)}__{osp.basename(src)}"
        dest_dir = osp.join(args.dest, category)
        dest_path = osp.join(dest_dir, dest_name)

        src_sha = sha256_of(real)

        action = "COPY"
        if osp.isfile(dest_path) and sha256_of(dest_path) == src_sha:
            action = "SKIP (identical)"
            skipped += 1
        elif not args.dry_run:
            os.makedirs(dest_dir, exist_ok=True)
            tmp = dest_path + ".part"
            shutil.copyfile(real, tmp)          # follows symlink == cp -L
            os.replace(tmp, dest_path)          # atomic publish
            copied += 1
            total_copied_bytes += size
        else:
            copied += 1
            total_copied_bytes += size

        print(f"  [{action:16s}] {category}/{dest_name}  ({human(size)})")

        manifest.append(dict(
            label=label,
            category=category,
            run=run_name(src),
            source=src,
            source_real=real,
            dest=dest_path,
            size_bytes=size,
            sha256=src_sha,
            source_mtime_utc=mtime,
            imported_utc=_dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        ))

    print(f"\nCopied {copied}, skipped {skipped}; "
          f"{human(total_copied_bytes)} of new data"
          f"{' (dry run, nothing written)' if args.dry_run else ''}.")

    if not args.dry_run:
        os.makedirs(args.dest, exist_ok=True)
        manifest_path = osp.join(args.dest, "checkpoints_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(dict(
                src_root=args.src_root,
                dest=args.dest,
                generated_utc=_dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                count=len(manifest),
                checkpoints=manifest,
            ), f, indent=2)
        print(f"Provenance manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
