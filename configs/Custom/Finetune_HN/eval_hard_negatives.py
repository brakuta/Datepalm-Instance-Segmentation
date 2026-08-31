#!/usr/bin/env python3
# =============================================================================
# eval_hard_negatives.py
# -----------------------------------------------------------------------------
# The gate the hard-negative fine-tune was missing: a direct measurement of
# false-positive suppression, on tiles that contain no palms.
#
# WHY THIS EXISTS
#   The fine-tune validates on GE val and selects the best checkpoint on GE val
#   segm mAP@50. GE val contains no desert confusers, so that metric cannot
#   move when the model stops hallucinating palms in the desert. In the first
#   run it did not move: the last seven evaluations were identical to four
#   decimal places, and `save_best` therefore chose a checkpoint by noise.
#
#   On a tile known to contain no palms, EVERY detection is a false positive.
#   No annotation is needed and no matching rule is involved. Detections per
#   tile is the measurement, and it is exact.
#
# CONTAMINATION -- READ THIS BEFORE TRUSTING A NUMBER
#   Tiles the model trained on are not evidence. A fine-tune can drive
#   detections to zero on its own training negatives while changing nothing
#   elsewhere; that is memorisation, not suppression. This script therefore
#   REFUSES to report a headline number over tiles listed in the training
#   COCO unless --allow-contaminated is passed, and always reports the clean
#   and contaminated subsets separately when both are present.
#
#   Generate a clean evaluation set from AOIs or candidates the fine-tune did
#   not use, e.g. a second make_aoi_tiles.py run with a different --seed or a
#   raised --max-per-aoi, then subtract the training file names.
#
# WHAT IT REPORTS, PER CHECKPOINT AND PER SCORE THRESHOLD
#   det_per_tile      mean detections on a palm-free tile. The headline.
#   frac_tiles_dirty  fraction of tiles with at least one detection. Often the
#                     more operationally meaningful number: one spurious crown
#                     in an otherwise clean square kilometre still puts a
#                     polygon on the map.
#   p95_det_per_tile  the tail. A model can have a good mean and still produce
#                     unusable output over a handful of confuser-dense tiles.
#
#   Run it over two or more checkpoints (the base model and each candidate) in
#   one invocation so the comparison uses one decode path and one threshold
#   grid. A suppression claim needs the before number as much as the after.
#
# WHAT IT DOES NOT MEASURE
#   Recall. This script cannot tell you whether the adaptation cost you real
#   palms; only GE val (or a labelled positive set) can. Ship on BOTH: recall
#   held on GE val, false positives down here. Either alone is half a gate.
#
# USAGE
#   python configs/Custom/Finetune_HN/eval_hard_negatives.py \
#       --config configs/Custom/maskrcnn_palm_finetune_hn/maskrcnn_spatialmamba_s_finetune_hn.py \
#       --checkpoint base=work_dirs/Stage_C/maskrcnn_spatialmamba_s_stagec/best_GE_segm_mAP_50_iter_75001.pth \
#       --checkpoint hn4000=work_dirs/Finetune_HN/maskrcnn_spatialmamba_s_finetune_hn/best_coco_segm_mAP_50_iter_4000.pth \
#       --checkpoint hn13000=work_dirs/Finetune_HN/maskrcnn_spatialmamba_s_finetune_hn/iter_13000.pth \
#       --images /workspace/datasets/COCO/HardNeg_GE_eval/images \
#       --trained-on /workspace/datasets/COCO/HardNeg_GE_v2/annotations/hard_neg.json \
#       --scores 0.25 0.35 0.45 0.55 \
#       --out /workspace/work_dirs/hn_eval
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def load_trained_names(paths):
    """File names the fine-tune actually trained on, for contamination flagging.

    Accepts COCO jsons and/or plain image directories, because the negative
    set may have been assembled either way.
    """
    names = set()
    for p in paths or []:
        p = Path(p)
        if not p.exists():
            sys.exit(f"[FATAL] --trained-on path does not exist: {p}")
        if p.is_dir():
            names |= {f.name for f in p.iterdir()
                      if f.suffix.lower() in IMG_EXTS}
        else:
            d = json.loads(p.read_text())
            names |= {Path(im["file_name"]).name for im in d.get("images", [])}
    return names


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="the fine-tune config; supplies the model definition "
                         "and its custom_imports")
    ap.add_argument("--checkpoint", action="append", required=True,
                    metavar="NAME=PATH",
                    help="repeatable. NAME labels the row, e.g. "
                         "base=..., hn13000=... . Give the pre-adaptation "
                         "model too: a suppression claim needs a before.")
    ap.add_argument("--images", required=True,
                    help="directory of palm-free evaluation tiles")
    ap.add_argument("--trained-on", nargs="*", default=[],
                    help="COCO json(s) and/or image dir(s) the fine-tune "
                         "trained on. Tiles matching these are reported "
                         "separately and excluded from the headline.")
    ap.add_argument("--allow-contaminated", action="store_true",
                    help="report a headline number even when every evaluation "
                         "tile was trained on. Off by default because that "
                         "number measures memorisation, not suppression.")
    ap.add_argument("--scores", nargs="+", type=float,
                    default=[0.25, 0.35, 0.45, 0.55])
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N tiles (smoke test)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True,
                    help="output directory; keep it OUTSIDE the repository "
                         "tree so a git operation cannot remove it")
    args = ap.parse_args()

    from mmdet.apis import inference_detector, init_detector

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Recursive: COCO roots nest their pixels (val_GE/JPEGImages/...), and a
    # flat listing silently found nothing and reported it as an empty folder.
    root = Path(args.images)
    if not root.exists():
        sys.exit(f"[FATAL] --images does not exist: {root}")
    tiles = sorted(f for f in root.rglob("*")
                   if f.suffix.lower() in IMG_EXTS)
    if not tiles:
        sys.exit(f"[FATAL] no images under {root} (searched recursively)")
    if args.limit:
        tiles = tiles[:args.limit]

    trained = load_trained_names(args.trained_on)
    seen = np.array([t.name in trained for t in tiles])
    n_clean = int((~seen).sum())
    print(f"{len(tiles)} evaluation tiles: {n_clean} unseen, "
          f"{int(seen.sum())} present in the training negatives")
    if n_clean == 0 and not args.allow_contaminated:
        sys.exit(
            "[FATAL] every evaluation tile was trained on. A false-positive "
            "count over training tiles measures memorisation, not "
            "suppression. Generate fresh tiles (another make_aoi_tiles.py run "
            "with a different --seed or a raised --max-per-aoi, minus these "
            "file names), or pass --allow-contaminated to report it anyway "
            "and label it as such in the write-up.")
    if n_clean < 200:
        print(f"  [warn] only {n_clean} unseen tiles. The mean will be noisy; "
              f"prefer several hundred before quoting a reduction.")

    ckpts = []
    for spec in args.checkpoint:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"--checkpoint needs NAME=PATH, got {spec!r}")
        if not Path(path).exists():
            sys.exit(f"[FATAL] missing checkpoint: {path}")
        ckpts.append((name, path))

    rows, per_tile = [], []
    for name, path in ckpts:
        print(f"\n[{name}] {path}")
        model = init_detector(args.config, path, device=args.device)
        # Scores are collected ONCE per tile and thresholded afterwards. The
        # thresholds only change how the same detections are counted, so a
        # grid costs one inference pass rather than one per threshold.
        scores_per_tile = []
        for i, t in enumerate(tiles, 1):
            res = inference_detector(model, str(t))
            s = res.pred_instances.scores.detach().cpu().numpy()
            scores_per_tile.append(s)
            if i % 200 == 0:
                print(f"    {i}/{len(tiles)}")
        del model

        for thr in args.scores:
            counts = np.array([int((s >= thr).sum()) for s in scores_per_tile])
            for subset, mask in (("unseen", ~seen), ("trained_on", seen),
                                 ("all", np.ones(len(tiles), bool))):
                if not mask.any():
                    continue
                c = counts[mask]
                rows.append(dict(
                    checkpoint=name, path=path, score_thr=thr, subset=subset,
                    n_tiles=int(mask.sum()),
                    n_detections=int(c.sum()),
                    det_per_tile=float(c.mean()),
                    frac_tiles_dirty=float((c > 0).mean()),
                    p95_det_per_tile=float(np.percentile(c, 95)),
                    max_det_per_tile=int(c.max())))
            per_tile.append(pd.DataFrame(dict(
                checkpoint=name, score_thr=thr,
                tile=[t.name for t in tiles],
                trained_on=seen, n_det=counts)))

    res = pd.DataFrame(rows)
    res.to_csv(out / "hard_negative_fp.csv", index=False)
    pd.concat(per_tile, ignore_index=True).to_csv(
        out / "hard_negative_fp_per_tile.csv", index=False)

    head = res[res["subset"] == ("unseen" if n_clean else "all")]
    print("\nfalse positives on palm-free tiles"
          f" ({'unseen' if n_clean else 'CONTAMINATED -- training tiles'}):")
    print(head[["checkpoint", "score_thr", "n_tiles", "det_per_tile",
                "frac_tiles_dirty", "p95_det_per_tile"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Reduction against the first checkpoint given, which is the one the user
    # is expected to pass as the pre-adaptation baseline.
    base_name = ckpts[0][0]
    base = head[head["checkpoint"] == base_name].set_index("score_thr")
    if len(ckpts) > 1 and len(base):
        print(f"\nreduction relative to '{base_name}':")
        for name, _ in ckpts[1:]:
            g = head[head["checkpoint"] == name].set_index("score_thr")
            for thr in sorted(set(base.index) & set(g.index)):
                b = base.loc[thr, "det_per_tile"]
                v = g.loc[thr, "det_per_tile"]
                pct = (100.0 * (v - b) / b) if b else float("nan")
                print(f"  {name:<12s} score {thr:.2f}: "
                      f"{b:.4f} -> {v:.4f} det/tile ({pct:+.1f}%), "
                      f"dirty tiles {base.loc[thr, 'frac_tiles_dirty']:.3f} -> "
                      f"{g.loc[thr, 'frac_tiles_dirty']:.3f}")
        print("\nThis is one half of the gate. Confirm on GE val that recall "
              "did not fall before shipping any of these.")
    print(f"\ntables -> {out / 'hard_negative_fp.csv'}, "
          f"{out / 'hard_negative_fp_per_tile.csv'}")


if __name__ == "__main__":
    main()
