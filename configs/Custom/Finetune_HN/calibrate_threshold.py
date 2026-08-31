#!/usr/bin/env python3
# =============================================================================
# calibrate_threshold.py
# -----------------------------------------------------------------------------
# Re-derive the operating threshold after adaptation, on BOTH axes at once.
#
# WHY THIS EXISTS
#   The country pipeline ships a fixed SCORE_THR (0.35) chosen for the
#   pre-adaptation model. Fine-tuning moves the score distribution, so that
#   number is no longer the right one, and it was never derived from an F1
#   optimum in the first place -- it was inherited.
#
#   More importantly, hard-negative adaptation changes the TRADE the threshold
#   is making. Before adaptation, lowering the threshold recovered faint palms
#   at the cost of flooding the desert with spurious crowns, so the threshold
#   had to stay high and real palms were lost. If the false positives are gone,
#   that constraint is gone with them and the threshold should move down. This
#   script measures both sides on the same grid so the choice is made from
#   evidence rather than inherited.
#
# WHAT IT COMPUTES
#   On a LABELLED set (GE val), per threshold: precision, recall and F1 under
#   COCO-style score-ordered greedy matching at --iou-thr, identical to the
#   rule used by the crown-size analysis and by extract_instance_errors.py. A
#   renderer or metric that re-matches globally by IoU will disagree.
#
#   On an optional PALM-FREE set (the held-out desert tiles), per threshold:
#   false positives per tile. Every detection there is an error by construction.
#
#   The two are reported side by side. F1 alone will happily pick a threshold
#   that is unusable at country scale, because a labelled validation set of
#   farmland does not contain the desert where the errors actually occur, and
#   the desert is most of the UAE.
#
# COST
#   Inference runs ONCE per image; the grid only changes how the stored scores
#   and IoUs are counted. A 20-point grid costs the same as a single threshold.
#
# USAGE
#   python configs/Custom/Finetune_HN/calibrate_threshold.py \
#       --config configs/Custom/maskrcnn_palm_finetune_hn/maskrcnn_spatialmamba_s_finetune_hn.py \
#       --checkpoint work_dirs/Finetune_HN/maskrcnn_spatialmamba_s_finetune_hn/best_coco_segm_mAP_50_iter_4000.pth \
#       --ann /workspace/datasets/COCO/GE_15cm/Annotations/val_GE.json \
#       --images /workspace/datasets/COCO/GE_15cm/val_GE \
#       --neg-images /workspace/datasets/COCO/HardNeg_GE_eval/images \
#       --out /workspace/work_dirs/hn_threshold
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ann", required=True,
                    help="COCO json for the labelled positive set")
    ap.add_argument("--images", required=True,
                    help="image root for --ann (searched recursively)")
    ap.add_argument("--neg-images", default=None,
                    help="palm-free tiles; every detection there is a false "
                         "positive. Optional but strongly advised: F1 on "
                         "farmland cannot see the desert error mode.")
    ap.add_argument("--iou-thr", type=float, default=0.50)
    ap.add_argument("--grid", nargs="+", type=float, default=None,
                    help="score thresholds (default 0.05 to 0.95 by 0.05)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--neg-limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True,
                    help="output directory; keep it OUTSIDE the repo tree")
    args = ap.parse_args()

    from mmdet.apis import inference_detector, init_detector
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils

    grid = np.asarray(args.grid if args.grid else
                      np.round(np.arange(0.05, 0.96, 0.05), 2), float)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def to_rle(seg, h, w):
        if isinstance(seg, dict):
            rle = dict(seg)
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode()
            return rle
        if isinstance(seg, list):
            return maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
        return maskUtils.encode(np.asfortranarray(
            np.asarray(seg).astype(np.uint8)))

    def greedy(ious, order, thr):
        """COCO-style score-ordered greedy assignment. Returns the matched
        mask over ground truth and the number of detections that matched."""
        n_g = ious.shape[1] if ious.size else 0
        matched = np.zeros(n_g, bool)
        n_tp = 0
        for d in order:
            row = ious[d].copy()
            row[matched] = -1.0
            if row.size == 0:
                continue
            g = int(np.argmax(row))
            if row[g] >= thr:
                matched[g] = True
                n_tp += 1
        return matched, n_tp

    print(f"loading {args.checkpoint}")
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # ---- labelled positives ------------------------------------------------
    coco = COCO(args.ann)
    root = Path(args.images)
    by_name = {p.name: p for p in root.rglob("*")
               if p.suffix.lower() in IMG_EXTS}
    img_ids = sorted(coco.getImgIds())
    if args.limit:
        img_ids = img_ids[:args.limit]

    # Per image: the full IoU matrix and detection scores, computed once with
    # NO score filter. The grid then only decides which rows to read.
    per_img = []
    n_missing = 0
    for n, img_id in enumerate(img_ids, 1):
        info = coco.loadImgs(img_id)[0]
        p = by_name.get(Path(info["file_name"]).name)
        if p is None:
            n_missing += 1
            continue
        h, w = info["height"], info["width"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))
        g_rles = [to_rle(a["segmentation"], h, w) for a in anns]
        res = inference_detector(model, str(p))
        pi = res.pred_instances
        scores = pi.scores.detach().cpu().numpy()
        if len(scores) and g_rles:
            masks = pi.masks.detach().cpu().numpy()
            d_rles = [maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
                      for m in masks]
            iou = np.asarray(maskUtils.iou(d_rles, g_rles, [0] * len(g_rles))
                             ).reshape(len(d_rles), len(g_rles))
        else:
            iou = np.zeros((len(scores), len(g_rles)))
        per_img.append((scores, iou, len(g_rles)))
        if n % 100 == 0:
            print(f"    {n}/{len(img_ids)}")
    if n_missing:
        print(f"  [warn] {n_missing} annotated image(s) had no file under "
              f"{root}; they are excluded, so recall is over the rest.")
    if not per_img:
        sys.exit("[FATAL] no images matched between --ann and --images")

    rows = []
    for thr in grid:
        tp = fp = fn = 0
        for scores, iou, n_gt in per_img:
            keep = np.where(scores >= thr)[0]
            sub = iou[keep] if len(keep) else np.zeros((0, n_gt))
            order = np.argsort(-scores[keep]) if len(keep) else []
            matched, n_tp = greedy(sub, order, args.iou_thr)
            tp += n_tp
            fp += len(keep) - n_tp
            fn += n_gt - int(matched.sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if prec and rec and np.isfinite(prec + rec) else float("nan"))
        rows.append(dict(score_thr=float(thr), tp=tp, fp=fp, fn=fn,
                         precision=prec, recall=rec, f1=f1))
    df = pd.DataFrame(rows)

    # ---- palm-free tiles ---------------------------------------------------
    if args.neg_images:
        nroot = Path(args.neg_images)
        neg = sorted(f for f in nroot.rglob("*")
                     if f.suffix.lower() in IMG_EXTS)
        if args.neg_limit:
            neg = neg[:args.neg_limit]
        print(f"palm-free tiles: {len(neg)}")
        neg_scores = []
        for n, f in enumerate(neg, 1):
            r = inference_detector(model, str(f))
            neg_scores.append(r.pred_instances.scores.detach().cpu().numpy())
            if n % 200 == 0:
                print(f"    {n}/{len(neg)}")
        df["fp_per_negative_tile"] = [
            float(np.mean([int((s >= t).sum()) for s in neg_scores]))
            for t in grid]
        df["frac_negative_tiles_dirty"] = [
            float(np.mean([bool((s >= t).any()) for s in neg_scores]))
            for t in grid]

    df.to_csv(out / "threshold_calibration.csv", index=False)
    cols = [c for c in ("score_thr", "precision", "recall", "f1",
                        "fp_per_negative_tile", "frac_negative_tiles_dirty")
            if c in df]
    print("\n" + df[cols].to_string(index=False,
                                    float_format=lambda x: f"{x:.4f}"))

    best = df.loc[df["f1"].idxmax()]
    print(f"\nF1 optimum: score {best['score_thr']:.2f} "
          f"(P {best['precision']:.4f}, R {best['recall']:.4f}, "
          f"F1 {best['f1']:.4f})")
    if "fp_per_negative_tile" in df:
        print(f"  at that threshold, {best['fp_per_negative_tile']:.4f} false "
              f"positive(s) per palm-free tile")
        # The operating point is a decision, not a maximum. Recall is what a
        # national inventory is short of, so show what a small F1 concession
        # buys and what it costs in the desert.
        near = df[df["f1"] >= best["f1"] - 0.01]
        if len(near) > 1:
            r = near.loc[near["recall"].idxmax()]
            print(f"  within 0.01 F1, the most permissive point is score "
                  f"{r['score_thr']:.2f}: recall {r['recall']:.4f} "
                  f"({r['recall'] - best['recall']:+.4f}), "
                  f"{r['fp_per_negative_tile']:.4f} FP per palm-free tile")
    print("\nChoose from BOTH columns. F1 is computed on labelled farmland "
          "and cannot see the desert error mode, which is most of the "
          "country by area.")
    print(f"\ntable -> {out / 'threshold_calibration.csv'}")


if __name__ == "__main__":
    main()
