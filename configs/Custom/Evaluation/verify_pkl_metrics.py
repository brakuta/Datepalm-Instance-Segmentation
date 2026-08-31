"""
verify_pkl_metrics.py
=========================================================================
Recompute accuracy metrics DIRECTLY FROM THE SAVED PKLs and (optionally)
diff them against the previously compiled Stage C matrix.

WHY THIS EXISTS
    tools/test.py --out writes DumpDetResults output: per-image
    pred_instances only. NO metrics are stored inside the pkl. The mAP
    numbers printed during the run went to stdout (and to your tee'd log).
    The pkl is, however, metric-SUFFICIENT: it contains every prediction
    CocoMetric scored, so the numbers can be reproduced exactly offline.

    This is the stronger cross-check of the two available: it verifies the
    pkl CONTENT (what the figures will actually render), not merely what
    some earlier process printed. A pkl written from the wrong checkpoint
    or the wrong split reproduces the wrong mAP here and is caught.

PROTOCOL MATCHED TO MMDET
    MMDet's CocoMetric was configured with proposal_nums=(100, 300, 1000),
    so its reported AP@0.50 uses maxDets=1000 -- NOT the COCO default of
    100. This script sets maxDets=[100, 300, 1000] and reads AP@0.50 at
    the 1000 slot, reproducing the logged 'coco/segm_mAP_50' value.

USAGE
    python verify_pkl_metrics.py                       # all 30 pkls -> CSV
    python verify_pkl_metrics.py --only SpatialMamba-S
    python verify_pkl_metrics.py --compare /path/Cross_transfer_matrix_stagec_segm_mAP_50.csv

OUTPUT
    results/qual/pkl_metrics_check.csv  with columns:
        model, test_set, n_images, n_pred, bbox_mAP_50, segm_mAP_50
    and, with --compare, a printed diff against the compiled matrix.
"""

import argparse
import contextlib
import glob
import io
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# ==========================================================================
# CONFIG
# ==========================================================================
PKL_DIR = "results/qual"
ANN = {   # test-set annotation JSONs (must be the ones the models were scored on)
    "UAV":    "/workspace/datasets/COCO/UAV_5cm/Annotations/test_UAV.json",
    "GE":     "/workspace/datasets/COCO/GE_15cm/Annotations/test_GE.json",
    "Aerial": "/workspace/datasets/COCO/Aerial_15cm/Annotations/test_aerial.json",
}
OUT_CSV = "results/qual/pkl_metrics_check.csv"
# Roots scanned to IDENTIFY which annotation file a pkl actually matches when
# its img_ids do not fit the expected test set. Any *.json under these dirs.
ANN_SEARCH_GLOBS = [
    "/root/datasets/COCO/*/Annotations/*.json",
    "/workspace/datasets/COCO/*/Annotations/*.json",
]
# Column names in the compiled matrix, keyed by test set (for --compare).
COMPILED_COLS = {"UAV": "UAV", "GE": "GE", "Aerial": "Aerial"}
# ==========================================================================


def to_rle(m, h, w):
    """Accept RLE dict (DumpDetResults default), list-of-poly, or bitmap."""
    if isinstance(m, dict):
        rle = dict(m)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode()
        return rle
    if isinstance(m, list):
        return maskUtils.merge(maskUtils.frPyObjects(m, h, w))
    arr = np.asarray(m)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return maskUtils.encode(np.asfortranarray(arr))


def pkl_to_coco_results(pkl_path, cat_id):
    """Convert DumpDetResults pkl -> COCO detection result dicts."""
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    dets, n_img = [], len(results)
    for r in results:
        img_id = r["img_id"]
        h, w = r.get("ori_shape", r.get("img_shape"))[:2]
        pi = r["pred_instances"]
        boxes = np.asarray(pi["bboxes"])
        scores = np.asarray(pi["scores"])
        masks = pi.get("masks", None)
        for i in range(len(scores)):
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            d = {"image_id": int(img_id), "category_id": cat_id,
                 "bbox": [x1, y1, x2 - x1, y2 - y1],
                 "score": float(scores[i])}
            if masks is not None:
                rle = to_rle(masks[i], h, w)
                d["segmentation"] = {"size": list(rle["size"]),
                                     "counts": rle["counts"].decode()
                                     if isinstance(rle["counts"], bytes)
                                     else rle["counts"]}
            dets.append(d)
    return dets, n_img


def _img_ids_of(json_path):
    """Cheap img_id set read (no full COCO index build)."""
    import json
    try:
        with open(json_path) as f:
            d = json.load(f)
        return {int(i["id"]) for i in d.get("images", [])}
    except Exception:
        return set()


def identify(pkl_ids):
    """Report which annotation files this pkl's img_ids fit, best first."""
    cands = []
    for g in ANN_SEARCH_GLOBS:
        for p in glob.glob(g):
            ids = _img_ids_of(p)
            if not ids:
                continue
            inter = len(pkl_ids & ids)
            cands.append((inter / max(len(pkl_ids), 1), inter, len(ids), p))
    cands.sort(reverse=True)
    return cands[:4]


def ap50(coco_gt, dets, iou_type):
    """AP@0.50 with maxDets=1000, matching MMDet's proposal_nums setting."""
    if not dets:
        return float("nan")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):          # silence COCOeval spam
        coco_dt = coco_gt.loadRes([dict(d) for d in dets])
        E = COCOeval(coco_gt, coco_dt, iou_type)
        E.params.maxDets = [100, 300, 1000]
        E.evaluate(); E.accumulate()
    # precision dims: [T(iou), R(recall), K(cat), A(area), M(maxDets)]
    t = np.where(np.isclose(E.params.iouThrs, 0.5))[0][0]
    p = E.eval["precision"][t, :, :, 0, 2]         # area=all, maxDets=1000
    p = p[p > -1]
    return float(np.mean(p)) if p.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="single model key")
    ap.add_argument("--compare", default=None,
                    help="compiled matrix CSV to diff against")
    ap.add_argument("--tol", type=float, default=0.003,
                    help="flag |diff| above this (default 0.003)")
    args = ap.parse_args()

    coco_cache, rows = {}, []
    paths = sorted(glob.glob(f"{PKL_DIR}/*_stageC_*.pkl"))
    if not paths:
        sys.exit(f"[FATAL] no pkls under {PKL_DIR}/")

    for p in paths:
        m = re.match(r"(.+)_stageC_(UAV|GE|Aerial)\.pkl$", Path(p).name)
        if not m:
            continue
        model, tset = m.group(1), m.group(2)
        if args.only and model != args.only:
            continue
        if tset not in coco_cache:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                coco_cache[tset] = COCO(ANN[tset])
        gt = coco_cache[tset]
        cat_id = gt.getCatIds()[0]
        dets, n_img = pkl_to_coco_results(p, cat_id)
        pkl_ids = {int(d["image_id"]) for d in dets}
        gt_ids = set(int(i) for i in gt.getImgIds())
        n_gt_img = len(gt_ids)
        stray = pkl_ids - gt_ids
        coverage = len(pkl_ids & gt_ids) / max(n_gt_img, 1)

        status = "ok"
        if stray:
            status = f"WRONG_GT ({len(stray)} unknown img_ids)"
        elif n_img < n_gt_img:
            status = f"INCOMPLETE ({n_img}/{n_gt_img} images)"

        if stray:
            # Do not call loadRes: it asserts. Diagnose instead.
            print(f"{model:>18} | {tset:>6} | {n_img:4d} imgs | "
                  f"{len(dets):6d} preds | *** {status} ***")
            print(f"{'':>18}   expected GT: {ANN[tset]} ({n_gt_img} images)")
            print(f"{'':>18}   pkl img_ids fit these annotation files best:")
            for frac, inter, n_ids, path in identify(pkl_ids):
                print(f"{'':>18}     {frac*100:5.1f}% of pkl ids | "
                      f"{inter:5d}/{n_ids:5d} | {path}")
            rows.append({"model": model, "test_set": tset, "n_images": n_img,
                         "n_gt_images": n_gt_img, "n_pred": len(dets),
                         "bbox_mAP_50": float("nan"),
                         "segm_mAP_50": float("nan"), "status": status})
            continue

        row = {"model": model, "test_set": tset, "n_images": n_img,
               "n_gt_images": n_gt_img, "n_pred": len(dets),
               "bbox_mAP_50": ap50(gt, dets, "bbox"),
               "segm_mAP_50": ap50(gt, dets, "segm"),
               "status": status}
        rows.append(row)
        flag = "" if status == "ok" else f"   *** {status} ***"
        print(f"{model:>18} | {tset:>6} | {n_img:4d}/{n_gt_img:4d} imgs | "
              f"{len(dets):6d} preds | bbox50={row['bbox_mAP_50']:.3f} | "
              f"segm50={row['segm_mAP_50']:.3f}{flag}")

    df = pd.DataFrame(rows)
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] {OUT_CSV}")

    bad = df[df["status"] != "ok"]
    if len(bad):
        print(f"\n*** {len(bad)} of {len(df)} cells are NOT usable "
              f"-- regenerate these pkls before rendering figures ***")
        print(bad[["model", "test_set", "n_images", "n_gt_images",
                   "status"]].to_string(index=False))
    else:
        print("\nAll cells: full image coverage, img_ids consistent with GT.")

    if args.compare:
        ref = pd.read_csv(args.compare)
        key = ref.columns[0]
        print(f"\n=== diff vs {args.compare} (segm mAP@0.5, tol {args.tol}) ===")
        for _, r in df.iterrows():
            if r["status"] != "ok":
                print(f"  {r['model']:>18} {r['test_set']:>6}: skipped ({r['status']})")
                continue
            col = COMPILED_COLS[r["test_set"]]
            hit = ref[ref[key].astype(str).str.contains(r["model"],
                                                        case=False, regex=False)]
            if hit.empty or col not in ref.columns:
                print(f"  {r['model']:>18} {r['test_set']:>6}: no matching "
                      f"row/column in compiled matrix")
                continue
            old = float(hit.iloc[0][col])
            d = r["segm_mAP_50"] - old
            flag = "  <-- CHECK" if abs(d) > args.tol else ""
            print(f"  {r['model']:>18} {r['test_set']:>6}: pkl={r['segm_mAP_50']:.3f} "
                  f"compiled={old:.3f}  diff={d:+.3f}{flag}")


if __name__ == "__main__":
    main()
