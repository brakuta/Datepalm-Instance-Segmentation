#!/usr/bin/env python3
# ==========================================================================
# audit_predictions.py
# --------------------------------------------------------------------------
# Ground-truth audit visualiser for instance-segmentation predictions.
#
# Runs a trained detector on each annotated image, matches predictions to
# GT by IoU (greedy, score-ordered), classifies every instance as
#   TP  (prediction matched a GT)         -> drawn GREEN
#   FP  (prediction with no GT match)     -> drawn RED   == NEEDS CHECKING
#   FN  (GT with no prediction match)     -> drawn BLUE  == model missed it
# and renders POLYGON OUTLINES ONLY (no text labels), so the underlying
# imagery stays visible and the red FP polygons can be visually judged as
# "real palm the GT missed" vs "genuine false alarm".
#
# Also writes audit_counts.csv (per-image TP/FP/FN) sorted by FP count, so
# the worst tiles for over-prediction surface first.
#
# Usage
# -----
#   python audit_predictions.py \
#       --config   work_dirs/Stage_D/r50_b100_stagec/maskrcnn_r50_staged_ft.py \
#       --checkpoint work_dirs/Stage_D/r50_b100_stagec/best_coco_segm_mAP_50_iter_4000.pth \
#       --ann      /root/datasets/COCO/Sat_30cm/Annotations/test_sat.json \
#       --img-root /root/datasets/COCO/Sat_30cm/test_sat \
#       --out-dir  work_dirs/Stage_D/r50_b100_stagec/audit \
#       --score-thr 0.25 --iou-thr 0.5 --match segm \
#       --show all          # all | fp | fn | fp_fn
#       --max-images 30     # cap for a quick look (omit for the full set)
# ==========================================================================

import argparse
import os
import os.path as osp
import sys

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import cv2  # noqa: E402

# colours in BGR
C_TP = (0, 170, 0)        # green  : matched (correct)
C_FP = (0, 0, 235)        # red    : false positive -> CHECK if real palm
C_FN = (235, 120, 0)      # blue   : false negative -> model missed
LEGEND = [("TP (matched)", C_TP), ("FP - CHECK", C_FP), ("FN (missed)", C_FN)]


def bbox_iou(a, b):
    """IoU of one box a=[x1,y1,x2,y2] against array b [N,4]."""
    if b.size == 0:
        return np.zeros((0,), np.float32)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa + ab - inter, 1e-6, None)


def mask_iou(m, masks):
    """IoU of one bool mask m [H,W] against array masks [N,H,W]."""
    if masks.shape[0] == 0:
        return np.zeros((0,), np.float32)
    m = m.astype(bool)
    inter = np.logical_and(m[None], masks).sum(axis=(1, 2))
    union = np.logical_or(m[None], masks).sum(axis=(1, 2))
    return (inter / np.clip(union, 1e-6, None)).astype(np.float32)


def draw_mask_outline(img, mask, color, thickness=2):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, thickness, cv2.LINE_AA)


def draw_box_outline(img, box, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_legend(img, counts):
    x, y = 12, 24
    for (txt, col), key in zip(LEGEND, ('TP', 'FP', 'FN')):
        label = f"{txt}: {counts[key]}"
        cv2.rectangle(img, (x, y - 14), (x + 18, y), col, -1)
        cv2.putText(img, label, (x + 26, y - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (x + 26, y - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--ann', required=True, help='COCO GT json')
    ap.add_argument('--img-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--score-thr', type=float, default=0.25)
    ap.add_argument('--iou-thr', type=float, default=0.50)
    ap.add_argument('--max-dets', type=int, default=500)
    ap.add_argument('--match', choices=['bbox', 'segm'], default='segm',
                    help='IoU used for matching (segm = mask IoU)')
    ap.add_argument('--show', choices=['all', 'fp', 'fn', 'fp_fn'],
                    default='all', help='which categories to draw')
    ap.add_argument('--thickness', type=int, default=2)
    ap.add_argument('--max-images', type=int, default=None)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    from mmdet.apis import init_detector, inference_detector
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils

    os.makedirs(args.out_dir, exist_ok=True)
    coco = COCO(args.ann)
    img_ids = coco.getImgIds()
    if args.max_images:
        img_ids = img_ids[:args.max_images]

    model = init_detector(args.config, args.checkpoint, device=args.device)

    rows = []
    for n, img_id in enumerate(img_ids, 1):
        info = coco.loadImgs(img_id)[0]
        fpath = osp.join(args.img_root, info['file_name'])
        if not osp.isfile(fpath):
            # file_name may already include a leading subdir; try basename too
            alt = osp.join(args.img_root, osp.basename(info['file_name']))
            fpath = alt if osp.isfile(alt) else fpath
        img = cv2.imread(fpath)
        if img is None:
            print(f'[skip] cannot read {fpath}')
            continue
        H, W = img.shape[:2]

        # ---- GT ----
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        gt_boxes, gt_masks = [], []
        for a in anns:
            x, y, w, h = a['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            if args.match == 'segm':
                gt_masks.append(coco.annToMask(a).astype(bool))
        gt_boxes = np.array(gt_boxes, np.float32) if gt_boxes else np.zeros((0, 4), np.float32)
        gt_masks = np.array(gt_masks) if gt_masks else np.zeros((0, H, W), bool)
        gt_used = np.zeros(len(gt_boxes), bool)

        # ---- predictions ----
        res = inference_detector(model, fpath)
        pi = res.pred_instances
        scores = pi.scores.cpu().numpy()
        boxes = pi.bboxes.cpu().numpy()
        masks = pi.masks.cpu().numpy() if 'masks' in pi else None
        keep = scores >= args.score_thr
        order = np.argsort(-scores[keep])[:args.max_dets]
        idx = np.nonzero(keep)[0][order]            # score-desc, capped
        scores, boxes = scores[idx], boxes[idx]
        masks = masks[idx] if masks is not None else None

        # ---- greedy match (score order) ----
        tp_list, fp_list = [], []
        for i in range(len(boxes)):
            if args.match == 'segm' and masks is not None and len(gt_masks):
                ious = mask_iou(masks[i], gt_masks)
            else:
                ious = bbox_iou(boxes[i], gt_boxes)
            ious = np.where(gt_used, -1.0, ious) if ious.size else ious
            j = int(np.argmax(ious)) if ious.size else -1
            if j >= 0 and ious[j] >= args.iou_thr:
                gt_used[j] = True
                tp_list.append(i)
            else:
                fp_list.append(i)
        fn_idx = np.nonzero(~gt_used)[0]

        counts = dict(TP=len(tp_list), FP=len(fp_list), FN=len(fn_idx))
        rows.append((info['file_name'], counts['TP'], counts['FP'], counts['FN']))

        # ---- draw (outlines only, NO text labels) ----
        vis = img.copy()
        want_tp = args.show == 'all'
        want_fp = args.show in ('all', 'fp', 'fp_fn')
        want_fn = args.show in ('all', 'fn', 'fp_fn')

        if want_tp:
            for i in tp_list:
                if masks is not None:
                    draw_mask_outline(vis, masks[i], C_TP, args.thickness)
                else:
                    draw_box_outline(vis, boxes[i], C_TP, args.thickness)
        if want_fp:
            for i in fp_list:
                if masks is not None:
                    draw_mask_outline(vis, masks[i], C_FP, args.thickness)
                else:
                    draw_box_outline(vis, boxes[i], C_FP, args.thickness)
        if want_fn:
            for j in fn_idx:
                if args.match == 'segm' and len(gt_masks):
                    draw_mask_outline(vis, gt_masks[j], C_FN, args.thickness)
                else:
                    draw_box_outline(vis, gt_boxes[j], C_FN, args.thickness)

        draw_legend(vis, counts)
        out = osp.join(args.out_dir, osp.basename(info['file_name']))
        cv2.imwrite(out, vis)
        if n % 10 == 0 or n == len(img_ids):
            print(f'[{n}/{len(img_ids)}] {osp.basename(out)}  '
                  f'TP={counts["TP"]} FP={counts["FP"]} FN={counts["FN"]}')

    # ---- summary CSV, sorted by FP desc (worst over-prediction first) ----
    import csv
    rows.sort(key=lambda r: -r[2])
    csv_path = osp.join(args.out_dir, 'audit_counts.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file_name', 'TP', 'FP', 'FN'])
        w.writerows(rows)
    tot = np.array([[r[1], r[2], r[3]] for r in rows]).sum(axis=0) if rows else [0, 0, 0]
    print(f'\n[totals] TP={tot[0]}  FP={tot[1]}  FN={tot[2]}')
    print(f'[summary] {csv_path}  (sorted by FP desc)')
    print(f'[images]  {args.out_dir}  (red = FP to check, blue = missed GT)')


if __name__ == '__main__':
    main()
