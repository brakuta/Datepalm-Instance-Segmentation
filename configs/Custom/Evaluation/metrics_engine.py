# ==========================================================================
# metrics_engine.py
# ==========================================================================
# Date Palm Benchmark — accurate and fast evaluation of MMDetection
# instance-segmentation models (Mask R-CNN family with CNN, Transformer,
# and Mamba-based backbones) across UAV / aerial / satellite test sets.
#
# Reported metrics
# ----------------
#   bbox / segm  mAP @ IoU 0.50:0.05:0.95             (supplementary)
#   bbox / segm  mAP @ IoU 0.50                       (PRIMARY)
#   bbox / segm  Precision, Recall, F1 @ IoU = 0.50,
#                  at the F1-optimal score threshold  (PRIMARY)
#   bbox / segm  Precision, Recall, F1 @ IoU = 0.50,
#                  at fixed score_thr = 0.05          (protocol traceability)
#
# Why two F-score numbers?
# ------------------------
#   COCO mAP integrates precision over the whole recall range, so a
#   permissive score threshold (0.05) is appropriate for mAP. A single-
#   point F1 at score = 0.05 is, however, dominated by tens of thousands
#   of low-confidence detections that the model itself does not believe
#   in, artificially depressing precision while leaving mAP@50 unchanged.
#   The F1-optimal threshold is the operating point that an end user
#   would actually deploy and is the value reported as PRIMARY in the
#   manuscript. The fixed-threshold value is retained for traceability
#   to prior date-palm benchmarks that adopted the COCO 0.05 convention.
#
# Protocol invariants (LOCKED)
# ----------------------------
#   iou_thr   = 0.50     TP-matching threshold for F-score
#   maxDets   = 100      COCO default for AP@.50:.95 / AP@50
#   score_thr = 0.05     reported only as the fixed-threshold reference
#
# Design summary
# --------------
#   1. Inference uses MMDetection's `Runner.test()` with the config's own
#      `test_dataloader`. This activates true GPU batching, persistent
#      DataLoader workers, and the framework's optimised test pipeline.
#
#   2. Mask predictions are encoded to RLE inside the custom metric the
#      moment they leave the model, and all subsequent IoU computations
#      use `pycocotools.mask.iou`.
#
#   3. F-score evaluation runs ONE COCOeval pass at IoU = iou_thr on the
#      unfiltered predictions. Per-detection match arrays are gathered
#      from `evalImgs[*]['dtMatches'/'dtScores']` and used twice:
#         (a) at score >= score_thr  -> fixed-threshold P/R/F1,
#         (b) swept over score grid  -> F1-optimal P/R/F1 + threshold.
#      This avoids any redundant evaluation work.
#
#   4. mAP is computed in a separate `COCOeval` pass on the unfiltered
#      predictions, as required by the COCO protocol.
#
# Compatibility
# -------------
#   MMDetection >= 3.0   |   mmengine >= 0.7   |   pycocotools >= 2.0.6
#   Tested on PyTorch 2.x with NVIDIA Titan RTX (24 GB) inside Docker.
#
# Usage
# -----
#   python metrics_engine.py \
#       --config     configs/Custom/1_single_sensor_uav_5cm/maskrcnn_r50_uav5cm.py \
#       --checkpoint work_dirs/maskrcnn_r50_uav5cm/best_coco_segm_mAP_50_iter_*.pth
#       --batch-size 8     --num-workers 8
#   Output
#       results/<config_stem>__<ann_stem>.csv
# ==========================================================================

import argparse
import copy
import os
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


# ===========================================================================
# 1. Custom metric
# ===========================================================================
#
# A single Metric subclass collects predictions in COCO format and produces
# both mAP (full IoU sweep) and Precision / Recall / F1 (IoU = 0.50,
# score_thr = 0.05) for bbox and segm. Predictions are stored as
# lightweight dicts; masks are RLE-encoded at collection time.
#
# The metric is registered under a dedicated name so it can be injected
# into any config via `--cfg-options test_evaluator.type=PalmBenchmark`.
# ===========================================================================

@METRICS.register_module()
class PalmBenchmarkMetric(BaseMetric):
    """COCO mAP plus protocol-locked Precision / Recall / F1 at IoU = 0.50.

    Args:
        ann_file (str): Path to the COCO-format ground-truth JSON.
        score_thr (float): Confidence threshold for F-score evaluation.
            Locked to 0.05 by the benchmark protocol.
        iou_thr (float): IoU threshold for F-score TP matching.
            Locked to 0.50 by the benchmark protocol.
        applied_score_thr (float | None): Externally supplied operating
            threshold for the reported "optimal" P/R/F1 block. When None
            (default), the operating point is found by sweeping the score
            grid ON THE TEST SET (an oracle upper bound). When a value is
            given — typically the F1-maximising threshold selected on a
            held-out validation set — the reported operating point is that
            fixed value instead, removing the test-set tuning bias. The
            fixed-threshold (score=0.05) block is unaffected either way.
        max_dets (int): Maximum detections per image scored by COCOeval,
            applied identically to the mAP and the P/R/F1 passes. The COCO
            default of 100 is designed for natural images with few objects
            and silently caps recall on dense scenes: on a tile holding
            more than `max_dets` instances, every ground-truth beyond the
            top-`max_dets` detections becomes a forced false negative
            regardless of prediction quality. For dense plantation tiles
            this must be raised to at least the densest tile's instance
            count (and is bounded above by the model's test_cfg
            `max_per_img`). Defaults to 100 for COCO comparability; set it
            higher to remove the cap-induced recall ceiling.
        metric_items (Sequence[str]): Selects which metric heads to run.
            Defaults to ('bbox', 'segm').
        collect_device (str): mmengine collection device.
        prefix (str | None): Metric-name prefix in logged results.
    """

    default_prefix = 'palm'

    def __init__(self,
                 ann_file: str,
                 score_thr: float = 0.05,
                 iou_thr: float = 0.50,
                 applied_score_thr: Optional[float] = None,
                 max_dets: int = 100,
                 metric_items: Sequence[str] = ('bbox', 'segm'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ann_file = ann_file
        self.score_thr = float(score_thr)
        self.iou_thr = float(iou_thr)
        self.applied_score_thr = (None if applied_score_thr is None
                                  else float(applied_score_thr))
        self.max_dets = int(max_dets)
        self.metric_items = tuple(metric_items)

        # Load GT once on the main process. COCO is read-only after this.
        self._coco_gt = COCO(ann_file)
        cats = self._coco_gt.loadCats(self._coco_gt.getCatIds())
        # Map model contiguous label index -> COCO category id.
        # Assumes the dataset class order matches sorted COCO category ids,
        # which is the MMDetection default for CocoDataset.
        sorted_cats = sorted(cats, key=lambda c: c['id'])
        self._cat_id_map: Dict[int, int] = {
            idx: c['id'] for idx, c in enumerate(sorted_cats)
        }
        self._cat_names = [c['name'] for c in sorted_cats]

    # -------------------------------------------------------------------
    # Per-batch prediction collection
    # -------------------------------------------------------------------
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Encodes predictions for every sample in the batch.

        Called by the Runner after each forward pass. `data_samples` is
        the mmdet `DetDataSample` list with `.pred_instances` attached.
        """
        for sample in data_samples:
            img_id = sample.get('img_id',
                                sample.get('image_id', None))
            if img_id is None:
                # Fallback: filename-derived id (unsupported here, signals
                # a malformed annotation pipeline).
                raise KeyError(
                    'DetDataSample does not carry img_id; '
                    'verify the test dataset is COCO-format.'
                )

            pred = sample['pred_instances']
            if len(pred) == 0:
                continue

            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy().astype(int)
            bboxes = pred['bboxes'].cpu().numpy()  # (N, 4) xyxy

            # ---- bbox predictions in COCO format -------------------------
            bbox_records = []
            for s, l, b in zip(scores, labels, bboxes):
                x1, y1, x2, y2 = b.tolist()
                bbox_records.append(dict(
                    image_id=int(img_id),
                    category_id=self._cat_id_map[int(l)],
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    score=float(s),
                ))

            # ---- segm predictions in COCO RLE format ---------------------
            segm_records: List[dict] = []
            if 'masks' in pred and pred['masks'] is not None:
                masks = pred['masks']
                # mmdet returns torch BoolTensor on CUDA — pull to CPU once.
                if isinstance(masks, torch.Tensor):
                    masks_np = masks.detach().cpu().numpy()
                else:
                    masks_np = np.asarray(masks)
                # Encode each mask to RLE in Fortran order (pycocotools API).
                for s, l, m in zip(scores, labels, masks_np):
                    rle = mask_utils.encode(
                        np.asfortranarray(m.astype(np.uint8))
                    )
                    rle['counts'] = rle['counts'].decode('ascii')
                    segm_records.append(dict(
                        image_id=int(img_id),
                        category_id=self._cat_id_map[int(l)],
                        segmentation=rle,
                        score=float(s),
                    ))

            self.results.append(dict(
                bbox=bbox_records,
                segm=segm_records,
            ))

    # -------------------------------------------------------------------
    # COCOeval helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _run_coco_eval(coco_gt: COCO,
                       predictions: List[dict],
                       iou_type: str,
                       iou_thrs: Optional[np.ndarray] = None,
                       single_area: bool = False,
                       max_dets: Optional[int] = None
                       ) -> Optional[COCOeval]:
        """Runs evaluate()+accumulate() and returns the COCOeval object.

        Returns None when `predictions` is empty (no detections produced).

        single_area:
            When True, the COCO area-range partition (all / small / medium
            / large) is collapsed to a single 'all' range BEFORE
            evaluate(). This is required for the F-score path: pycocotools
            lays out ``evalImgs`` as catId x areaRng x imgId, so with the
            default four ranges every detection's per-image match arrays
            appear four times. `_gather_matches` concatenates across all
            ``evalImgs`` entries, which would therefore count each
            detection and each ground-truth roughly twice (the in-range
            subsets of small+medium+large re-sum to the 'all' set already
            counted). Restricting to one area range makes the gathered
            TP/FP/FN counts exact. The mAP path must NOT set this, as the
            COCO summary needs the full area partition.

        max_dets:
            When given, overrides COCOeval's per-image detection cap.
            pycocotools both (a) reports AP/AR at ``max(params.maxDets)``
            and (b) builds ``evalImgs`` keeping only the top
            ``max(params.maxDets)`` detections per image. The default of
            100 caps recall on dense tiles. Passing a higher value raises
            the cap for BOTH the mAP and the F-score passes so they remain
            mutually consistent. A three-element list is used so the
            standard ``summarize()`` indexing (which references
            maxDets[0..2]) does not break; the headline AP is taken at the
            last (largest) entry.
        """
        if len(predictions) == 0:
            return None

        coco_dt = coco_gt.loadRes(copy.deepcopy(predictions))
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        if iou_thrs is not None:
            coco_eval.params.iouThrs = np.asarray(iou_thrs)
        if single_area:
            # Collapse to a single 'all' area range so the per-detection
            # match arrays are not replicated across four ranges.
            coco_eval.params.areaRng = [[0, 1e5 ** 2]]
            coco_eval.params.areaRngLbl = ['all']
        if max_dets is not None:
            # Keep exactly three entries for summarize() compatibility;
            # the largest governs both AP reporting and evalImgs retention.
            coco_eval.params.maxDets = [1, 10, int(max_dets)]
        coco_eval.evaluate()
        coco_eval.accumulate()
        return coco_eval

    @staticmethod
    def _ap_from(coco_eval: Optional[COCOeval]) -> Dict[str, float]:
        """Extracts mAP and mAP@50 from a fully-accumulated COCOeval.

        Reads the AP directly from the accumulated ``eval['precision']``
        tensor of shape [T, R, K, A, M] (IoU x recall x category x
        area-range x maxDets) rather than from ``summarize()``. The stock
        ``summarize()`` prints (and stores in ``stats``) the headline
        AP@[.5:.95] line at a HARD-CODED ``maxDets=100``; when the cap is
        raised (``params.maxDets = [1, 10, max_dets]`` with max_dets != 100)
        that index does not exist and pycocotools returns its sentinel
        ``-1``. Selecting the 'all' area range and the largest maxDets
        index from the tensor yields the correct AP for whatever cap is in
        force, so the supplementary mAP@[.5:.95] is valid at the same
        maxDets as the primary mAP@50 and the two are mutually consistent.
        A compact, correct console line is printed in place of the stock
        summary so no ``-1`` artifact appears.
        """
        if coco_eval is None:
            return dict(mAP=0.0, mAP_50=0.0)

        p = coco_eval.params
        precision = coco_eval.eval['precision']   # [T, R, K, A, M]
        recall = coco_eval.eval['recall']         # [T, K, A, M]

        def _area_idx(label: str) -> Optional[int]:
            return next((i for i, l in enumerate(p.areaRngLbl)
                         if l == label), None)

        def _mean_pos(x: np.ndarray) -> float:
            x = x[x > -1]
            return float(x.mean()) if x.size else 0.0

        a_all = _area_idx('all') or 0
        m_last = len(p.maxDets) - 1               # largest cap in force

        mAP = _mean_pos(precision[:, :, :, a_all, m_last])

        t50 = np.where(np.isclose(p.iouThrs, 0.50))[0]
        mAP_50 = (_mean_pos(precision[t50[0], :, :, a_all, m_last])
                  if t50.size else 0.0)

        # Compact, correct console block (replaces the stock summary, whose
        # AP@[.5:.95] line is hard-coded at maxDets=100 and prints -1 here).
        cap = p.maxDets[m_last]
        print(f'[metric]   AP@[.50:.95] = {mAP:.4f}   '
              f'AP@50 = {mAP_50:.4f}   (area=all, maxDets={cap})')
        for label in ('small', 'medium', 'large'):
            ai = _area_idx(label)
            if ai is not None:
                ap_l = _mean_pos(precision[:, :, :, ai, m_last])
                ar_l = _mean_pos(recall[:, :, ai, m_last])
                print(f'[metric]   {label:<6} AP = {ap_l:.4f}   '
                      f'AR = {ar_l:.4f}')

        return dict(mAP=mAP, mAP_50=mAP_50)

    @staticmethod
    def _gather_matches(coco_eval: COCOeval
                        ) -> Dict[str, np.ndarray]:
        """Aggregates per-image match arrays into flat detection arrays.

        After `evaluate()` the COCOeval object stores, for every
        (image, category) cell of `evalImgs`, three per-detection arrays
        already sorted internally by descending score:
            dtScores  (D,)       — detection confidence
            dtMatches (T, D)     — matched-GT id per IoU threshold
            dtIgnore  (T, D)     — ignore flag per IoU threshold

        With `iouThrs = [0.50]`, T = 1 and the relevant slices are the
        first row. We concatenate across all images / categories and
        return four flat 1-D arrays plus the total non-ignore GT count.
        These are the inputs needed to sweep a score threshold without
        re-running COCOeval.
        """
        scores_list = []
        matches_list = []
        dt_ignore_list = []
        n_gt = 0

        for ev in coco_eval.evalImgs:
            if ev is None:
                continue
            scores_list.append(np.asarray(ev['dtScores']))
            matches_list.append(np.asarray(ev['dtMatches'][0]))
            dt_ignore_list.append(np.asarray(ev['dtIgnore'][0],
                                             dtype=bool))
            gt_ignore = np.asarray(ev['gtIgnore'], dtype=bool)
            n_gt += int(np.sum(~gt_ignore))

        if not scores_list:
            return dict(
                scores=np.zeros(0),
                matches=np.zeros(0),
                dt_ignore=np.zeros(0, dtype=bool),
                n_gt=0,
            )

        return dict(
            scores=np.concatenate(scores_list),
            matches=np.concatenate(matches_list),
            dt_ignore=np.concatenate(dt_ignore_list),
            n_gt=int(n_gt),
        )

    @staticmethod
    def _prf_at_threshold(scores: np.ndarray,
                          matches: np.ndarray,
                          dt_ignore: np.ndarray,
                          n_gt: int,
                          score_thr: float) -> Dict[str, float]:
        """Computes P / R / F1 / TP / FP / FN at a single score threshold.

        A detection is counted iff (score >= score_thr) and not ignored.
        It is a TP iff additionally `matches > 0` (i.e. matched a GT at
        the operating IoU threshold). FN is the number of non-ignore GTs
        that no admitted detection matched — equivalently, n_gt minus
        the number of *unique* GTs covered by admitted TP detections.
        Because COCOeval enforces one-to-one matching greedily by score
        and writes the matched GT id into `matches`, "admitted TPs" are
        already in one-to-one correspondence with covered GTs, and the
        unique-count reduces to the TP count.

        This function therefore evaluates exactly the same TP/FP/FN
        semantics as a full COCOeval pass with the predictions filtered
        to score >= score_thr, but without re-running evaluate().
        """
        if scores.size == 0 or n_gt == 0:
            return dict(precision=0.0, recall=0.0, f1=0.0,
                        tp=0, fp=0, fn=int(n_gt),
                        score_thr=float(score_thr))

        keep = (scores >= score_thr) & (~dt_ignore)
        admitted_matches = matches[keep]
        tp = int(np.sum(admitted_matches > 0))
        fp = int(np.sum(admitted_matches == 0))
        fn = int(n_gt - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return dict(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            tp=int(tp), fp=int(fp), fn=int(fn),
            score_thr=float(score_thr),
        )

    @staticmethod
    def _prf_sweep(scores: np.ndarray,
                   matches: np.ndarray,
                   dt_ignore: np.ndarray,
                   n_gt: int,
                   thr_grid: Optional[np.ndarray] = None
                   ) -> Dict[str, float]:
        """Sweeps the score threshold and returns the F1-optimal point.

        The sweep grid defaults to every observed detection score (no
        information loss) but is capped at 1000 evenly-spaced quantiles
        when more than 1000 detections exist, which is sufficient
        resolution for an F1 curve and keeps the operation O(N log N).
        """
        if scores.size == 0 or n_gt == 0:
            return dict(precision=0.0, recall=0.0, f1=0.0,
                        tp=0, fp=0, fn=int(n_gt),
                        score_thr=0.0)

        if thr_grid is None:
            uniq = np.unique(scores)
            if uniq.size > 1000:
                idx = np.linspace(0, uniq.size - 1, 1000).astype(int)
                thr_grid = uniq[idx]
            else:
                thr_grid = uniq

        best = None
        for thr in thr_grid:
            stats = PalmBenchmarkMetric._prf_at_threshold(
                scores, matches, dt_ignore, n_gt, float(thr)
            )
            if (best is None) or (stats['f1'] > best['f1']):
                best = stats
        return best

    # -------------------------------------------------------------------
    # Final aggregation
    # -------------------------------------------------------------------
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Runs the COCO passes and returns a flat metric dict.

        Two evaluation passes are performed for each iou_type:

          (a) Full COCO pass on unfiltered predictions for mAP and
              mAP@50 (COCO protocol).

          (b) Single COCO pass at IoU = self.iou_thr on the *unfiltered*
              predictions, from which (i) the F1-optimal score threshold
              is found by sweeping the in-image score grid over the
              cached match arrays, and (ii) the fixed-threshold P/R/F1
              at score >= self.score_thr is reported for protocol
              traceability. Both numbers come from a single COCOeval
              call — no second evaluation is required.
        """
        # Flatten per-image prediction lists collected by `process`.
        bbox_preds: List[dict] = []
        segm_preds: List[dict] = []
        for r in results:
            bbox_preds.extend(r['bbox'])
            segm_preds.extend(r['segm'])

        out: Dict[str, float] = OrderedDict()

        for iou_type, preds in (('bbox', bbox_preds),
                                ('segm', segm_preds)):
            if iou_type not in self.metric_items:
                continue
            if iou_type == 'segm' and len(preds) == 0:
                # Bbox-only model; report zeros and continue without
                # corrupting the bbox track.
                for k in ('mAP', 'mAP_50',
                          'precision', 'recall', 'f1',
                          'precision_opt', 'recall_opt', 'f1_opt',
                          'best_score_thr',
                          'tp', 'fp', 'fn'):
                    out[f'segm_{k}'] = 0.0
                continue

            # ---- Pass 1: full COCO mAP on unfiltered predictions --------
            print(f'\n[metric] {iou_type}: '
                  f'computing mAP on {len(preds)} unfiltered predictions ...')
            coco_eval_full = self._run_coco_eval(
                self._coco_gt, preds, iou_type, iou_thrs=None,
                max_dets=self.max_dets,
            )
            ap_stats = self._ap_from(coco_eval_full)

            # ---- Pass 2: IoU = iou_thr only, retain match arrays --------
            # single_area=True collapses the COCO area-range partition so
            # the gathered per-detection match arrays are counted once,
            # not four times (see _run_coco_eval docstring).
            print(f'[metric] {iou_type}: '
                  f'computing P/R/F1 on {len(preds)} predictions '
                  f'at IoU = {self.iou_thr}')
            coco_eval_iou = self._run_coco_eval(
                self._coco_gt, preds, iou_type,
                iou_thrs=np.array([self.iou_thr]),
                single_area=True,
                max_dets=self.max_dets,
            )

            if coco_eval_iou is None:
                fixed_stats = dict(precision=0.0, recall=0.0, f1=0.0,
                                   tp=0, fp=0, fn=0,
                                   score_thr=float(self.score_thr))
                opt_stats = dict(precision=0.0, recall=0.0, f1=0.0,
                                 tp=0, fp=0, fn=0, score_thr=0.0)
            else:
                gathered = self._gather_matches(coco_eval_iou)
                # (i) protocol-fixed threshold
                fixed_stats = self._prf_at_threshold(
                    gathered['scores'], gathered['matches'],
                    gathered['dt_ignore'], gathered['n_gt'],
                    float(self.score_thr),
                )
                # (ii) Operating point.
                #   * applied_score_thr given  -> report P/R/F1 at that
                #     fixed (typically validation-selected) threshold;
                #     no test-set tuning.
                #   * applied_score_thr None    -> sweep the score grid on
                #     the test set for the F1-maximising point (an oracle
                #     upper bound; flag as test-tuned in the manuscript).
                if self.applied_score_thr is not None:
                    opt_stats = self._prf_at_threshold(
                        gathered['scores'], gathered['matches'],
                        gathered['dt_ignore'], gathered['n_gt'],
                        float(self.applied_score_thr),
                    )
                    print(f'[metric] {iou_type}: '
                          f'applied (val-selected) threshold = '
                          f'{opt_stats["score_thr"]:.3f}  '
                          f'F1 = {opt_stats["f1"]:.4f}  '
                          f'(P = {opt_stats["precision"]:.4f}, '
                          f'R = {opt_stats["recall"]:.4f})')
                else:
                    opt_stats = self._prf_sweep(
                        gathered['scores'], gathered['matches'],
                        gathered['dt_ignore'], gathered['n_gt'],
                    )
                    print(f'[metric] {iou_type}: '
                          f'F1-optimal threshold (test-swept) = '
                          f'{opt_stats["score_thr"]:.3f}  '
                          f'F1 = {opt_stats["f1"]:.4f}  '
                          f'(P = {opt_stats["precision"]:.4f}, '
                          f'R = {opt_stats["recall"]:.4f})')

            out[f'{iou_type}_mAP'] = ap_stats['mAP']
            out[f'{iou_type}_mAP_50'] = ap_stats['mAP_50']

            # Fixed-threshold (protocol) numbers
            out[f'{iou_type}_precision'] = fixed_stats['precision']
            out[f'{iou_type}_recall'] = fixed_stats['recall']
            out[f'{iou_type}_f1'] = fixed_stats['f1']
            out[f'{iou_type}_tp'] = fixed_stats['tp']
            out[f'{iou_type}_fp'] = fixed_stats['fp']
            out[f'{iou_type}_fn'] = fixed_stats['fn']

            # F1-optimal (sweep) numbers — primary for publication
            out[f'{iou_type}_precision_opt'] = opt_stats['precision']
            out[f'{iou_type}_recall_opt'] = opt_stats['recall']
            out[f'{iou_type}_f1_opt'] = opt_stats['f1']
            out[f'{iou_type}_best_score_thr'] = opt_stats['score_thr']

        return out


# ===========================================================================
# 2. Config patching
# ===========================================================================
#
# The config's `test_dataloader` and `test_evaluator` are rewritten so
# that (a) the user's CLI overrides take effect without editing config
# files, and (b) the evaluator becomes the PalmBenchmarkMetric defined
# above. All other config sections (model, test_pipeline) are untouched.
# ===========================================================================

def patch_config(cfg: Config,
                 ann_file: Optional[str],
                 img_prefix: Optional[str],
                 batch_size: Optional[int],
                 num_workers: Optional[int],
                 score_thr: float,
                 iou_thr: float,
                 work_dir: str,
                 applied_score_thr: Optional[float] = None,
                 max_dets: int = 100) -> Config:
    """Returns a patched copy of `cfg` ready for Runner.test().

    applied_score_thr is forwarded to PalmBenchmarkMetric: when not None,
    the reported operating point is fixed at that (validation-selected)
    threshold instead of being swept on the test set. Defaults to None,
    so existing callers are unaffected.

    max_dets is forwarded to PalmBenchmarkMetric and sets the per-image
    detection cap for both COCO passes. Defaults to 100 (COCO standard);
    raise it for dense tiles to remove the cap-induced recall ceiling.
    """
    cfg = copy.deepcopy(cfg)

    cfg.work_dir = work_dir
    cfg.load_from = None  # Runner.test() will load from the checkpoint arg.

    # ---- DataLoader overrides --------------------------------------------
    test_dl = cfg.test_dataloader
    if batch_size is not None:
        test_dl['batch_size'] = int(batch_size)
    if num_workers is not None:
        test_dl['num_workers'] = int(num_workers)
        test_dl['persistent_workers'] = num_workers > 0
    test_dl.setdefault('pin_memory', True)

    ds = test_dl['dataset']
    # Some training configs leave test_dataloader.dataset as a wrapper
    # (ConcatDataset of val_GE+val_aerial, RepeatDataset, etc.). Injecting
    # ann_file/data_prefix into a wrapper is invalid (ConcatDataset.__init__
    # has no such kwargs) and silently mixes sources. For single-sensor test
    # evaluation we REPLACE any wrapper with a fresh single CocoDataset on the
    # requested ann_file/img_prefix, using the config's standard test_pipeline.
    _ds_type = ds.get('type', '') if isinstance(ds, dict) else ''
    _is_wrapper = _ds_type in ('ConcatDataset', 'RepeatDataset',
                               'ClassBalancedDataset', 'MultiImageMixDataset') \
                  or 'datasets' in ds or ('dataset' in ds and 'ann_file' not in ds)
    if _is_wrapper or ann_file is not None or img_prefix is not None:
        # pull a clean pipeline + metainfo from the config (not the wrapper)
        pipeline = cfg.get('test_pipeline', None)
        if pipeline is None:
            # fall back to a leaf's pipeline if test_pipeline is absent
            leaf = ds
            while isinstance(leaf, dict) and ('datasets' in leaf or 'dataset' in leaf):
                leaf = (leaf['datasets'][0] if 'datasets' in leaf else leaf['dataset'])
            pipeline = leaf.get('pipeline') if isinstance(leaf, dict) else None
        metainfo = dict(classes=('DatePalm',))
        new_ds = dict(
            type='CocoDataset',
            ann_file=ann_file if ann_file is not None else ds.get('ann_file'),
            data_prefix=dict(img=img_prefix) if img_prefix is not None
                        else ds.get('data_prefix', {}),
            data_root='',
            metainfo=metainfo,
            test_mode=True,
            pipeline=pipeline,
        )
        test_dl['dataset'] = new_ds
        ds = new_ds

    # Resolve the absolute ann_file the metric needs.
    resolved_ann = ds['ann_file']
    if not osp.isabs(resolved_ann) and ds.get('data_root'):
        resolved_ann = osp.join(ds['data_root'], resolved_ann)

    # ---- Replace evaluator -----------------------------------------------
    cfg.test_evaluator = dict(
        type='PalmBenchmarkMetric',
        ann_file=resolved_ann,
        score_thr=float(score_thr),
        iou_thr=float(iou_thr),
        metric_items=('bbox', 'segm'),
        max_dets=int(max_dets),
    )
    if applied_score_thr is not None:
        cfg.test_evaluator['applied_score_thr'] = float(applied_score_thr)
    # Validation evaluator may share the same plumbing if the config
    # references it elsewhere; harmless if unused during test.
    if 'val_evaluator' in cfg:
        cfg.val_evaluator = copy.deepcopy(cfg.test_evaluator)

    # ---- Drop training-only custom hooks ---------------------------------
    # Training configs may carry custom_hooks (e.g. PalmBenchmarkLoggingHook,
    # EarlyStoppingHook) registered via the config's own custom_imports.
    # Runner.test() needs none of them, and the runner builds every declared
    # hook at construction — so a custom hook whose module is not imported at
    # eval time raises KeyError before testing begins. Removing custom_hooks
    # makes evaluation independent of training instrumentation and pre-empts
    # that failure for every backbone, not just the ones seen to fail.
    cfg.custom_hooks = []

    return cfg


# ===========================================================================
# 3. Reporting
# ===========================================================================

def print_summary(metrics: Dict[str, float],
                  config_path: str,
                  ckpt_path: str,
                  ann_path: str,
                  score_thr: float,
                  iou_thr: float) -> None:
    line1 = '=' * 76
    line2 = '-' * 76
    print(f'\n{line1}')
    print('  DATE PALM BENCHMARK — TEST EVALUATION SUMMARY')
    print(line2)
    print(f'  Model      : {osp.basename(config_path)}')
    print(f'  Checkpoint : {osp.basename(ckpt_path)}')
    print(f'  Test set   : {osp.basename(ann_path)}')
    print(f'  Protocol   : score_thr = {score_thr}   iou_thr = {iou_thr}')
    print(line2)
    print(f'  {"Metric":<42}{"BBox":>16}{"Segm":>16}')
    print(line2)

    def _row(label: str, key_b: str, key_s: str, fmt: str = '{:>16.4f}'):
        b = metrics.get(key_b, float('nan'))
        s = metrics.get(key_s, float('nan'))
        print(f'  {label:<42}{fmt.format(b)}{fmt.format(s)}')

    _row('mAP @ IoU 0.50:0.95   [supplementary]',
         'bbox_mAP', 'segm_mAP')
    _row('mAP @ IoU 0.50        [PRIMARY]',
         'bbox_mAP_50', 'segm_mAP_50')
    print(line2)
    print(f'  F1-OPTIMAL OPERATING POINT  [PRIMARY]')
    best_b = metrics.get('bbox_best_score_thr', float('nan'))
    best_s = metrics.get('segm_best_score_thr', float('nan'))
    print(f'  {"Best score threshold":<42}{best_b:>16.3f}{best_s:>16.3f}')
    _row('Precision @ best score, IoU 0.50',
         'bbox_precision_opt', 'segm_precision_opt')
    _row('Recall    @ best score, IoU 0.50',
         'bbox_recall_opt', 'segm_recall_opt')
    _row('F1-score  @ best score, IoU 0.50',
         'bbox_f1_opt', 'segm_f1_opt')
    print(line2)
    print(f'  FIXED THRESHOLD score = {score_thr}  [protocol traceability]')
    _row('Precision @ fixed score, IoU 0.50',
         'bbox_precision', 'segm_precision')
    _row('Recall    @ fixed score, IoU 0.50',
         'bbox_recall', 'segm_recall')
    _row('F1-score  @ fixed score, IoU 0.50',
         'bbox_f1', 'segm_f1')
    print(line2)
    _row('TP / FP / FN  (TP)',
         'bbox_tp', 'segm_tp', fmt='{:>16d}')
    _row('              (FP)',
         'bbox_fp', 'segm_fp', fmt='{:>16d}')
    _row('              (FN)',
         'bbox_fn', 'segm_fn', fmt='{:>16d}')
    print(line1)
    print('  Cross-check: bbox/segm mAP_50 above must match the training')
    print('  log keys  coco/bbox_mAP_50  and  coco/segm_mAP_50.')
    print(line1)


def write_csv(metrics: Dict[str, float],
              out_path: str,
              config_stem: str,
              ann_stem: str,
              score_thr: float,
              iou_thr: float) -> None:
    import csv
    out_dir = osp.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows = [
        ('mAP@50 [PRIMARY]',
         metrics.get('bbox_mAP_50'), metrics.get('segm_mAP_50')),
        ('F1@0.5 (optimal threshold) [PRIMARY]',
         metrics.get('bbox_f1_opt'), metrics.get('segm_f1_opt')),
        ('Precision@0.5 (optimal threshold)',
         metrics.get('bbox_precision_opt'), metrics.get('segm_precision_opt')),
        ('Recall@0.5 (optimal threshold)',
         metrics.get('bbox_recall_opt'), metrics.get('segm_recall_opt')),
        ('Best score threshold',
         metrics.get('bbox_best_score_thr'),
         metrics.get('segm_best_score_thr')),
        ('F1@0.5 (fixed score=0.05)',
         metrics.get('bbox_f1'), metrics.get('segm_f1')),
        ('Precision@0.5 (fixed score=0.05)',
         metrics.get('bbox_precision'), metrics.get('segm_precision')),
        ('Recall@0.5 (fixed score=0.05)',
         metrics.get('bbox_recall'), metrics.get('segm_recall')),
        ('mAP@[.5:.95] [suppl.]',
         metrics.get('bbox_mAP'), metrics.get('segm_mAP')),
        ('TP (fixed score=0.05)',
         metrics.get('bbox_tp'), metrics.get('segm_tp')),
        ('FP (fixed score=0.05)',
         metrics.get('bbox_fp'), metrics.get('segm_fp')),
        ('FN (fixed score=0.05)',
         metrics.get('bbox_fn'), metrics.get('segm_fn')),
    ]
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'metric', 'bbox', 'segm',
                    'score_thr', 'iou_thr', 'test_set'])
        for label, b, s in rows:
            b_str = (f'{b:.4f}' if isinstance(b, float)
                     else ('' if b is None else str(b)))
            s_str = (f'{s:.4f}' if isinstance(s, float)
                     else ('' if s is None else str(s)))
            w.writerow([config_stem, label, b_str, s_str,
                        score_thr, iou_thr, ann_stem])
    print(f'\n  Results saved -> {out_path}\n')


# ===========================================================================
# 4. CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Date Palm Benchmark — fast and accurate '
                    'mAP@50 + Precision / Recall / F1 evaluation.')
    p.add_argument('--config', required=True,
                   help='MMDetection config (.py).')
    p.add_argument('--checkpoint', required=True,
                   help='Trained model checkpoint (.pth).')
    p.add_argument('--ann-file', default=None,
                   help='Override the COCO annotation JSON in the '
                        'config\'s test_dataloader.')
    p.add_argument('--img-prefix', default=None,
                   help='Override the image directory in the config\'s '
                        'test_dataloader.')
    p.add_argument('--batch-size', type=int, default=None,
                   help='DataLoader batch size (default: from config).')
    p.add_argument('--num-workers', type=int, default=None,
                   help='DataLoader workers (default: from config). '
                        'Use 0 if running inside Docker Desktop on WSL2 '
                        'with shared-memory issues.')
    p.add_argument('--score-thr', type=float, default=0.05,
                   help='F-score confidence threshold. '
                        'Protocol-locked at 0.05.')
    p.add_argument('--iou-thr', type=float, default=0.50,
                   help='F-score TP IoU threshold. '
                        'Protocol-locked at 0.50.')
    p.add_argument('--applied-score-thr', type=float, default=None,
                   help='Fixed operating threshold for the reported '
                        'P/R/F1 (optimal) block, typically the '
                        'F1-maximising threshold selected on a held-out '
                        'validation set. When omitted, the operating '
                        'point is swept on the test set (oracle upper '
                        'bound).')
    p.add_argument('--max-dets', type=int, default=100,
                   help='Per-image detection cap for both COCO passes. '
                        'COCO default 100 caps recall on dense tiles; '
                        'raise it (up to the model test_cfg max_per_img) '
                        'for dense plantation scenes.')
    p.add_argument('--out', default=None,
                   help='Output CSV path (default: '
                        'results/<config>__<ann>.csv).')
    p.add_argument('--work-dir', default=None,
                   help='Runner work_dir (default: '
                        '/tmp/eval_<config>).')
    p.add_argument('--cfg-options', nargs='+', action=DictAction,
                   help='Additional config overrides, e.g. '
                        'model.test_cfg.rcnn.score_thr=0.0')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if abs(args.score_thr - 0.05) > 1e-6:
        print(f'\n[WARN] score_thr = {args.score_thr} deviates from the '
              f'protocol-locked value of 0.05.\n')
    if abs(args.iou_thr - 0.50) > 1e-6:
        print(f'\n[WARN] iou_thr = {args.iou_thr} deviates from the '
              f'protocol-locked value of 0.50.\n')

    # ---- Load and patch config -------------------------------------------
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    config_stem = osp.splitext(osp.basename(args.config))[0]
    work_dir = args.work_dir or osp.join('/tmp', f'eval_{config_stem}')
    os.makedirs(work_dir, exist_ok=True)

    cfg = patch_config(
        cfg,
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        work_dir=work_dir,
        applied_score_thr=args.applied_score_thr,
        max_dets=args.max_dets,
    )

    # ---- Resolve paths for reporting -------------------------------------
    ds = cfg.test_dataloader['dataset']
    resolved_ann = ds['ann_file']
    if not osp.isabs(resolved_ann) and ds.get('data_root'):
        resolved_ann = osp.join(ds['data_root'], resolved_ann)
    ann_stem = osp.splitext(osp.basename(resolved_ann))[0]

    out_path = args.out or osp.join('results',
                                    f'{config_stem}__{ann_stem}.csv')

    # ---- Build runner and run test ---------------------------------------
    cfg.load_from = args.checkpoint  # consumed by Runner.test()

    print(f'\n[1/2] Building runner ...')
    print(f'      config     : {args.config}')
    print(f'      checkpoint : {args.checkpoint}')
    print(f'      ann_file   : {resolved_ann}')
    print(f'      batch_size : {cfg.test_dataloader["batch_size"]}')
    print(f'      num_workers: {cfg.test_dataloader["num_workers"]}')

    runner = Runner.from_cfg(cfg)

    print(f'\n[2/2] Running Runner.test() ...')
    metrics = runner.test()
    # Strip the metric prefix ("palm/") that mmengine prepends.
    metrics_clean: Dict[str, float] = {}
    for k, v in metrics.items():
        bare = k.split('/', 1)[-1] if '/' in k else k
        metrics_clean[bare] = v

    # ---- Report ----------------------------------------------------------
    print_summary(
        metrics_clean,
        config_path=args.config,
        ckpt_path=args.checkpoint,
        ann_path=resolved_ann,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
    )
    write_csv(
        metrics_clean,
        out_path=out_path,
        config_stem=config_stem,
        ann_stem=ann_stem,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
    )


if __name__ == '__main__':
    main()