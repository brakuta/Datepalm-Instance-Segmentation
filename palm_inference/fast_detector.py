"""Fast batched detector with GPU-side mask cropping.

Key optimisations versus the original:
  * Masks are cropped to their bounding boxes on the GPU before transfer
    to host memory. Typical palm masks are ~80 px across; transferring
    (N, 1024, 1024) bool tensors was the dominant PCIe cost per batch
    (~200 MB at N=200). Bbox-cropped transfer reduces this by 40–100x.
  * The explicit torch.cuda.synchronize() is removed. CUDA's implicit
    sync on .cpu() transfer is sufficient once we are no longer moving
    large device tensors that could race with the next launch.
  * Pinned host staging buffer + non_blocking H2D kept for fast upload.
  * FP16 autocast kept for Tensor Core acceleration.

Output format: for each tile, a dict with
    scores : (N,) cpu float32 tensor
    bboxes : (N, 4) cpu float32 tensor, [x1,y1,x2,y2] in tile-local px
    masks  : list[np.uint8 array of shape (hi,wi)] — N cropped binary masks,
             one per detection, at the bbox resolution
    mask_bboxes : (N, 4) int32 array — integer bbox [x1,y1,x2,y2] used for
                  cropping, so the vectoriser can re-project mask-local
                  coordinates to tile-local.
"""
from __future__ import annotations
import logging
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger("palm_inference.fast_detector")


class FastBatchedDetector:
    _MEAN_BGR = [103.530, 116.280, 123.675]
    _STD_BGR  = [57.375,  57.120,  58.395]

    def __init__(self, cfg):
        from mmengine.config import Config as MMConfig
        from mmengine.registry import init_default_scope
        from mmdet.apis import init_detector
        init_default_scope("mmdet")
        mm_cfg = MMConfig.fromfile(str(cfg.config_file))
        self.model = init_detector(
            mm_cfg, str(cfg.checkpoint_file), device=cfg.device)
        self.model.eval()
        self.device = torch.device(cfg.device)
        self.use_fp16 = cfg.use_fp16
        self.score_threshold = cfg.score_threshold

        # NMS threshold overrides at inference time.
        # IMPORTANT: The validated training protocol uses rcnn iou_threshold=0.7.
        # Overriding to 0.4 merges adjacent palms in dense plantation scenes.
        # These overrides default to None (= use model's trained value) so that
        # inference behaviour matches the training evaluation protocol.
        rcnn_iou_override: Optional[float] = getattr(cfg, "rcnn_nms_iou_threshold", None)
        rpn_iou_override:  Optional[float] = getattr(cfg, "rpn_nms_iou_threshold",  None)

        if rcnn_iou_override is not None:
            try:
                self.model.test_cfg.rcnn.nms.iou_threshold = rcnn_iou_override
                logger.info(f"RoI NMS threshold overridden to {rcnn_iou_override}")
            except Exception as e:
                logger.warning(f"Could not override RoI NMS threshold: {e}")
        else:
            try:
                current = self.model.test_cfg.rcnn.nms.iou_threshold
                logger.info(f"RoI NMS threshold: {current} (from model config — not overridden)")
            except Exception:
                pass

        if rpn_iou_override is not None:
            try:
                self.model.test_cfg.rpn.nms.iou_threshold = rpn_iou_override
                logger.info(f"RPN NMS threshold overridden to {rpn_iou_override}")
            except Exception as e:
                logger.warning(f"Could not override RPN NMS threshold: {e}")
        else:
            try:
                current = self.model.test_cfg.rpn.nms.iou_threshold
                logger.info(f"RPN NMS threshold: {current} (from model config — not overridden)")
            except Exception:
                pass

        dev = self.device
        self._mean = torch.tensor(self._MEAN_BGR, dtype=torch.float32, device=dev).view(3, 1, 1)
        self._std  = torch.tensor(self._STD_BGR,  dtype=torch.float32, device=dev).view(3, 1, 1)
        self._pin_buf   = None
        self._pin_shape = None
        logger.info(f"FastBatchedDetector ready | fp16={'on' if self.use_fp16 else 'off'}")

    def _get_pinned_buffer(self, shape):
        if self._pin_buf is None or self._pin_shape != shape:
            self._pin_buf   = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
            self._pin_shape = shape
        return self._pin_buf

    @torch.inference_mode()
    def predict_batch(self, tiles_bgr: List[np.ndarray]) -> list:
        if not tiles_bgr:
            return []
        B    = len(tiles_bgr)
        H, W = tiles_bgr[0].shape[:2]

        # Pinned staging buffer -> non-blocking H2D
        buf = self._get_pinned_buffer((B, H, W, 3))
        for i, tile in enumerate(tiles_bgr):
            buf[i].copy_(torch.from_numpy(np.ascontiguousarray(tile)))
        gpu = buf.to(self.device, non_blocking=True)

        # uint8 HWC BGR -> float32 NCHW, normalise on GPU
        x = gpu.permute(0, 3, 1, 2).float()
        x = (x - self._mean) / self._std

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_fp16):
            from mmdet.structures import DetDataSample
            data_samples = []
            for _ in range(B):
                ds = DetDataSample()
                ds.set_metainfo({
                    "img_shape":         (H, W),
                    "ori_shape":         (H, W),
                    "scale_factor":      np.array([1., 1., 1., 1.], dtype=np.float32),
                    "batch_input_shape": (H, W),
                })
                data_samples.append(ds)

            feats   = self.model.backbone(x)
            feats   = self.model.neck(feats)
            props   = self.model.rpn_head.predict(feats, data_samples, rescale=False)
            results = self.model.roi_head.predict(feats, props, data_samples, rescale=False)

        # Build per-tile output on the GPU, then transfer compact results to CPU.
        # Do NOT call torch.cuda.synchronize() here — the per-tensor .cpu() calls
        # below synchronise implicitly, and removing the explicit barrier lets
        # the next kernel launch begin earlier on the main stream.
        out = []
        score_thr = self.score_threshold
        for r in results:
            inst = r.pred_instances if hasattr(r, "pred_instances") else r
            scores_d = inst.scores
            # Early GPU-side score filter to reduce CPU transfer volume.
            keep = scores_d > score_thr
            if not keep.any():
                out.append({
                    "scores": torch.empty(0),
                    "bboxes": torch.empty(0, 4),
                    "masks": [],
                    "mask_bboxes": np.empty((0, 4), dtype=np.int32),
                })
                continue

            scores_d = scores_d[keep]
            bboxes_d = inst.bboxes[keep]
            masks_d  = inst.masks[keep] if hasattr(inst, "masks") and inst.masks is not None else None

            # Compute integer bboxes for cropping, clamped to tile dimensions.
            mb = bboxes_d.round().to(torch.int32)
            mb[:, 0].clamp_(0, W)
            mb[:, 1].clamp_(0, H)
            mb[:, 2].clamp_(0, W)
            mb[:, 3].clamp_(0, H)
            mb_cpu = mb.cpu().numpy()

            cropped_masks: List[np.ndarray] = []
            if masks_d is not None:
                # Crop each mask to its bbox on GPU, then transfer the small
                # crop. This is the dominant PCIe-bandwidth saving.
                masks_u8 = masks_d.to(torch.uint8)  # still on GPU
                for k in range(masks_u8.shape[0]):
                    x1, y1, x2, y2 = mb_cpu[k]
                    if x2 <= x1 or y2 <= y1:
                        cropped_masks.append(np.zeros((1, 1), dtype=np.uint8))
                        continue
                    crop = masks_u8[k, y1:y2, x1:x2].contiguous().cpu().numpy()
                    cropped_masks.append(crop)
            else:
                cropped_masks = [np.zeros((1, 1), dtype=np.uint8)] * scores_d.shape[0]

            out.append({
                "scores": scores_d.cpu(),
                "bboxes": bboxes_d.cpu(),
                "masks":  cropped_masks,
                "mask_bboxes": mb_cpu,
            })
        return out
