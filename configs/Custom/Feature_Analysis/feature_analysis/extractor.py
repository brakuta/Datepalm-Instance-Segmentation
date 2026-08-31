# -*- coding: utf-8 -*-
"""Model-intact feature capture.

Intermediate features are obtained through PyTorch forward hooks registered on
``model.backbone`` and ``model.neck``; the detector architecture and trained
weights are never modified. MMDetection / torch are imported lazily so that the
analysis and configuration code can run without them.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def to_chw_numpy(fmap) -> np.ndarray:
    """Accept a (1,C,H,W) or (C,H,W) tensor/array; return (C,H,W) float32."""
    arr = fmap.numpy() if hasattr(fmap, "numpy") else np.asarray(fmap)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.float32)


class FeatureExtractor:
    """Build a detector, hook its backbone and neck, and return the captured
    multi-level feature maps for a single forward pass through the standard
    MMDet test pipeline. One detector is resident per instance; call
    :meth:`close` to release it before building the next."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str):
        from mmdet.apis import init_detector
        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.model.eval()
        self._store: Dict[str, Tuple] = {}
        self._handles = []
        self._register()

    def _register(self) -> None:
        def make_hook(name: str):
            def hook(_module, _inp, out):
                if isinstance(out, (list, tuple)):
                    self._store[name] = tuple(o.detach().float().cpu() for o in out)
                else:
                    self._store[name] = (out.detach().float().cpu(),)
            return hook

        self._handles.append(self.model.backbone.register_forward_hook(make_hook("backbone")))
        if getattr(self.model, "neck", None) is None:
            raise RuntimeError("Model has no `neck`; FPN-level analysis requires an FPN.")
        self._handles.append(self.model.neck.register_forward_hook(make_hook("neck")))

    def _run(self, img_path: str) -> Dict[str, Tuple]:
        import torch
        from mmdet.apis import inference_detector
        self._store.clear()
        with torch.no_grad():
            _ = inference_detector(self.model, img_path)
        return self._store

    def extract(self, img_path: str) -> Tuple:
        """Return the FPN-level feature tuple ((1,C,H,W) per level)."""
        store = self._run(img_path)
        if "neck" not in store:
            raise RuntimeError("Neck features were not captured during inference.")
        return store["neck"]

    def extract_all(self, img_path: str) -> Dict[str, Tuple]:
        """Return both backbone-stage and FPN-level feature tuples."""
        store = self._run(img_path)
        if "backbone" not in store:
            raise RuntimeError("Backbone features were not captured during inference.")
        return dict(store)

    def close(self) -> None:
        import torch
        for h in self._handles:
            h.remove()
        self._handles.clear()
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
