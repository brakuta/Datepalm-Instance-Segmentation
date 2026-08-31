# Copyright (c) Stage C Date Palm Benchmark.
# nms_fp32_guard.py
# ==========================================================================
# AMP-safe wrapper for batched_nms.
#
# PROBLEM
# -------
# Under AmpOptimWrapper(float16) the RPN head emits half-precision boxes and
# scores. Inside batched_nms, the post-NMS write-back
#     scores_after_nms[mask[keep]] = dets[:, -1]
# fails when destination (Half) and source (Float) dtypes disagree:
#     RuntimeError: Index put requires the source and destination dtypes
#     match, got Half for the destination and Float for the source.
#
# FIX
# ---
# Wrap batched_nms so boxes/scores are upcast to float32, NMS runs with
# autocast disabled, and the returned dets are cast back to the input dtype
# so the write-back is dtype-consistent. Architecture-preserving; no model,
# backbone, or detector code is changed. No-op fast-path under FP32.
#
# IMPORT-RESOLUTION NOTE
# ----------------------
# `import mmcv.ops.nms as m; m.batched_nms` is NOT reliable: in some mmcv
# builds the name `mmcv.ops.nms` resolves to the re-exported `nms` *function*
# rather than the submodule, so `.batched_nms` raises AttributeError (this is
# exactly the failure this file previously hit). We therefore resolve the
# original callable from the public `mmcv.ops` package (and from the submodule
# object fetched via importlib only as a fallback), then rebind it in every
# namespace that may already hold a reference -- most importantly mmdet's RPN
# head, which is the actual caller in the traceback.
# ==========================================================================

import functools
import importlib

import torch


def _resolve_original():
    """Return (orig_callable, list_of_(module, attname)_to_patch)."""
    targets = []
    orig = None

    # Public package export -- the reliable source of the callable.
    import mmcv.ops as mmcv_ops
    if hasattr(mmcv_ops, 'batched_nms'):
        orig = mmcv_ops.batched_nms
        targets.append((mmcv_ops, 'batched_nms'))

    # The submodule OBJECT (via importlib, so we get the module, not a
    # possibly re-exported function bound to the same dotted name).
    try:
        nms_mod = importlib.import_module('mmcv.ops.nms')
        if hasattr(nms_mod, 'batched_nms'):
            if orig is None:
                orig = nms_mod.batched_nms
            targets.append((nms_mod, 'batched_nms'))
    except Exception:
        pass

    if orig is None:
        raise ImportError(
            'nms_fp32_guard: could not resolve batched_nms from mmcv.ops'
        )
    return orig, targets


_ORIG_BATCHED_NMS, _PATCH_TARGETS = _resolve_original()


@functools.wraps(_ORIG_BATCHED_NMS)
def batched_nms_fp32(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """float32-consistent batched_nms (see module docstring)."""
    orig_dtype = scores.dtype
    boxes_f32 = boxes.float()
    scores_f32 = scores.float()

    with torch.cuda.amp.autocast(enabled=False):
        dets, keep = _ORIG_BATCHED_NMS(
            boxes_f32, scores_f32, idxs, nms_cfg, class_agnostic
        )

    if dets.dtype != orig_dtype:
        dets = dets.to(orig_dtype)
    return dets, keep


def _patch_everywhere():
    patched = []

    # 1. mmcv namespaces resolved above.
    for module, attr in _PATCH_TARGETS:
        setattr(module, attr, batched_nms_fp32)
        patched.append(f'{module.__name__}.{attr}')

    # 2. mmdet modules that did `from mmcv.ops import batched_nms` at import
    #    time -- rebinding mmcv alone would not touch these already-bound names.
    for modname in (
        'mmdet.models.dense_heads.rpn_head',
        'mmdet.models.dense_heads.base_dense_head',
        'mmdet.models.layers',                 # some versions re-export here
        'mmdet.models.task_modules.coders',    # defensive; harmless if absent
    ):
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, 'batched_nms'):
                mod.batched_nms = batched_nms_fp32
                patched.append(f'{modname}.batched_nms')
        except Exception:
            pass

    return patched


_PATCHED = _patch_everywhere()
print(f'[nms_fp32_guard] batched_nms wrapped for AMP dtype safety -> '
      f'patched: {_PATCHED}')