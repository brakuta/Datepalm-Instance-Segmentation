# ==========================================================================
# efficiency.py
# ==========================================================================
# Computational-efficiency measurement for the date palm detection
# benchmark. Produces the per-model efficiency figures reported alongside
# the accuracy metrics: inference latency (mean / std), frames per second,
# parameter count, floating-point operations (FLOPs), and peak GPU memory.
#
# DESIGN RATIONALE
# ----------------
# Accuracy and efficiency cannot be measured in the same forward pass.
# The accuracy pass (Runner.test) deliberately maximises throughput via
# multi-worker batched data loading, cuDNN autotuning, and RLE mask
# encoding on the critical path; those choices are correct for mAP but
# make any latency reading non-comparable across backbones. Efficiency is
# therefore measured here under a single, fixed regime applied identically
# to every backbone, so the reported figures isolate the architecture and
# nothing else:
#
#   * single reference GPU (the caller is responsible for pinning it);
#   * batch size fixed at 1;
#   * input fixed at 1024 x 1024 (the benchmark inference size);
#   * precision fixed at fp32 (hardware-neutral and reproducible; held
#     constant across CNN, Transformer, and SSM families so that
#     Tensor-Core-favoured convolutions are not advantaged over the
#     element-wise selective-scan kernels of the Mamba backbones);
#   * a fixed number of warm-up iterations discarded before timing, so
#     cuDNN autotuning and lazy CUDA initialisation do not contaminate the
#     measurement;
#   * each timed iteration bracketed by torch.cuda.synchronize(), so the
#     measured interval reflects completed GPU work rather than the time
#     to enqueue asynchronous kernels;
#   * only the model forward pass is timed; data loading, host transfer,
#     and mask post-processing are excluded.
#
# Parameters and FLOPs are static properties and are computed once,
# decoupled from the runtime loop.
#
# FLOPs HONESTY
# -------------
# Standard FLOP counters (fvcore) do not register the custom selective-
# scan CUDA kernels used by the SSM backbones (VMamba, SpatialMamba,
# GroupMamba, EfficientVMamba) nor every operator in MambaVision. A naive
# total would therefore undercount only the Mamba family — precisely the
# comparison this benchmark makes. This module captures fvcore's list of
# unsupported operators per model and records it. When a model reports
# uncounted operators, its FLOPs value is flagged as not directly
# comparable rather than presented as a clean number; the parameter count,
# which is exact for every family, carries the parameter-efficiency claim.
#
# Compatibility: MMDetection >= 3.0, mmengine >= 0.7, PyTorch 2.x.
# fvcore is optional; if absent, FLOPs are reported as unavailable and the
# remaining efficiency metrics are produced normally.
# ==========================================================================

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Protocol constants (LOCKED — mirror the accuracy-side invariants)
# --------------------------------------------------------------------------
EFF_INPUT_HW: Tuple[int, int] = (1024, 1024)   # benchmark inference size
EFF_DTYPE: torch.dtype = torch.float32         # fixed precision (see header)
EFF_WARMUP: int = 20                           # discarded warm-up iterations
EFF_ITERS: int = 200                           # timed iterations


# --------------------------------------------------------------------------
# Parameter count
# --------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Returns total and trainable parameter counts (exact for all families)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(params_total=int(total), params_trainable=int(trainable))


# --------------------------------------------------------------------------
# Synthetic input matching MMDetection's expected test-time signature
# --------------------------------------------------------------------------
def _build_synthetic_batch(model: nn.Module,
                           device: str,
                           dtype: torch.dtype,
                           input_hw: Tuple[int, int]):
    """Constructs a single-image batch in the structure MMDet expects.

    MMDetection detectors are called at inference as
    ``model.test_step(data)`` where ``data`` is the collated output of the
    model's own data preprocessor input format: a dict with 'inputs'
    (list of CHW uint8/float tensors) and 'data_samples' (list of
    DetDataSample carrying at least the image metainfo). A synthetic image
    and a minimal DetDataSample reproduce that signature without touching
    the dataloader, so the timing reflects the network alone.
    """
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData

    h, w = input_hw
    # Float CHW image; the model's data_preprocessor handles normalisation.
    img = torch.randn(3, h, w, dtype=dtype, device=device)

    sample = DetDataSample()
    sample.set_metainfo(dict(
        img_id=0,
        img_path='<synthetic>',
        ori_shape=(h, w),
        img_shape=(h, w),
        scale_factor=(1.0, 1.0),
        pad_shape=(h, w),
        batch_input_shape=(h, w),
    ))
    sample.gt_instances = InstanceData()

    return dict(inputs=[img], data_samples=[sample])


# --------------------------------------------------------------------------
# Latency / FPS / peak VRAM
# --------------------------------------------------------------------------
def measure_latency(model: nn.Module,
                    device: str,
                    dtype: torch.dtype = EFF_DTYPE,
                    input_hw: Tuple[int, int] = EFF_INPUT_HW,
                    warmup: int = EFF_WARMUP,
                    iters: int = EFF_ITERS) -> Dict[str, float]:
    """Times the forward pass under the fixed efficiency regime.

    The model must already be on ``device`` with its checkpoint loaded and
    be in eval mode. Returns per-image latency statistics (milliseconds),
    FPS, and peak allocated GPU memory (MB). All timing is performed under
    torch.no_grad with per-iteration CUDA synchronisation.
    """
    if not device.startswith('cuda'):
        raise RuntimeError(
            'Efficiency measurement requires a CUDA device; latency on CPU '
            'is not comparable to the reported GPU figures.')

    model.eval()
    # Cast the network to the fixed efficiency precision. The data
    # preprocessor inside the detector keeps its own buffers; casting the
    # whole module is sufficient because fp32 is the default and this is a
    # no-op when the checkpoint is already fp32.
    if dtype == torch.float32:
        model.float()

    data = _build_synthetic_batch(model, device, dtype, input_hw)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # ---- Warm-up (discarded): triggers cuDNN autotuning + lazy init -----
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.test_step(data)
    torch.cuda.synchronize(device)

    # ---- Timed iterations ----------------------------------------------
    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model.test_step(data)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    mean_ms = float(arr.mean())
    return dict(
        latency_mean_ms=mean_ms,
        latency_std_ms=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        latency_p50_ms=float(np.percentile(arr, 50)),
        latency_p90_ms=float(np.percentile(arr, 90)),
        fps=float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
        peak_vram_mb=float(peak_mb),
        eff_iters=int(iters),
        eff_warmup=int(warmup),
        eff_dtype=str(dtype).replace('torch.', ''),
        eff_input_h=int(input_hw[0]),
        eff_input_w=int(input_hw[1]),
    )


# --------------------------------------------------------------------------
# FLOPs (fvcore, with honest per-operator coverage reporting)
# --------------------------------------------------------------------------
def measure_flops(model: nn.Module,
                  device: str,
                  dtype: torch.dtype = EFF_DTYPE,
                  input_hw: Tuple[int, int] = EFF_INPUT_HW
                  ) -> Dict[str, object]:
    """Estimates forward FLOPs for one 1024x1024 image via fvcore.

    Returns a dict carrying the GFLOPs value, a boolean indicating whether
    any operator was left uncounted, and the list of uncounted operator
    names. When uncounted operators are present (typical for SSM / Mamba
    selective-scan kernels), the GFLOPs value undercounts the true cost
    and must be footnoted as not directly comparable for that backbone.

    fvcore counts multiply-accumulate operations (MACs). The convention in
    detection efficiency tables is to report these MACs as 'FLOPs'; that
    convention is followed here and stated in the output so the unit is
    unambiguous (1 MAC is reported as 1 FLOP, not 2).
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return dict(
            gflops=None,
            flops_uncounted=True,
            flops_uncounted_ops=['fvcore-not-installed'],
            flops_note='fvcore not installed; FLOPs unavailable.',
        )

    model.eval()
    if dtype == torch.float32:
        model.float()

    h, w = input_hw
    img = torch.randn(1, 3, h, w, dtype=dtype, device=device)

    # MMDet detectors are not callable on a raw tensor, and the detector's
    # _forward(batch_inputs, batch_data_samples) requires data samples that
    # fvcore cannot synthesise during tracing. The backbone + neck, reached
    # via extract_feat(batch_inputs), takes only the image tensor and is
    # precisely the part that differs across the CNN / Transformer / Mamba
    # families being compared. The shared Mask R-CNN + FPN RPN and ROI heads
    # are architecturally identical across all backbones, so reporting
    # backbone+neck FLOPs gives a fair, well-defined cross-architecture
    # comparison and avoids the data-sample dependency. The reported scope
    # is recorded in the note so the manuscript can state it.
    class _TraceWrapper(nn.Module):
        def __init__(self, det):
            super().__init__()
            self.det = det

        def forward(self, x):
            return self.det.extract_feat(x)

    wrapper = _TraceWrapper(model).to(device).eval()

    flops_scope = 'backbone+neck (extract_feat)'

    try:
        with torch.no_grad():
            fca = FlopCountAnalysis(wrapper, img)
            # Silence fvcore's per-call warnings; we collect them instead.
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            total_flops = fca.total()
            unsupported = dict(fca.unsupported_ops())
    except Exception as exc:  # noqa: BLE001
        return dict(
            gflops=None,
            flops_uncounted=True,
            flops_uncounted_ops=[f'trace-failed:{type(exc).__name__}'],
            flops_note=f'fvcore trace failed: {exc}',
            flops_scope=flops_scope,
        )

    uncounted_ops = sorted(unsupported.keys())
    has_uncounted = len(uncounted_ops) > 0
    gflops = float(total_flops) / 1e9

    note = f'FLOPs scope: {flops_scope}.'
    if has_uncounted:
        note += (' FLOPs undercount the true cost: the following operators '
                 'were not registered by fvcore and are excluded — '
                 + ', '.join(uncounted_ops) + '. Not directly comparable.')

    return dict(
        gflops=gflops,
        flops_uncounted=has_uncounted,
        flops_uncounted_ops=uncounted_ops,
        flops_note=note,
        flops_scope=flops_scope,
    )


# --------------------------------------------------------------------------
# Top-level convenience: full efficiency profile for a built model
# --------------------------------------------------------------------------
def profile_model(model: nn.Module,
                  device: str,
                  dtype: torch.dtype = EFF_DTYPE,
                  input_hw: Tuple[int, int] = EFF_INPUT_HW,
                  warmup: int = EFF_WARMUP,
                  iters: int = EFF_ITERS,
                  measure_flops_too: bool = True) -> Dict[str, object]:
    """Computes the complete efficiency profile for one checkpoint-loaded model.

    Combines parameter count, FLOPs (optional), and the timed latency /
    FPS / peak-VRAM measurement into a single flat dict whose keys are the
    column stems consumed by the CSV writer. All measurements share the
    fixed regime documented at module top.
    """
    profile: Dict[str, object] = {}
    profile.update(count_parameters(model))

    if measure_flops_too:
        profile.update(measure_flops(model, device, dtype, input_hw))

    # Latency last: it sets the peak-memory high-water mark we report.
    profile.update(measure_latency(
        model, device, dtype=dtype, input_hw=input_hw,
        warmup=warmup, iters=iters))

    return profile
