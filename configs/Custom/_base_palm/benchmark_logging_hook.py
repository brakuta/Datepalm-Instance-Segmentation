# ==========================================================================
# configs/Custom/_base_palm/benchmark_logging_hook.py
# --------------------------------------------------------------------------
# Logs benchmark metadata for publication-grade model comparison.
#
# Reports:
#   - total detector parameters (total and trainable)
#   - backbone / neck / head parameter breakdown
#   - estimated backbone+neck FLOPs / GFLOPs for a fixed input size
#   - training wall-clock time and seconds per iteration
#   - peak CUDA memory
#
# Usage — add to runtime_palm.py:
#
#   custom_imports = dict(
#       imports=['configs.Custom._base_palm.benchmark_logging_hook'],
#       allow_failed_imports=False,
#   )
#
#   custom_hooks = [
#       dict(type='EarlyStoppingHook', ...),
#       dict(
#           type='BenchmarkLoggingHook',
#           input_shape=(3, 1024, 1024),
#           compute_flops=True,
#           save_json=True,
#       ),
#   ]
#
# Notes:
#   - Parameter counts are exact.
#   - GFLOPs are estimated for backbone + neck only.
#   - Full Mask R-CNN detector FLOPs are not used because RPN/RoI operations
#     are proposal-dependent and require detector-specific data_samples.
#   - FLOPs must always be reported together with input_shape.
#   - Mamba/SSM custom ops may be partially unsupported by FLOPs profilers.
#   - after_train is @master_only for safe use in distributed training.
# ==========================================================================

import time
import json
from pathlib import Path

import torch
import torch.nn as nn

from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmengine.dist import master_only
from mmengine.model import is_model_wrapper


def _unwrap_model(model):
    """Unwrap DDP / AMP model wrappers to get the raw detector."""
    if is_model_wrapper(model):
        return model.module
    return model


def _count_params(module):
    """
    Return total and trainable parameter counts.

    Args:
        module: PyTorch module or None.

    Returns:
        tuple[int, int]: total_params, trainable_params.
    """
    if module is None:
        return 0, 0

    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

    return total, trainable


class BackboneNeckWrapper(nn.Module):
    """
    Lightweight wrapper for profiling backbone + neck only.

    This avoids profiling the full MMDetection detector, which normally
    expects data_samples, proposals, metainfo, and detector-specific forward
    logic. For backbone benchmarking, backbone+neck GFLOPs are more stable
    and more comparable across models.
    """

    def __init__(self, detector):
        super().__init__()
        self.backbone = detector.backbone
        self.neck = detector.neck

    def forward(self, x):
        feats = self.backbone(x)

        if self.neck is not None:
            feats = self.neck(feats)

        return feats


@HOOKS.register_module()
class PalmBenchmarkLoggingHook(Hook):
    """
    Benchmark logger for detector comparison studies.

    Logs:
        - total detector parameters
        - trainable detector parameters
        - backbone / neck / head parameter breakdown
        - estimated backbone+neck GFLOPs
        - wall-clock training time
        - seconds per iteration
        - peak CUDA memory

    Args:
        input_shape (tuple[int]): Input tensor shape as (C, H, W).
            Default: (3, 1024, 1024).
        compute_flops (bool): Whether to estimate backbone+neck GFLOPs
            before training. Default: True.
        save_json (bool): Whether to write benchmark_record.json to work_dir.
            Default: True.
    """

    priority = 'LOW'

    def __init__(
        self,
        input_shape=(3, 1024, 1024),
        compute_flops=True,
        save_json=True,
    ):
        self.input_shape = tuple(input_shape)
        self.compute_flops = compute_flops
        self.save_json = save_json

        self._start_time = None
        self._record = {}

    # ------------------------------------------------------------------
    # before_train: parameter counts + optional FLOPs
    # ------------------------------------------------------------------
    def before_train(self, runner):
        self._start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model = _unwrap_model(runner.model)

        # --------------------------------------------------------------
        # Parameter counts
        # --------------------------------------------------------------
        total_p, total_t = _count_params(model)

        bb_p, bb_t = _count_params(
            getattr(model, 'backbone', None)
        )

        neck_p, neck_t = _count_params(
            getattr(model, 'neck', None)
        )

        rpn_p, rpn_t = _count_params(
            getattr(model, 'rpn_head', None)
        )

        roi_p, roi_t = _count_params(
            getattr(model, 'roi_head', None)
        )

        head_p = rpn_p + roi_p
        head_t = rpn_t + roi_t

        self._record.update({
            'work_dir': str(runner.work_dir),
            'config': (
                Path(runner.cfg.filename).name
                if runner.cfg.filename else None
            ),

            # Input size used for complexity estimation
            'input_shape': self.input_shape,
            'input_height': self.input_shape[1],
            'input_width': self.input_shape[2],

            # Full detector parameters
            'total_detector_params': total_p,
            'trainable_detector_params': total_t,
            'total_detector_params_million': round(total_p / 1e6, 3),
            'trainable_detector_params_million': round(total_t / 1e6, 3),

            # Backbone parameters
            'backbone_params': bb_p,
            'trainable_backbone_params': bb_t,
            'backbone_params_million': round(bb_p / 1e6, 3),
            'trainable_backbone_params_million': round(bb_t / 1e6, 3),

            # Neck parameters
            'neck_params': neck_p,
            'trainable_neck_params': neck_t,
            'neck_params_million': round(neck_p / 1e6, 3),
            'trainable_neck_params_million': round(neck_t / 1e6, 3),

            # Head parameters
            'head_params': head_p,
            'trainable_head_params': head_t,
            'head_params_million': round(head_p / 1e6, 3),
            'trainable_head_params_million': round(head_t / 1e6, 3),

            # RPN / RoI split
            'rpn_params': rpn_p,
            'trainable_rpn_params': rpn_t,
            'rpn_params_million': round(rpn_p / 1e6, 3),
            'trainable_rpn_params_million': round(rpn_t / 1e6, 3),

            'roi_params': roi_p,
            'trainable_roi_params': roi_t,
            'roi_params_million': round(roi_p / 1e6, 3),
            'trainable_roi_params_million': round(roi_t / 1e6, 3),
        })

        runner.logger.info(
            '[BENCHMARK] '
            f'total={total_p:,} ({total_p / 1e6:.3f} M)  '
            f'trainable={total_t:,} ({total_t / 1e6:.3f} M)'
        )

        runner.logger.info(
            '[BENCHMARK] '
            f'backbone={bb_p / 1e6:.3f} M  '
            f'neck={neck_p / 1e6:.3f} M  '
            f'head={head_p / 1e6:.3f} M  '
            f'(rpn={rpn_p / 1e6:.3f} M  '
            f'roi={roi_p / 1e6:.3f} M)'
        )

        if self.compute_flops:
            self._compute_flops(runner, model)

    # ------------------------------------------------------------------
    # FLOPs estimation: backbone + neck only
    # ------------------------------------------------------------------
    def _compute_flops(self, runner, model):
        """
        Estimate GFLOPs for backbone + neck only.

        This is more stable than full-detector FLOPs for MMDetection models.
        Full Mask R-CNN FLOPs are proposal-dependent because RPN/RoI heads
        process a variable number of proposals and require data_samples.
        """
        try:
            from mmengine.analysis import get_model_complexity_info

            device = next(model.parameters()).device
            was_training = model.training

            model.eval()

            wrapped_model = BackboneNeckWrapper(model).to(device)
            wrapped_model.eval()

            complexity = get_model_complexity_info(
                wrapped_model,
                self.input_shape,
                show_table=False,
                show_arch=False,
            )

            if was_training:
                model.train()

            flops = complexity.get('flops', None)
            flops_str = complexity.get('flops_str', None)
            params_str = complexity.get('params_str', None)

            activations = complexity.get('activations', None)
            activations_str = complexity.get('activations_str', None)

            gflops = round(flops / 1e9, 3) if flops is not None else None

            self._record.update({
                'flops_scope': 'backbone_plus_neck',
                'flops': flops,
                'gflops': gflops,
                'flops_str': flops_str,
                'complexity_params_str': params_str,
                'activations': activations,
                'activations_str': activations_str,
                'flops_input_shape': self.input_shape,
                'flops_note': (
                    'GFLOPs estimated for backbone + neck only using '
                    'MMEngine complexity analysis. Full Mask R-CNN detector '
                    'FLOPs were not used because RPN/RoI heads are '
                    'proposal-dependent and require detector-specific '
                    'data_samples. Mamba/SSM custom operators may be '
                    'partially unsupported by the profiler.'
                ),
            })

            runner.logger.info(
                '[BENCHMARK] '
                f'FLOPs scope=backbone+neck  '
                f'GFLOPs={gflops}  '
                f'flops_str={flops_str}  '
                f'params_str={params_str}  '
                f'activations={activations_str}  '
                f'input_shape={self.input_shape}'
            )

        except Exception as exc:
            if model.training is False:
                model.train()

            self._record.update({
                'flops_scope': 'backbone_plus_neck',
                'flops': None,
                'gflops': None,
                'flops_error': repr(exc),
                'flops_input_shape': self.input_shape,
                'flops_note': (
                    'Backbone+neck FLOPs estimation failed. This may happen '
                    'with unsupported custom operators, especially Mamba/SSM '
                    'modules or backbones with non-standard forward behavior.'
                ),
            })

            runner.logger.warning(
                f'[BENCHMARK] FLOPs estimation failed: {repr(exc)}'
            )

    # ------------------------------------------------------------------
    # after_train: timing, peak VRAM, JSON export
    # ------------------------------------------------------------------
    @master_only
    def after_train(self, runner):
        elapsed = time.time() - self._start_time

        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        total_iters = runner.iter
        sec_per_iter = elapsed / max(total_iters, 1)

        peak_mem_mb = (
            round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            if torch.cuda.is_available()
            else None
        )

        self._record.update({
            'training_time_seconds': round(elapsed, 1),
            'training_time_hours': round(elapsed / 3600, 4),
            'training_time_hms': f'{int(h)}h {int(m)}m {int(s)}s',
            'total_iters': total_iters,
            'sec_per_iter': round(sec_per_iter, 4),
            'peak_cuda_memory_mb': peak_mem_mb,
        })

        runner.logger.info(
            '[BENCHMARK] '
            f'training_time={int(h)}h {int(m)}m {int(s)}s '
            f'({elapsed:.1f} s)  '
            f'iters={total_iters}  '
            f'sec/iter={sec_per_iter:.4f}  '
            f'peak_VRAM={peak_mem_mb} MB'
        )

        if self.save_json:
            out_path = Path(runner.work_dir) / 'benchmark_record.json'

            with open(out_path, 'w') as f:
                json.dump(self._record, f, indent=2)

            runner.logger.info(f'[BENCHMARK] record → {out_path}')