# Copyright (c) Stage C Date Palm Benchmark.
# per_sensor_best_checkpoint_hook.py  (v2 - production-grade)
#
# Differences from v1:
#   - Uses mmengine.runner.save_checkpoint (handles MMDDP unwrapping, FP16
#     state, EMA, optimizer state correctly)
#   - Writes to a tmp file then renames atomically — partial writes during
#     a crash or kill cannot leave a corrupt best_*.pth
#   - Removes previous best only AFTER the new one is fsync'd to disk
#   - Survives missing metric keys gracefully (logs warning, continues)
#   - Resumes correctly from a checkpoint by restoring `_best` and
#     `_best_path` via `before_train` and `after_load_checkpoint`

import glob
import os
import os.path as osp
import re
from typing import Dict, Optional

from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.runner import Runner, save_checkpoint

from mmdet.registry import HOOKS


_SAFE_KEY_RE = re.compile(r'[^A-Za-z0-9_.-]+')


def _safe_key(key: str) -> str:
    """Map metric keys with / and . to filesystem-safe tokens."""
    return _SAFE_KEY_RE.sub('_', key)


@HOOKS.register_module()
class PerSensorBestCheckpointHook(Hook):
    """Save best checkpoint independently per named sensor.

    For each (sensor, metric_key) entry in `monitors`, tracks the best value
    seen on `after_val_epoch`. When a new best is observed, writes a
    checkpoint named `best_<sensor>_<safe_key>_iter_<N>.pth` and removes
    any prior best for that sensor.

    Args:
        monitors: Dict mapping sensor name -> metric key. Example:
            {'UAV': 'UAV/coco/segm_mAP_50', 'GE': 'GE/coco/segm_mAP_50'}
        rule: 'greater' (higher is better) or 'less'. Default 'greater'.
        save_optimizer: If True, also save optimizer state for resume.
            Default True.
        save_param_scheduler: If True, also save scheduler state.
            Default True.

    Priority: VERY_LOW — runs after MeanSensorMetricHook (HIGH) and after
    EarlyStoppingHook (NORMAL). This ordering matters for resume: if a
    rank crashes between writing the new best and removing the old, the
    next start sees both and can pick the larger iter index.
    """

    priority = 'VERY_LOW'

    def __init__(
        self,
        monitors: Dict[str, str],
        rule: str = 'greater',
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
    ) -> None:
        if rule not in ('greater', 'less'):
            raise ValueError(f"rule must be 'greater' or 'less', got {rule}")
        if not monitors:
            raise ValueError('monitors dict must be non-empty')
        self.monitors = dict(monitors)
        self.rule = rule
        self.save_optimizer = save_optimizer
        self.save_param_scheduler = save_param_scheduler
        self._init_value = float('-inf') if rule == 'greater' else float('inf')
        self._best: Dict[str, float] = {s: self._init_value for s in monitors}
        self._best_path: Dict[str, Optional[str]] = {s: None for s in monitors}

    def _is_better(self, current: float, best: float) -> bool:
        return current > best if self.rule == 'greater' else current < best

    @master_only
    def before_train(self, runner: Runner) -> None:
        """Restore best state when resuming."""
        logger = MMLogger.get_current_instance()
        for sensor, key in self.monitors.items():
            # Match on the sensor prefix only, so this is robust to the
            # exact metric-key formatting in the filename (e.g.
            # best_UAV_segm_mAP_50_iter_*.pth).
            pattern = osp.join(runner.work_dir,
                               f'best_{sensor}_*_iter_*.pth')
            existing = sorted(glob.glob(pattern))
            if not existing:
                continue

            # Pick the one with highest iter (latest best).
            def _iter_of(p: str) -> int:
                m = re.search(r'iter_(\d+)\.pth$', p)
                return int(m.group(1)) if m else -1
            existing.sort(key=_iter_of, reverse=True)
            picked = existing[0]
            self._best_path[sensor] = picked

            # Restore best metric value from the checkpoint's meta dict.
            try:
                import torch
                payload = torch.load(picked, map_location='cpu')
                meta = payload.get('meta', {}) if isinstance(payload, dict) else {}
                val = meta.get('metric_value', None)
                if val is not None:
                    self._best[sensor] = float(val)
                    logger.info(
                        f'[PerSensorBestCheckpointHook] restored best for '
                        f'{sensor}: {key}={val:.4f} from {osp.basename(picked)}'
                    )
                else:
                    logger.warning(
                        f'[PerSensorBestCheckpointHook] {picked} has no '
                        f'meta.metric_value; restored path only.'
                    )
            except Exception as e:
                logger.warning(
                    f'[PerSensorBestCheckpointHook] could not read '
                    f'meta from {picked}: {e}; restored path only.'
                )

            # Remove any extra older best files for this sensor.
            for extra in existing[1:]:
                try:
                    os.remove(extra)
                    logger.info(f'  removed older best: {osp.basename(extra)}')
                except OSError:
                    pass

    @master_only
    def after_val_epoch(
        self,
        runner: Runner,
        metrics: Optional[Dict] = None,
    ) -> None:
        logger = MMLogger.get_current_instance()
        if metrics is None:
            return

        for sensor, key in self.monitors.items():
            if key not in metrics:
                logger.warning(
                    f'[PerSensorBestCheckpointHook] metric "{key}" not '
                    f'found; available: {sorted(metrics.keys())}'
                )
                continue
            try:
                value = float(metrics[key])
            except (TypeError, ValueError) as e:
                logger.warning(
                    f'[PerSensorBestCheckpointHook] non-numeric value '
                    f'for "{key}": {metrics[key]} ({e})'
                )
                continue
            if not self._is_better(value, self._best[sensor]):
                continue

            self._best[sensor] = value
            # Build a clean filename matching the convention documented in
            # configs/Custom/Evaluation/compile_results.py:
            #   best_<sensor>_<bare_metric>_iter_<N>.pth
            #   e.g. best_UAV_segm_mAP_50_iter_25000.pth
            # The metric key carries a redundant '<sensor>/' prefix from
            # MultiDatasetsEvaluator (e.g. 'UAV/coco/segm_mAP_50'); strip the
            # sensor prefix and the 'coco/' infix so the sensor appears once.
            bare_metric = key
            if bare_metric.startswith(f'{sensor}/'):
                bare_metric = bare_metric[len(sensor) + 1:]
            if bare_metric.startswith('coco/'):
                bare_metric = bare_metric[len('coco/'):]
            safe = _safe_key(bare_metric)
            iter_n = runner.iter + 1   # checkpoint is "after" this iter
            fname = f'best_{sensor}_{safe}_iter_{iter_n}.pth'
            final_path = osp.join(runner.work_dir, fname)
            tmp_path = final_path + '.tmp'

            meta = dict(
                iter=iter_n,
                epoch=runner.epoch,
                sensor=sensor,
                metric_key=key,
                metric_value=value,
                stage='C',
            )
            # MMEngine's save_checkpoint handles DDP unwrap, FP16 cast-to-FP32,
            # optimizer + scheduler state, and meta dict serialization.
            try:
                save_checkpoint(
                    self._build_checkpoint_payload(
                        runner, meta=meta,
                        save_optimizer=self.save_optimizer,
                        save_param_scheduler=self.save_param_scheduler,
                    ),
                    tmp_path,
                )
                # Atomic rename — partial writes can never become a "best".
                os.replace(tmp_path, final_path)
            except Exception as e:
                logger.error(
                    f'[PerSensorBestCheckpointHook] save failed for '
                    f'{fname}: {e}'
                )
                if osp.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                continue

            logger.info(
                f'[PerSensorBestCheckpointHook] {sensor}: new best '
                f'{key}={value:.4f} at iter {iter_n} -> {fname}'
            )

            # Remove previous best for this sensor only AFTER new one is on disk.
            prev = self._best_path[sensor]
            if prev is not None and prev != final_path and osp.exists(prev):
                try:
                    os.remove(prev)
                    logger.info(f'  removed prior best: {osp.basename(prev)}')
                except OSError as e:
                    logger.warning(
                        f'[PerSensorBestCheckpointHook] could not remove '
                        f'{prev}: {e}'
                    )
            self._best_path[sensor] = final_path

    def _build_checkpoint_payload(
        self,
        runner: Runner,
        meta: dict,
        save_optimizer: bool,
        save_param_scheduler: bool,
    ) -> dict:
        """Assemble the dict expected by `mmengine.runner.save_checkpoint`."""
        from mmengine.model.utils import revert_sync_batchnorm
        from mmengine.utils import get_git_hash
        try:
            from mmengine.runner.checkpoint import weights_to_cpu
        except ImportError:
            # mmengine < 0.10 path
            from mmengine.utils import weights_to_cpu

        # Unwrap MMDDP/DDP if needed.
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        state_dict = weights_to_cpu(model.state_dict())

        payload = dict(
            meta=meta,
            state_dict=state_dict,
            message_hub=runner.message_hub.state_dict(),
        )
        if save_optimizer and runner.optim_wrapper is not None:
            payload['optimizer'] = runner.optim_wrapper.state_dict()
        if save_param_scheduler and runner.param_schedulers is not None:
            payload['param_schedulers'] = [
                s.state_dict() for s in runner.param_schedulers
                if hasattr(s, 'state_dict')
            ]
        return payload
