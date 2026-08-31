# Copyright (c) Stage C Date Palm Benchmark.
# mean_sensor_metric_hook.py  (v2 - production-grade)
#
# Computes the unweighted mean of named sensor-prefixed metric keys and
# writes it into the validation metrics dict under `out_key`. Required by
# EarlyStoppingHook to monitor a single combined value across two val sources.
#
# Differences from v1:
#   - NaN-safe: a non-finite per-sensor value is logged and the mean is
#     still computed from finite values. If all values are non-finite, the
#     out_key is set to the worst-case value so EarlyStopping is not tricked
#     into "improving" against NaN.
#   - Also writes the mean to runner.message_hub so it appears in logs
#     and downstream logging hooks pick it up.
#   - Priority bumped to HIGHEST so it executes before EarlyStoppingHook
#     and PerSensorBestCheckpointHook unambiguously.

import math
from typing import List, Optional

from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class MeanSensorMetricHook(Hook):
    """Inject the unweighted mean of named sensor metrics under `out_key`.

    Args:
        sensor_keys: List of metric keys to average. Example:
            ['UAV/coco/segm_mAP_50', 'GE/coco/segm_mAP_50']
        out_key: The output key for the mean. Example: 'mean/segm_mAP_50'
        rule: 'greater' (worst-case = -inf) or 'less' (worst-case = +inf).
            Determines the sentinel value used when ALL sensor values are
            non-finite. Should match the rule of the downstream
            EarlyStoppingHook. Default 'greater'.
    """

    priority = 'HIGHEST'   # must run before EarlyStopping and best-ckpt

    def __init__(
        self,
        sensor_keys: List[str],
        out_key: str,
        rule: str = 'greater',
    ) -> None:
        if len(sensor_keys) < 2:
            raise ValueError(
                f'sensor_keys must contain at least 2 entries, '
                f'got {sensor_keys}'
            )
        if rule not in ('greater', 'less'):
            raise ValueError(f"rule must be 'greater' or 'less', got {rule}")
        self.sensor_keys = list(sensor_keys)
        self.out_key = out_key
        self.rule = rule
        self._worst = float('-inf') if rule == 'greater' else float('inf')

    def after_val_epoch(
        self,
        runner: Runner,
        metrics: Optional[dict] = None,
    ) -> None:
        if metrics is None:
            return
        logger = MMLogger.get_current_instance()

        # Collect finite values, log non-finite and missing.
        finite_pairs = []
        missing = []
        nonfinite = []
        for k in self.sensor_keys:
            if k not in metrics:
                missing.append(k)
                continue
            try:
                v = float(metrics[k])
            except (TypeError, ValueError):
                nonfinite.append(k)
                continue
            if math.isfinite(v):
                finite_pairs.append((k, v))
            else:
                nonfinite.append(k)

        if missing:
            logger.warning(
                f'[MeanSensorMetricHook] missing keys {missing}; '
                f'available: {sorted(metrics.keys())}'
            )
        if nonfinite:
            logger.warning(
                f'[MeanSensorMetricHook] non-finite values for {nonfinite}'
            )

        if not finite_pairs:
            logger.warning(
                f'[MeanSensorMetricHook] no finite values; setting '
                f'{self.out_key} = {self._worst}'
            )
            metrics[self.out_key] = self._worst
            return

        mean_value = sum(v for _, v in finite_pairs) / len(finite_pairs)
        metrics[self.out_key] = mean_value

        # Also publish to message hub so it appears in WandB/TensorBoard logs.
        try:
            runner.message_hub.update_scalar(
                f'val/{self.out_key}', mean_value, count=1,
            )
        except Exception:
            pass

        breakdown = ' '.join(f'{k}={v:.4f}' for k, v in finite_pairs)
        logger.info(
            f'[MeanSensorMetricHook] {self.out_key}={mean_value:.4f} '
            f'(from {len(finite_pairs)}/{len(self.sensor_keys)}: {breakdown})'
        )
