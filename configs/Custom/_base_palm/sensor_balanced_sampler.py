# ==========================================================================
# _base_palm/sensor_balanced_sampler.py
# --------------------------------------------------------------------------
# Single-axis sensor-rebalancing weighted sampler for the pooled MS-15 cm
# corpus (GE 15 cm + Aerial 15 cm).
#
# Design rationale:
#   The pooled training corpus combines two sources with a ~16:1 size
#   imbalance (GE: 19,616 tiles; Aerial: 1,204 tiles). Naive uniform
#   sampling exposes the model almost exclusively to GE radiometry,
#   producing a model that fails to generalise to Aerial acquisition
#   geometry. Conversely, full 50/50 rebalancing oversamples the
#   single-location Aerial source so aggressively that overfitting on
#   location-specific features (palm rows, soil colour, shadow direction)
#   becomes likely.
#
#   This sampler implements a continuous interpolation between the two
#   regimes via a single hyperparameter `alpha`:
#
#       w_i = alpha * (1 / N_S(i)) + (1 - alpha) * (1 / N_total)
#
#   where N_S(i) is the count of samples sharing the sensor tag of
#   sample i, and N_total is the total dataset size.
#
#   alpha = 0.0  →  uniform sampling (preserves native ~94/6 split)
#   alpha = 0.5  →  mild rebalancing (~72% GE / ~28% Aerial)  ← used
#   alpha = 1.0  →  full sensor balance (~50/50)
#
# Sensor identification:
#   The sensor identity of each sample is inferred from the dataset
#   construction order in ConcatDataset, NOT from per-image annotation
#   fields. This avoids any modification of the COCO JSONs. The
#   `sensor_names` argument must list one tag per sub-dataset, matched
#   to the order of `datasets=[...]` in dataset_MS15_pooled.py.
#
# Augmentation interaction:
#   Each Aerial tile is drawn approximately 92 times per 100,000
#   iterations (effective batch 4) at alpha=0.5. The augmentation
#   pipeline (RandomFlip x2, PhotoMetricDistortion) acts AFTER sampling
#   and is the primary safeguard against memorisation of repeated draws.
# ==========================================================================

from collections import Counter
from typing import Iterator, Optional, Tuple

import torch
from torch.utils.data import WeightedRandomSampler

from mmdet.registry import DATA_SAMPLERS
from mmengine.dist import sync_random_seed


@DATA_SAMPLERS.register_module()
class SensorBalancedSampler(WeightedRandomSampler):
    """Sensor-rebalancing sampler keyed on ConcatDataset boundaries.

    Args:
        dataset: A ConcatDataset whose `datasets` list orders the
            sub-datasets in correspondence with `sensor_names`.
        alpha (float): Rebalancing strength in [0, 1].
            0.0 = uniform sampling (preserves native ~94/6 split).
            0.5 = mild rebalancing (~72% GE / ~28% Aerial; recommended).
            1.0 = full sensor balance (~50/50).
        sensor_names (tuple[str, ...]): Sensor tags ordered to match
            the construction order of `dataset.datasets`.
            For Stage B: ('GE', 'Aerial').
        num_samples (int, optional): Samples drawn per epoch-equivalent.
            Defaults to len(dataset).
        seed (int): RNG seed; synchronised across distributed workers.

    Notes:
        - Replacement is fixed to True. WeightedRandomSampler requires
          replacement for non-uniform weights to be respected, and this
          is also the only mode compatible with iteration-based training.
        - Distribution diagnostics are exposed via the `diagnostics`
          property and are also logged at construction time.
    """

    def __init__(
        self,
        dataset,
        alpha: float = 0.5,
        sensor_names: Tuple[str, ...] = ('GE', 'Aerial'),
        num_samples: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f'alpha must lie in [0, 1], got {alpha}'
            )

        if not hasattr(dataset, 'cumulative_sizes'):
            raise TypeError(
                f'SensorBalancedSampler requires a ConcatDataset, '
                f'got {type(dataset).__name__}. The pooled MS-15 cm '
                f'dataloader must wrap GE and Aerial via ConcatDataset.'
            )

        if len(dataset.datasets) != len(sensor_names):
            raise ValueError(
                f'sensor_names has {len(sensor_names)} entries '
                f'({sensor_names}) but ConcatDataset has '
                f'{len(dataset.datasets)} sub-datasets. Counts must match.'
            )

        # --- Build per-index sensor tag list from cumulative_sizes ----
        # boundaries: [0, N_GE, N_GE + N_Aerial]
        boundaries = [0] + list(dataset.cumulative_sizes)
        sensors = []
        for sensor_idx, name in enumerate(sensor_names):
            block_size = boundaries[sensor_idx + 1] - boundaries[sensor_idx]
            sensors.extend([name] * block_size)

        n_total = len(sensors)
        sensor_counts = Counter(sensors)

        # --- Compute per-sample weights -------------------------------
        weights = torch.tensor(
            [
                alpha / sensor_counts[s] + (1.0 - alpha) / n_total
                for s in sensors
            ],
            dtype=torch.double,
        )

        # --- Bookkeeping ----------------------------------------------
        self.seed = sync_random_seed() if seed is None else seed
        self.alpha = alpha
        self.sensor_names = tuple(sensor_names)
        self._sensor_counts = dict(sensor_counts)

        super().__init__(
            weights=weights,
            num_samples=num_samples or n_total,
            replacement=True,
        )

        # --- Construction-time diagnostic log -------------------------
        # Expected per-batch sensor probability under this alpha.
        expected_p = {
            s: (alpha / sensor_counts[s] + (1.0 - alpha) / n_total)
               * sensor_counts[s]
            for s in sensor_counts
        }
        norm = sum(expected_p.values())
        expected_p_norm = {s: p / norm for s, p in expected_p.items()}

        msg = (
            f'[SensorBalancedSampler] alpha={alpha} '
            f'pool={n_total} '
            f'counts={dict(sensor_counts)} '
            f'expected_per_batch_pct='
            f'{ {s: round(100 * p, 2) for s, p in expected_p_norm.items()} }'
        )
        # Plain print so the message lands in stdout regardless of logger
        # initialisation order at config-parse time.
        print(msg)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed)
        return iter(
            torch.multinomial(
                self.weights,
                self.num_samples,
                self.replacement,
                generator=g,
            ).tolist()
        )

    @property
    def diagnostics(self) -> dict:
        """Diagnostic snapshot for verification logging."""
        return {
            'alpha': self.alpha,
            'sensor_names': self.sensor_names,
            'sensor_counts': dict(self._sensor_counts),
            'total_samples': sum(self._sensor_counts.values()),
        }
