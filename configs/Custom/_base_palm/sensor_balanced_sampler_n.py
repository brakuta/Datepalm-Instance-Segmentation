# Copyright (c) Stage C Date Palm Benchmark.
# sensor_balanced_sampler_n.py  (v2 - production-grade)
#
# Source-local weighted sampler over an N-source ConcatDataset.
#
# DESIGN
# ------
# Each batch contains tiles from EXACTLY ONE source. This is the critical
# departure from v1, which globally shuffled all indices and destroyed
# the kernel page cache locality that makes multi-mount training fast.
#
# Within an epoch, the per-source sample budgets are:
#     N_s_epoch = round( w_s * N_s / sum_t(w_t * N_t) * sum_t(N_t) )
#
# The sampler emits indices in CHUNKS of `chunk_size` (default = batch_size)
# from one source at a time, then jumps to another source. Source selection
# between chunks is randomised in proportion to remaining budget, so the
# overall epoch composition still matches the target weights but consecutive
# samples (within a batch and across a small window) come from the same mount.
#
# IMPACT ON KERNEL READAHEAD
# --------------------------
# With v1 (global shuffle): consecutive reads hit 3 different mounts → 0%
# readahead benefit, hot page cache wasted.
# With v2 (chunked): consecutive reads hit the same mount until the chunk
# is exhausted → readahead and page cache work as intended.
#
# Distributed-safe: ranks receive disjoint chunks. Multi-rank determinism
# preserved across epochs by `set_epoch`. Worker-init seeding is handled
# in dataset_UAV_GE_Aerial_pooled_C.py.

import math
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dataset import ConcatDataset
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SensorBalancedSamplerN(Sampler):
    """Per-source-chunked weighted sampler over an N-source ConcatDataset.

    Args:
        dataset: The wrapped ConcatDataset.
        weights: Per-source weight vector. Length must equal len(dataset.datasets).
            Effective per-epoch quota for source s:
                quota_s = round( w_s * N_s / sum_t(w_t * N_t) * sum_t(N_t) )
        chunk_size: Indices are emitted in contiguous same-source chunks of this
            size. Set equal to `batch_size` (or batch_size * world_size for DDP)
            so that each global batch comes from one mount. Default 2.
        seed: Optional global seed. Synchronised across ranks if None.
        round_up: If True, pad indices so all ranks receive equal counts.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        weights: Sequence[float],
        chunk_size: int = 2,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        if not isinstance(dataset, ConcatDataset):
            raise TypeError(
                f'SensorBalancedSamplerN expects ConcatDataset, '
                f'got {type(dataset).__name__}'
            )
        if len(weights) != len(dataset.datasets):
            raise ValueError(
                f'weights length ({len(weights)}) must equal number of '
                f'sources ({len(dataset.datasets)})'
            )
        if any(w < 0 for w in weights):
            raise ValueError(f'weights must be non-negative, got {weights}')
        if sum(weights) <= 0:
            raise ValueError('sum of weights must be positive')
        if chunk_size < 1:
            raise ValueError(f'chunk_size must be >= 1, got {chunk_size}')

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.weights = np.asarray(weights, dtype=np.float64)
        self.chunk_size = int(chunk_size)
        self.round_up = round_up
        self.epoch = 0
        self.seed = sync_random_seed() if seed is None else seed

        # Per-source sizes and global-index offsets.
        self.source_sizes: List[int] = [len(d) for d in dataset.datasets]
        self.cum_sizes: List[int] = list(dataset.cumulative_sizes)
        self.offsets: List[int] = [0] + self.cum_sizes[:-1]
        self.num_sources = len(self.source_sizes)

        # Per-epoch per-source quotas.
        total = sum(self.source_sizes)
        eff = self.weights * np.asarray(self.source_sizes, dtype=np.float64)
        proportions = eff / eff.sum()
        raw = proportions * total
        quotas = np.floor(raw).astype(np.int64)
        residual = total - int(quotas.sum())
        if residual > 0:
            top = np.argsort(-(raw - quotas))[:residual]
            quotas[top] += 1
        self.quotas: List[int] = quotas.tolist()
        self.total_size = int(sum(self.quotas))

        # Round per-source quotas DOWN to a multiple of chunk_size so we never
        # split a final partial chunk across two sources (which would
        # re-introduce cross-source batches).
        #
        # Algorithm: first compute the remainder each source must shed, then
        # donate the total shed remainder to the source with the largest
        # effective weight, rounded down to a multiple of chunk_size. Any
        # un-distributable residual (< chunk_size) is simply dropped from the
        # epoch — this is at most chunk_size-1 tiles per epoch, negligible
        # against ~37k tiles.
        remainders = [q % self.chunk_size for q in self.quotas]
        total_shed = sum(remainders)
        self.quotas = [q - r for q, r in zip(self.quotas, remainders)]
        if total_shed >= self.chunk_size:
            # Donate as many full chunks as possible to the strongest source.
            donor = int(np.argmax(
                self.weights * np.asarray(self.source_sizes, dtype=np.float64)
            ))
            donation = (total_shed // self.chunk_size) * self.chunk_size
            self.quotas[donor] += donation
        # Final consistency: every quota is a multiple of chunk_size.
        assert all(q % self.chunk_size == 0 for q in self.quotas), (
            f'internal: chunk alignment failed, quotas={self.quotas}, '
            f'chunk_size={self.chunk_size}'
        )
        self.total_size = int(sum(self.quotas))

        # Per-rank sizes (rounded up to a multiple of chunk_size * world_size
        # so the rank split itself is chunk-aligned).
        align = self.chunk_size * self.world_size
        if self.round_up:
            self.padded_size = math.ceil(self.total_size / align) * align
            self.num_samples = self.padded_size // self.world_size
        else:
            self.padded_size = self.total_size
            self.num_samples = math.ceil(
                (self.total_size - self.rank) / self.world_size
            )

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 1. Build a permuted local index pool per source.
        per_source_pool: List[List[int]] = []
        for src_idx, (size, quota, offset) in enumerate(
            zip(self.source_sizes, self.quotas, self.offsets)
        ):
            if quota == 0 or size == 0:
                per_source_pool.append([])
                continue
            full, rem = divmod(quota, size)
            pool: List[int] = []
            for _ in range(full):
                perm = torch.randperm(size, generator=g).tolist()
                pool.extend(offset + i for i in perm)
            if rem > 0:
                perm = torch.randperm(size, generator=g).tolist()[:rem]
                pool.extend(offset + i for i in perm)
            per_source_pool.append(pool)

        # 2. Build a chunk schedule: which source contributes each chunk.
        #    Number of chunks per source = quota_s / chunk_size (exact, by
        #    construction above).
        chunks_per_source = [q // self.chunk_size for q in self.quotas]
        total_chunks = sum(chunks_per_source)
        # Random ordering of source-tagged chunk slots, e.g. for budgets
        # [3, 4, 1] → tags = [0,0,0,1,1,1,1,2] then shuffled.
        chunk_tags = np.empty(total_chunks, dtype=np.int64)
        cursor = 0
        for s, n in enumerate(chunks_per_source):
            chunk_tags[cursor : cursor + n] = s
            cursor += n
        # Shuffle the chunk tag order.
        perm = torch.randperm(total_chunks, generator=g).numpy()
        chunk_tags = chunk_tags[perm]

        # 3. Emit indices chunk by chunk, drawing from each source's pool
        #    in order.
        per_source_cursor = [0] * self.num_sources
        all_indices: List[int] = []
        cs = self.chunk_size
        for tag in chunk_tags.tolist():
            start = per_source_cursor[tag]
            chunk = per_source_pool[tag][start : start + cs]
            all_indices.extend(chunk)
            per_source_cursor[tag] = start + cs

        # 4. Pad (whole chunks only) to `padded_size`.
        if self.round_up and len(all_indices) < self.padded_size:
            pad_needed = self.padded_size - len(all_indices)
            assert pad_needed % cs == 0, (
                'internal: padding misaligned with chunk_size'
            )
            # Repeat from the start in whole-chunk units.
            extra_chunks: List[int] = []
            i = 0
            while len(extra_chunks) < pad_needed:
                extra_chunks.extend(all_indices[i : i + cs])
                i = (i + cs) % len(all_indices)
            all_indices.extend(extra_chunks[:pad_needed])

        # 5. Distribute chunks across ranks. We split by chunk, not by
        #    individual index, so each rank still sees same-source batches.
        cs_world = cs * self.world_size
        rank_indices: List[int] = []
        for chunk_start in range(0, self.padded_size, cs_world):
            this_world_chunk = all_indices[chunk_start : chunk_start + cs_world]
            rank_slice = this_world_chunk[self.rank * cs : (self.rank + 1) * cs]
            rank_indices.extend(rank_slice)

        assert len(rank_indices) == self.num_samples, (
            f'rank {self.rank}: got {len(rank_indices)} '
            f'(expected {self.num_samples})'
        )
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
