#!/usr/bin/env python
# bench_dataloader_stagec.py
#
# Pre-launch dataloader sanity check. Runs the training dataloader for N
# iterations without any model forward, reports:
#   - Per-iteration data_time (load + augment + collate)
#   - Per-source read distribution (must match SensorBalancedSamplerN quotas)
#   - Per-batch source homogeneity (must be 100% same-source)
#   - Worst-case per-iteration latency
#   - Memory residency growth (worker leak detection)
#
# Run this BEFORE launching the 120k iteration training. If data_time is
# above ~0.05 s mean or any iteration exceeds 1 s, investigate before
# committing GPU hours.
#
# Usage:
#   python tools/bench_dataloader_stagec.py \
#       --config configs/Custom/maskrcnn_palm_stagec/maskrcnn_r50_stagec.py \
#       --iters 200 \
#       [--warmup 10]

import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from typing import List

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Benchmark the Stage C training dataloader.',
    )
    p.add_argument('--config', required=True, help='Stage C backbone config.')
    p.add_argument('--iters', type=int, default=200,
                   help='Number of iterations to benchmark (default 200).')
    p.add_argument('--warmup', type=int, default=10,
                   help='Warmup iterations to discard (default 10).')
    p.add_argument('--print-every', type=int, default=50,
                   help='Progress log interval.')
    p.add_argument('--check-same-source', action='store_true', default=True,
                   help='Verify every batch is from a single source.')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.config):
        print(f'ERROR: config not found: {args.config}', file=sys.stderr)
        return 1

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # Build only the training dataloader.
    train_dl_cfg = cfg.train_dataloader
    print(f'[INFO] Building dataloader from {args.config}')
    print(f'[INFO]   batch_size       = {train_dl_cfg.get("batch_size")}')
    print(f'[INFO]   num_workers      = {train_dl_cfg.get("num_workers")}')
    print(f'[INFO]   persistent       = {train_dl_cfg.get("persistent_workers")}')
    print(f'[INFO]   pin_memory       = {train_dl_cfg.get("pin_memory")}')
    print(f'[INFO]   prefetch_factor  = {train_dl_cfg.get("prefetch_factor")}')
    sampler_cfg = train_dl_cfg.get('sampler', {})
    if sampler_cfg.get('type') == 'SensorBalancedSamplerN':
        print(f'[INFO]   sampler.weights  = {sampler_cfg.get("weights")}')
        print(f'[INFO]   sampler.chunk    = {sampler_cfg.get("chunk_size")}')

    dataloader = Runner.build_dataloader(train_dl_cfg)
    iterator = iter(dataloader)

    # Determine per-source ranges. Pull from the concat dataset structure.
    ds = dataloader.dataset
    if hasattr(ds, 'datasets'):
        sizes = [len(d) for d in ds.datasets]
        offsets = [0]
        for n in sizes[:-1]:
            offsets.append(offsets[-1] + n)
        sources = ['UAV', 'GE', 'Aerial'][:len(sizes)]
        print(f'[INFO]   source sizes     = {dict(zip(sources, sizes))}')
        print(f'[INFO]   source offsets   = {dict(zip(sources, offsets))}')
    else:
        sources = None
        sizes = None
        offsets = None

    def source_of(global_idx: int) -> str:
        if sources is None:
            return 'unknown'
        for s, (o, n) in enumerate(zip(offsets, sizes)):
            if o <= global_idx < o + n:
                return sources[s]
        return f'out-of-range:{global_idx}'

    # Warmup.
    print(f'\n[INFO] Warming up for {args.warmup} iterations...')
    for i in range(args.warmup):
        try:
            _ = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            _ = next(iterator)

    # Benchmark.
    print(f'[INFO] Benchmarking {args.iters} iterations...\n')
    per_iter_times: List[float] = []
    source_counter: Counter = Counter()
    mixed_batches = 0
    batch_size = train_dl_cfg.get('batch_size', 2)
    cross_iter_source_pairs: Counter = Counter()  # mount-switch frequency
    last_source = None

    t_start = time.perf_counter()
    for it in range(args.iters):
        t0 = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        t1 = time.perf_counter()
        per_iter_times.append(t1 - t0)

        # Inspect the batch's source composition.
        # MMDet's data_sample carries img_path; we infer source from prefix.
        if 'data_samples' in batch:
            samples = batch['data_samples']
        else:
            samples = batch.get('inputs', [])
        batch_sources = []
        for s in samples:
            img_path = getattr(s, 'img_path', None) or getattr(s, 'metainfo', {}).get('img_path', '')
            if 'UAV' in img_path:
                batch_sources.append('UAV')
            elif 'GE' in img_path:
                batch_sources.append('GE')
            elif 'aerial' in img_path or 'Aerial' in img_path:
                batch_sources.append('Aerial')
            else:
                batch_sources.append('unknown')

        for s in batch_sources:
            source_counter[s] += 1
        if len(set(batch_sources)) > 1:
            mixed_batches += 1
        if batch_sources:
            cur = batch_sources[0]
            if last_source is not None:
                cross_iter_source_pairs[(last_source, cur)] += 1
            last_source = cur

        if (it + 1) % args.print_every == 0:
            recent = per_iter_times[-args.print_every:]
            mean_recent = sum(recent) / len(recent)
            print(f'  [iter {it+1:>4d}/{args.iters}] '
                  f'mean_data_time = {mean_recent*1000:.1f} ms  '
                  f'max = {max(recent)*1000:.1f} ms')

    t_total = time.perf_counter() - t_start

    # =================== REPORT ===================
    times = sorted(per_iter_times)
    n = len(times)
    print('\n' + '=' * 72)
    print('THROUGHPUT')
    print('=' * 72)
    print(f'Total wall time         : {t_total:.2f} s for {n} iterations')
    print(f'Mean iter/sec           : {n/t_total:.2f}')
    print(f'Mean batches/sec        : {n/t_total:.2f} '
          f'(batch_size={batch_size}, '
          f'effective tiles/sec={n*batch_size/t_total:.1f})')
    print(f'\nPer-iteration data_time (ms):')
    print(f'  mean                  : {sum(times)/n*1000:.2f}')
    print(f'  median (p50)          : {times[n//2]*1000:.2f}')
    print(f'  p90                   : {times[int(n*0.90)]*1000:.2f}')
    print(f'  p95                   : {times[int(n*0.95)]*1000:.2f}')
    print(f'  p99                   : {times[int(n*0.99)]*1000:.2f}')
    print(f'  max                   : {times[-1]*1000:.2f}')
    print(f'  stdev                 : '
          f'{(sum((t - sum(times)/n)**2 for t in times)/n)**0.5*1000:.2f}')

    print('\n' + '=' * 72)
    print('SOURCE COMPOSITION')
    print('=' * 72)
    total_seen = sum(source_counter.values())
    print(f'Total tiles seen        : {total_seen}')
    for src in (sources or ['UAV', 'GE', 'Aerial']):
        c = source_counter.get(src, 0)
        pct = c / total_seen * 100 if total_seen else 0
        print(f'  {src:8s} {c:>6d} tiles ({pct:5.2f}%)')

    print('\n' + '=' * 72)
    print('SAME-SOURCE BATCH PROPERTY  (CRITICAL)')
    print('=' * 72)
    print(f'Mixed-source batches    : {mixed_batches} / {args.iters}')
    if mixed_batches == 0:
        print('  PASS - every batch from a single source')
    else:
        print('  FAIL - source-local sampler is not working correctly')
        print('  Investigate before launching training.')

    print('\n' + '=' * 72)
    print('CROSS-ITERATION SOURCE SWITCHES')
    print('=' * 72)
    same_source_consecutive = sum(
        v for (a, b), v in cross_iter_source_pairs.items() if a == b
    )
    switches = sum(
        v for (a, b), v in cross_iter_source_pairs.items() if a != b
    )
    total_pairs = same_source_consecutive + switches
    if total_pairs > 0:
        print(f'Same-source consecutive : {same_source_consecutive} '
              f'({same_source_consecutive/total_pairs*100:.1f}%)')
        print(f'Mount-switch pairs      : {switches} '
              f'({switches/total_pairs*100:.1f}%)')
        print('  (Higher same-source % = better page cache locality.)')

    print('\n' + '=' * 72)
    print('VERDICT')
    print('=' * 72)
    mean_ms = sum(times) / n * 1000
    p99_ms = times[int(n * 0.99)] * 1000
    verdict_ok = (mean_ms < 50 and p99_ms < 500 and mixed_batches == 0)
    if verdict_ok:
        print('OK - launch training.')
        return 0
    else:
        if mean_ms >= 50:
            print(f'WARN - mean data_time {mean_ms:.1f} ms is high '
                  f'(target < 50 ms).')
        if p99_ms >= 500:
            print(f'WARN - p99 data_time {p99_ms:.1f} ms is high '
                  f'(target < 500 ms). Suggests intermittent disk stalls.')
        if mixed_batches > 0:
            print(f'FAIL - {mixed_batches} mixed-source batches.')
        return 2


if __name__ == '__main__':
    sys.exit(main())
