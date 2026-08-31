# Copyright (c) Stage C Date Palm Benchmark.
# dataloader_worker_init.py
#
# Per-worker initialisation function. Called once per DataLoader worker at
# fork (or persistent_workers re-init). Does two important things:
#
# 1. Pins cv2 to ONE thread per worker. By default cv2 spawns its own thread
#    pool equal to physical cores; multiplied by num_workers this can cause
#    severe oversubscription and unpredictable per-batch latency. Forcing
#    cv2.setNumThreads(1) inside the worker is the standard mmdet/detectron2
#    fix for this issue. Likewise OMP/MKL thread pools.
#
# 2. Seeds the worker deterministically as (global_seed + worker_id + epoch),
#    so per-worker augmentation RNG is reproducible.
#
# This is registered as `worker_init_fn` in the train/val dataloaders.

import os
import random
from typing import Optional

import numpy as np
import torch


def stagec_worker_init_fn(worker_id: int,
                          num_workers: int = 1,
                          rank: int = 0,
                          seed: Optional[int] = None) -> None:
    """Per-worker init hook for the Stage C dataloaders.

    Args:
        worker_id: PyTorch worker index (0..num_workers-1).
        num_workers: Per-rank worker count.
        rank: Global rank (0 in single-GPU).
        seed: Global RNG seed. If None, derived from torch initial seed.
    """
    # 1. cv2 thread pool to 1 per worker.
    try:
        import cv2
        cv2.setNumThreads(1)
        # cv2.ocl.setUseOpenCL(False)  # disable OpenCL; usually irrelevant on training boxes
    except ImportError:
        pass

    # 2. OMP/MKL thread pools to 1 per worker. Critical on TITAN RTX
    # workstation where 2 workers x 16 OMP threads = 32 thread spawn per batch.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # 3. Deterministic seed.
    if seed is None:
        # torch.utils.data uses base_seed = torch.initial_seed() % 2**32 per worker.
        base_seed = torch.initial_seed() % (2**32)
    else:
        base_seed = (seed + rank * num_workers + worker_id) % (2**32)
    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)
