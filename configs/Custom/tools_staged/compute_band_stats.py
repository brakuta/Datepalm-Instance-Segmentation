#!/usr/bin/env python3
# =============================================================================
# compute_band_stats.py
# -----------------------------------------------------------------------------
# Per-band mean and standard deviation over a set of tiles, for the
# data_preprocessor of a multispectral run.
#
# WHY NOT JUST REUSE THE IMAGENET NUMBERS
#   The RGB configs normalise with ImageNet statistics ([123.675, 116.28,
#   103.53] / [58.395, 57.12, 57.375]). Those describe natural photographs,
#   and they are the right choice for RGB precisely because the pretrained
#   weights were fitted under them.
#
#   For a fourth or eighth channel there is no ImageNet number to inherit --
#   NIR has no counterpart in a photograph. Guessing (say, reusing the red
#   statistics) leaves that channel centred wrongly, and a badly centred
#   channel is one the network has to spend capacity correcting. Measure it.
#
#   Keep the first three entries at the ImageNet values so the RGB channels
#   stay consistent with the inflated stem, and append MEASURED values for the
#   extra bands. --keep-imagenet-rgb does exactly that and is the default.
#
# ONLY EVER MEASURE ON TRAIN
#   Statistics taken over val or test leak information about the evaluation
#   data into the model's normalisation. The effect is small but it is a leak,
#   and it is free to avoid.
#
# USAGE
#   python configs/Custom/tools_staged/compute_band_stats.py \
#       --images /workspace/datasets/COCO/Sat_30cm_MS/train_ms/JPEGImages \
#       --sample 400
# =============================================================================

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.12, 57.375]
EXTS = ('.tif', '.tiff')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', required=True,
                    help='TRAIN tile directory (searched recursively)')
    ap.add_argument('--sample', type=int, default=400,
                    help='tiles to sample; 0 = all')
    ap.add_argument('--seed', type=int, default=20260804)
    ap.add_argument('--ignore-zero', action='store_true',
                    help='exclude pixels that are 0 in every band. Masked '
                         'nodata is written as 0 by the tiler, and including '
                         'it drags every mean toward zero in proportion to how '
                         'much of the mosaic edge a split happens to contain.')
    ap.add_argument('--keep-imagenet-rgb', dest='keep_rgb',
                    action='store_true', default=True,
                    help='report ImageNet values for the first three channels '
                         '(default), measured values for the rest')
    ap.add_argument('--all-measured', dest='keep_rgb', action='store_false',
                    help='report measured values for every channel')
    args = ap.parse_args()

    root = Path(args.images)
    files = sorted(f for f in root.rglob('*') if f.suffix.lower() in EXTS)
    if not files:
        sys.exit(f'No .tif tiles under {root}')
    if args.sample and args.sample < len(files):
        files = random.Random(args.seed).sample(files, args.sample)
    print(f'{len(files)} tile(s)')

    import rasterio
    n_px = 0
    s1 = s2 = None
    for i, f in enumerate(files, 1):
        with rasterio.open(f) as src:
            a = src.read().astype(np.float64)          # (C, H, W)
        c = a.shape[0]
        flat = a.reshape(c, -1)
        if args.ignore_zero:
            keep = (flat != 0).any(axis=0)
            flat = flat[:, keep]
        if flat.shape[1] == 0:
            continue
        if s1 is None:
            s1, s2 = np.zeros(c), np.zeros(c)
        s1 += flat.sum(axis=1)
        s2 += (flat ** 2).sum(axis=1)
        n_px += flat.shape[1]
        if i % 100 == 0:
            print(f'   {i}/{len(files)}')

    mean = s1 / n_px
    std = np.sqrt(np.maximum(s2 / n_px - mean ** 2, 1e-12))

    print(f'\nmeasured over {n_px:,} pixels')
    for i, (m, s) in enumerate(zip(mean, std)):
        print(f'  band {i + 1}: mean {m:8.3f}  std {s:8.3f}')

    n = len(mean)
    if args.keep_rgb and n >= 3:
        out_m = IMAGENET_MEAN + [float(x) for x in mean[3:]]
        out_s = IMAGENET_STD + [float(x) for x in std[3:]]
        note = ('first three = ImageNet, matching the inflated stem; '
                'remainder measured')
    else:
        out_m = [float(x) for x in mean]
        out_s = [float(x) for x in std]
        note = 'all channels measured'

    print(f'\n# {note}')
    print('data_preprocessor = dict(')
    print(f'    mean={[round(x, 3) for x in out_m]},')
    print(f'    std={[round(x, 3) for x in out_s]},')
    print('    bgr_to_rgb=False,   # MUST be False for N-band input')
    print(')')


if __name__ == '__main__':
    main()
