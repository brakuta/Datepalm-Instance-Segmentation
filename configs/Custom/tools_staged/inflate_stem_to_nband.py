#!/usr/bin/env python3
# =============================================================================
# inflate_stem_to_nband.py
# -----------------------------------------------------------------------------
# Widen a 3-channel ImageNet stem to N channels, so a multispectral run can
# still start from pretrained weights.
#
# THE PROBLEM
#   Every backbone in this project starts from ImageNet, whose first
#   convolution takes 3 input channels. Feed it 4 (R,G,B,NIR) or 8 and the
#   shapes do not match: MMEngine reports a size mismatch and SKIPS that
#   tensor, leaving the stem randomly initialised. Training then proceeds and
#   the log looks normal -- but the single most important layer for early
#   features has lost its pretraining, and the multispectral arm is quietly
#   handicapped relative to the RGB arm it is being compared against. That is
#   a comparison you cannot trust, and nothing in the metrics reveals it.
#
# WHAT THIS DOES
#   Finds the stem convolution (the first weight of shape [out, 3, kh, kw]),
#   builds an [out, N, kh, kw] tensor, copies the RGB weights into the first
#   three input channels and fills the rest.
#
# HOW THE EXTRA CHANNELS ARE FILLED  (--mode)
#   mean   the mean of the RGB filters (default). A near-infrared channel is
#          spectrally closest to red but not equal to it, and the RGB mean is
#          the least committal starting point that still produces sensible
#          edge responses from iteration one. Total activation magnitude grows
#          by roughly N/3, which the following norm layer absorbs.
#   scaled the same, then the WHOLE tensor is multiplied by 3/N so the summed
#          response matches the 3-channel original (Carreira & Zisserman
#          inflation). Preserves activation scale but shrinks the RGB response,
#          which matters more when the backbone LR is low -- as it is in arm C.
#   zero   extra channels start at zero: the network initially ignores the new
#          bands and must learn to use them. The most conservative option, and
#          the one that most nearly guarantees "MS is at least as good as RGB",
#          but it wastes the early epochs.
#
#   There is no consensus best choice. `mean` is the common default in remote
#   sensing; `zero` is the safest if the multispectral arm must not lose to
#   RGB for optimisation reasons. Whichever you pick, use the SAME mode for
#   every backbone, and state it in Methods -- it is a real degree of freedom.
#
# USAGE
#   python configs/Custom/tools_staged/inflate_stem_to_nband.py \
#       --src checkpoints/swin_small_p4_w7_mmdet.pth \
#       --dst checkpoints/swin_small_p4_w7_mmdet_4band.pth \
#       --channels 4 --mode mean
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def find_stem_keys(sd: dict) -> list:
    """Weights that look like a stem conv: 4-D with exactly 3 input channels."""
    return [k for k, v in sd.items()
            if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 3]


def inflate(w: torch.Tensor, n: int, mode: str) -> torch.Tensor:
    out_c, _, kh, kw = w.shape
    new = w.new_zeros((out_c, n, kh, kw))
    new[:, :3] = w
    if n > 3 and mode in ('mean', 'scaled'):
        new[:, 3:] = w.mean(dim=1, keepdim=True).repeat(1, n - 3, 1, 1)
    if mode == 'scaled':
        new = new * (3.0 / n)
    return new


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='3-channel checkpoint')
    ap.add_argument('--dst', required=True, help='output checkpoint')
    ap.add_argument('--channels', type=int, required=True,
                    help='number of input bands the tiles carry')
    ap.add_argument('--mode', choices=('mean', 'scaled', 'zero'),
                    default='mean')
    ap.add_argument('--key', default=None,
                    help='inflate exactly this key instead of auto-detecting. '
                         'Use when auto-detection reports more than one '
                         'candidate and you know which is the stem.')
    ap.add_argument('--list', action='store_true',
                    help='list candidate stem keys and exit')
    args = ap.parse_args()

    if args.channels < 3:
        sys.exit('--channels must be at least 3')

    ck = torch.load(args.src, map_location='cpu')
    # Remember WHICH wrapper key held the weights, so the same one can be
    # written back. Spatial-Mamba's loader indexes ckpt['model'] directly and
    # raises KeyError: 'model' if the inflated file is written bare or under
    # 'state_dict' -- and its own loader swallows that into a log line, so the
    # backbone silently trains from random init at full speed.
    wrapper = None
    if isinstance(ck, dict):
        for key in ('state_dict', 'model'):
            if key in ck:
                wrapper = key
                break
    sd = ck[wrapper] if wrapper else ck

    cands = find_stem_keys(sd)
    if args.list:
        print(f'{len(cands)} candidate stem key(s) in {args.src}:')
        for k in cands:
            print(f'  {k}  {tuple(sd[k].shape)}')
        return

    if args.key:
        if args.key not in sd:
            sys.exit(f'--key {args.key!r} not in checkpoint')
        targets = [args.key]
    elif len(cands) == 1:
        targets = cands
    elif not cands:
        sys.exit(
            'No 4-D weight with 3 input channels found. Either this checkpoint '
            'has already been inflated, or the stem is not a plain conv -- '
            'inspect with --list and pass --key explicitly.')
    else:
        # Do not guess. Picking the wrong tensor corrupts the backbone in a way
        # that shows up only as poor accuracy, weeks later.
        print(f'{len(cands)} candidates; pass --key to choose one:',
              file=sys.stderr)
        for k in cands:
            print(f'  {k}  {tuple(sd[k].shape)}', file=sys.stderr)
        sys.exit(1)

    for k in targets:
        old = sd[k]
        sd[k] = inflate(old, args.channels, args.mode)
        print(f'{k}: {tuple(old.shape)} -> {tuple(sd[k].shape)}  '
              f'(mode={args.mode})')
        print(f'   RGB weight sum preserved: '
              f'{torch.allclose(sd[k][:, :3].sum(), old.sum() * (3.0/args.channels if args.mode=="scaled" else 1.0), rtol=1e-4)}')

    out = {wrapper: sd} if wrapper else sd
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dst)
    print(f'-> {args.dst}  (top-level key: {wrapper if wrapper else "<bare state dict>"})')
    print('\nSet the matching in_channels on the backbone, give the '
          'data_preprocessor a mean/std of the same length, and set '
          'bgr_to_rgb=False.')


if __name__ == '__main__':
    main()
