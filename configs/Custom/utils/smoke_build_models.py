#!/usr/bin/env python3
"""
Actually build every model and push a tensor through it.
=============================================================================
WHY THIS EXISTS

  handover_selftest.py proves the environment IMPORTS. It says so itself:

      "The Mamba-family kernels cannot be tested this way and are only
       exercised once a real model runs."

  That gap is not academic. An SSM backbone can import cleanly and still
  fail at the first forward pass, because the CUDA kernel it calls was
  compiled for a different architecture. The import touches Python; the
  kernel launch touches the GPU. Only the second one finds
  "no kernel image is available for execution on the device".

  So this builds each model from its real config and runs one forward pass
  on a dummy image. No checkpoint, no data, no GPU-hours -- it is the
  cheapest test that exercises the code path a real job depends on.

WHAT A FAILURE HERE MEANS
  A backbone that fails is a backbone this machine cannot run. That is
  useful rather than fatal: the CNN and transformer models are independent
  of the SSM kernels, so a machine that fails on vmamba can still run
  swin_s and convnext_t. The summary says which, so a recipient knows what
  they have rather than concluding the delivery is broken.

USAGE
  python smoke_build_models.py                       # the Stage C set
  python smoke_build_models.py --config <one.py>     # just one
  python smoke_build_models.py --size 512            # smaller dummy image
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback

OK, FAIL, SKIP = 'OK', 'FAIL', 'SKIP'


def build_and_run(cfg_path, size, device):
    """Return (status, detail, seconds)."""
    import torch
    from mmdet.apis import init_detector

    t0 = time.time()
    try:
        model = init_detector(cfg_path, None, device=device)
    except Exception as exc:                                   # noqa: BLE001
        return FAIL, f'build failed: {type(exc).__name__}: {exc}', time.time() - t0

    try:
        # A forward pass through the BACKBONE alone. The detector head needs
        # data-structure metadata this test deliberately does not fabricate --
        # the kernels under test all live in the backbone.
        x = torch.randn(1, 3, size, size, device=device)
        with torch.no_grad():
            feats = model.backbone(x)
        shapes = [tuple(f.shape) for f in feats] if isinstance(feats, (list, tuple)) \
            else [tuple(feats.shape)]
        torch.cuda.synchronize() if device.startswith('cuda') else None
        return OK, f'{len(shapes)} feature map(s), first {shapes[0]}', time.time() - t0
    except Exception as exc:                                   # noqa: BLE001
        msg = f'{type(exc).__name__}: {exc}'
        if 'no kernel image' in str(exc):
            msg += ('\n           -> the kernel was compiled for a different '
                    'GPU architecture.\n              Rebuild the kernels with a wider '
                    'TORCH_CUDA_ARCH_LIST\n              (see docker/Dockerfile.reconstructed)')
        return FAIL, f'forward failed: {msg}', time.time() - t0
    finally:
        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', action='append', default=[],
                    help='a specific config; repeatable. Default: the Stage C set')
    ap.add_argument('--glob', default='configs/Custom/3_unified_multisource/maskrcnn_*.py',
                    help='which configs to try when --config is not given')
    ap.add_argument('--size', type=int, default=1024,
                    help='dummy image side in pixels (default 1024, the '
                         'deployment tile size)')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    cfgs = args.config or sorted(glob.glob(args.glob))
    if not cfgs:
        sys.exit(f'[ERROR] no configs matched {args.glob}\n'
                 f'        Run this from the repository root.')

    try:
        import torch
    except ImportError:
        sys.exit('[ERROR] torch is not importable. You are not in the '
                 'project container.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        sys.exit('[ERROR] no GPU visible. Start the container with --gpus all.')

    print('=' * 68)
    print(' BUILDING EVERY MODEL AND RUNNING ONE FORWARD PASS')
    print('=' * 68)
    if args.device.startswith('cuda'):
        print(f' gpu    : {torch.cuda.get_device_name(0)} '
              f'(sm_{"".join(map(str, torch.cuda.get_device_capability(0)))})')
    print(f' input  : 1 x 3 x {args.size} x {args.size}')
    print(f' models : {len(cfgs)}')
    print('=' * 68 + '\n')

    results = []
    for cfg in cfgs:
        name = os.path.basename(cfg).replace('.py', '')
        print(f'  {name:.<52}', end=' ', flush=True)
        status, detail, secs = build_and_run(cfg, args.size, args.device)
        print(f'{status:4s} {secs:5.1f}s')
        if status != OK:
            print(f'           {detail}')
        results.append((status, name, detail))

    good = [r for r in results if r[0] == OK]
    bad = [r for r in results if r[0] == FAIL]

    print('\n' + '=' * 68)
    print(f' {len(good)} of {len(results)} model(s) built and ran')
    print('=' * 68)

    if bad:
        print('\n THESE CANNOT RUN ON THIS MACHINE:\n')
        for _, name, detail in bad:
            print(f'   {name}')
            print(f'     {detail.splitlines()[0]}')
        print('\n This is not necessarily a broken delivery. The CNN and')
        print(' transformer backbones do not use the SSM kernels, so if')
        print(' those passed, this machine can still train and run them.')
        print(' See the GPU-architecture note in the README ("Reproducing')
        print(' the environment") before concluding anything about the hardware.')
    else:
        print('\n Every model built and executed a forward pass. The compiled')
        print(' kernels work on this GPU -- which importing them does not')
        print(' prove, and which is what a real job depends on.')

    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
