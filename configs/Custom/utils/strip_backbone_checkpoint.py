#!/usr/bin/env python3
"""
Strip a classification checkpoint down to its backbone.
=============================================================================

    python configs/Custom/utils/strip_backbone_checkpoint.py \
        --src swin_small_224_b16x64_300e_imagenet.pth \
        --out checkpoints/swin_small_p4_w7_backbone_only.pth

    # preview without writing:
    python configs/Custom/utils/strip_backbone_checkpoint.py --src f.pth --list

WHY THIS EXISTS
  Detection configs load a backbone, not a classifier. Loading an upstream
  MMPreTrain/timm ImageNet checkpoint into a detector produces a long list
  of unexpected-key warnings (the classification head) at load time that
  hide real ones. The derived `*_backbone_only.pth` files in weights.yaml
  were produced by exactly this operation.

A RECONSTRUCTION, NOT THE ORIGINAL COMMAND
  The command originally used to derive those files was not recorded. This
  script reproduces the operation -- drop head keys, optionally remap the
  `backbone.` prefix -- but a regenerated file is only proven identical to
  the one a config expects by its SHA256 in weights.yaml. If the hashes
  differ, compare key sets first: `--list` prints them.

WHAT IT DOES
  1. Loads the checkpoint (handles both a bare state_dict and the
     {'state_dict': ...} / {'model': ...} wrappers).
  2. Drops keys whose first component is a classification-head name
     (head, fc, classifier, aux_head) or a bookkeeping entry (meta,
     optimizer, param_schedulers, message_hub).
  3. Optionally strips a `backbone.` prefix (--strip-prefix) so keys match
     the namespace the detector's init_cfg expects, or adds one
     (--add-prefix) for the mmdet-namespace variants.
  4. Saves {'state_dict': kept} with the SHA256 of the result printed, to
     be checked against weights.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HEAD_PREFIXES = ('head.', 'fc.', 'classifier.', 'aux_head.')
DROP_TOP_LEVEL = ('meta', 'optimizer', 'param_schedulers', 'message_hub')


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Strip a classification checkpoint to backbone-only form.')
    ap.add_argument('--src', required=True, help='upstream checkpoint')
    ap.add_argument('--out', help='output path (required unless --list)')
    ap.add_argument('--strip-prefix', default=None, metavar='PFX',
                    help="remove this key prefix, e.g. 'backbone.'")
    ap.add_argument('--add-prefix', default=None, metavar='PFX',
                    help="prepend this key prefix, e.g. 'backbone.'")
    ap.add_argument('--list', action='store_true',
                    help='print kept and dropped keys, write nothing')
    args = ap.parse_args()
    if not args.list and not args.out:
        ap.error('--out is required unless --list is given')

    import torch

    blob = torch.load(args.src, map_location='cpu')
    sd = blob
    for wrapper in ('state_dict', 'model'):
        if isinstance(sd, dict) and wrapper in sd and isinstance(
                sd[wrapper], dict):
            sd = sd[wrapper]
            break

    kept, dropped = {}, []
    for k, v in sd.items():
        if k in DROP_TOP_LEVEL or k.startswith(HEAD_PREFIXES):
            dropped.append(k)
            continue
        nk = k
        if args.strip_prefix and nk.startswith(args.strip_prefix):
            nk = nk[len(args.strip_prefix):]
        if args.add_prefix:
            nk = args.add_prefix + nk
        kept[nk] = v

    print(f'{len(kept)} keys kept, {len(dropped)} dropped')
    if args.list:
        for k in dropped:
            print(f'  drop  {k}')
        for k in list(kept)[:10]:
            print(f'  keep  {k}')
        if len(kept) > 10:
            print(f'  ...   and {len(kept) - 10} more')
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': kept}, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f'wrote {out}\nsha256 {sha}')
    print('Check this hash against weights.yaml before using the file.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
