#!/usr/bin/env python3
"""
Copy this repository's backbone wrappers into the installed mmdet package.
=============================================================================

    python tools/install_backbones.py            # copy the wrappers
    python tools/install_backbones.py --dry-run  # show what would be copied

WHY THIS EXISTS
  The configs load the wrappers with

      custom_imports = dict(imports=['mmdet.models.backbones.vmamba_backbone'])

  which requires the wrapper modules to live INSIDE the installed mmdet
  package. In the environment this work ran on, mmdet was a full source
  checkout and the wrappers sat in its tree. The public repository ships
  only the wrapper files, and `pip install mmdet==3.3.0` knows nothing
  about them -- so without this step, every Mamba-family config fails at
  load time with ModuleNotFoundError.

  Run this once after installing the environment (the Dockerfile does).
  It copies mmdet/models/backbones/*.py from this repository into the
  installed package. Nothing else in mmdet is touched; the wrappers
  register themselves and are only imported by configs that ask for them.

  Verify afterwards with:

      python configs/Custom/utils/handover_selftest.py
      python configs/Custom/utils/smoke_build_models.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / 'mmdet' / 'models' / 'backbones'


def installed_backbones_dir() -> Path:
    # The repository's own mmdet/ directory has no __init__.py, so a real
    # installation always wins the import; refuse to run if what we found
    # is this repository itself (mmdet is not installed).
    try:
        import mmdet
    except ImportError:
        sys.exit('mmdet is not installed. Install the environment first '
                 '(see README, "Reproducing the environment").')
    pkg = Path(mmdet.__file__).resolve().parent
    if REPO_ROOT in pkg.parents or pkg == REPO_ROOT / 'mmdet':
        sys.exit('`import mmdet` resolved to this repository, not to an '
                 'installed package. Install mmdet==3.3.0 first.')
    return pkg / 'models' / 'backbones'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--dry-run', action='store_true',
                    help='print the copies without performing them')
    args = ap.parse_args()

    dst = installed_backbones_dir()
    if not dst.is_dir():
        sys.exit(f'not a directory: {dst} -- is mmdet installed correctly?')

    wrappers = sorted(SRC.glob('*_backbone.py'))
    if not wrappers:
        sys.exit(f'no wrapper files found under {SRC}')

    for w in wrappers:
        target = dst / w.name
        state = 'overwrite' if target.exists() else 'new'
        print(f'  {w.name:<32} -> {target}  [{state}]')
        if not args.dry_run:
            shutil.copy2(w, target)

    if args.dry_run:
        print('\nDry run -- nothing copied.')
    else:
        print(f'\n{len(wrappers)} wrapper(s) installed into {dst}')
        print('Verify with: python configs/Custom/utils/handover_selftest.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
