#!/usr/bin/env python3
"""
Check that this repository is internally consistent. Standard library only.
=============================================================================
Run by CI on every push, and runnable locally in a second:

    python tools/validate_repo.py

WHY THIS EXISTS
  Every check below corresponds to a defect that was actually present in
  this repository at some point and was found by hand. A check that runs
  automatically is the difference between "we fixed it" and "it stays
  fixed".

    config_inheritance  Two configs inherited a base file that had not been
                        published. They could not be loaded by anyone, and
                        nothing said so until someone tried.
    doc_links           A README named files that were absent -- the
                        installation verifier and the build recipe.
    no_private_paths    A base config edited away from upstream carried
                        absolute paths from an unrelated project, plus a
                        username in a comment.
    no_artefacts        Checkpoints, imagery and __pycache__ do not belong
                        in a code repository and are easy to add by accident.

  It needs no torch, no GPU and no network, so it runs on a free CI runner
  in seconds.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Absolute paths and identifiers that must never appear. A drive letter is
# matched only when followed by a separator and a path-like character, so
# `torchvision://resnet50` and `https://` do not trip it.
PRIVATE = [
    ('windows absolute path',
     re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}[A-Za-z0-9_$][A-Za-z0-9_\\/. -]{2,}')),
    ('unix home path', re.compile(r'/home/[a-z0-9_-]+/')),
    ('windows user path', re.compile(r'[Uu]sers[\\/][A-Za-z0-9_.-]+')),
]
# Lines that legitimately look path-shaped.
ALLOW = re.compile(r'https?://|/path/to/|<[a-z_ -]+>|noqa:\s*leakscan', re.I)

ARTEFACTS = ('*.pth', '*.ckpt', '*.pkl', '*.npy', '*.tif', '*.tiff',
             '*.gpkg', '*.shp', '*.pyc')

TEXT = ('.py', '.md', '.txt', '.yaml', '.yml', '.json', '.sh', '.cff')


def config_inheritance():
    """Every _base_ reference resolves to a file that exists."""
    bad, n = [], 0
    for cfg in (ROOT / 'configs').rglob('*.py'):
        m = re.search(r'_base_\s*=\s*\[(.*?)\]', cfg.read_text(errors='replace'), re.S)
        if not m:
            continue
        for rel in re.findall(r"'([^']+\.py)'", m.group(1)):
            n += 1
            if not (cfg.parent / rel).resolve().exists():
                bad.append(f'{cfg.relative_to(ROOT)} -> {rel}')
    return f'{n} _base_ references', bad


def doc_links():
    """Every relative link in a markdown file points at something real."""
    bad, n = [], 0
    for md in ROOT.rglob('*.md'):
        if '.git' in md.parts:
            continue
        for text, link in re.findall(r'\[([^\]]+)\]\(([^)#]+)\)',
                                     md.read_text(errors='replace')):
            if link.startswith(('http', 'mailto:', '#')):
                continue
            n += 1
            if not (md.parent / link).resolve().exists():
                bad.append(f'{md.relative_to(ROOT)}: [{text}] -> {link}')
    return f'{n} relative links', bad


def no_private_paths():
    bad, n = [], 0
    for f in ROOT.rglob('*'):
        if '.git' in f.parts or not f.is_file() or f.suffix not in TEXT:
            continue
        n += 1
        for i, line in enumerate(f.read_text(errors='replace').splitlines(), 1):
            if ALLOW.search(line):
                continue
            for kind, rx in PRIVATE:
                if rx.search(line):
                    bad.append(f'{f.relative_to(ROOT)}:{i} [{kind}] '
                               f'{line.strip()[:90]}')
                    break
    return f'{n} text files', bad


def no_artefacts():
    bad = []
    for pat in ARTEFACTS:
        for f in ROOT.rglob(pat):
            if '.git' not in f.parts:
                bad.append(str(f.relative_to(ROOT)))
    for d in ROOT.rglob('__pycache__'):
        if '.git' not in d.parts:
            bad.append(str(d.relative_to(ROOT)) + '/')
    return 'checkpoints, imagery, caches', bad


CHECKS = [
    ('config inheritance', config_inheritance),
    ('documentation links', doc_links),
    ('no private paths', no_private_paths),
    ('no data artefacts', no_artefacts),
]


def main():
    print(f'validating {ROOT}\n')
    failed = 0
    for name, fn in CHECKS:
        scope, bad = fn()
        status = 'FAIL' if bad else ' OK '
        print(f'[{status}] {name:<24} {scope}')
        if bad:
            failed += 1
            for b in bad[:20]:
                print(f'         {b}')
            if len(bad) > 20:
                print(f'         ... and {len(bad) - 20} more')
    print()
    if failed:
        print(f'{failed} check(s) FAILED.')
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
