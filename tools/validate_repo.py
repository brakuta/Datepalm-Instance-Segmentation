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
    doc_paths           Commands and examples in READMEs, shell scripts and
                        docstrings referenced experiment folders by their
                        pre-publication names. Markdown links resolved, but
                        every copy-pasteable command failed. This check
                        walks path-like tokens everywhere doc_links cannot
                        see: fenced code, .sh files, .py strings.
    custom_imports      Configs declared custom_imports modules that no
                        published file provided. The _base_ chain resolved,
                        so nothing else caught it.
    no_private_paths    A base config edited away from upstream carried
                        absolute paths from an unrelated project, plus a
                        username in a comment.
    no_artefacts        Checkpoints, imagery and __pycache__ do not belong
                        in a code repository and are easy to add by accident.

  Only files that belong to the repository are checked: tracked files plus
  untracked files that .gitignore does not exclude. The checkpoints/,
  datasets/ and work_dirs/ trees a user is told to create are ignored and
  never fail the checks.

  It needs no torch, no GPU and no network, so it runs on a free CI runner
  in seconds.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def repo_files() -> list[Path]:
    """Tracked files plus untracked-but-not-ignored ones. Falls back to a
    filesystem walk when git is unavailable."""
    try:
        out = subprocess.run(
            ['git', 'ls-files', '-z', '--cached', '--others',
             '--exclude-standard'],
            cwd=ROOT, capture_output=True, check=True).stdout
        files = [ROOT / p.decode('utf-8', 'replace')
                 for p in out.split(b'\0') if p]
        return [f for f in files if f.is_file()]
    except (OSError, subprocess.CalledProcessError):
        return [f for f in ROOT.rglob('*')
                if f.is_file() and '.git' not in f.parts]


FILES = repo_files()


def read(f: Path) -> str:
    return f.read_text(encoding='utf-8', errors='replace')


QUOTED = re.compile(r'''['"]([^'"]+)['"]''')

# Absolute paths and identifiers that must never appear. A drive letter is
# matched only when followed by a separator and a path-like character, so
# `torchvision://resnet50` and `https://` do not trip it.
PRIVATE = [
    ('windows absolute path',
     re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}[A-Za-z0-9_$][A-Za-z0-9_\\/. -]{2,}')),
    ('unix home path', re.compile(r'/home/[a-z0-9_-]+/')),
    ('windows user path',
     re.compile(r'(?:^|[\\/:\s])[Uu]sers[\\/][A-Za-z0-9_.-]+')),
]
# Spans that legitimately look path-shaped are removed before matching, so
# a URL earlier on the line cannot hide a private path after it.
ALLOW_SPAN = re.compile(r'https?://\S+|/path/to/\S*|<[a-z_ -]+>', re.I)
ALLOW_LINE = re.compile(r'noqa:\s*leakscan', re.I)

ARTEFACTS = ('*.pth', '*.ckpt', '*.pkl', '*.npy', '*.tif', '*.tiff',
             '*.gpkg', '*.shp', '*.dbf', '*.shx', '*.prj', '*.cpg',
             '*.parquet', '*.png', '*.jpg', '*.jpeg', '*.pyc')

TEXT = ('.py', '.md', '.txt', '.yaml', '.yml', '.json', '.sh', '.cff')

# Path-like tokens in prose, commands and docstrings. Everything doc_links
# cannot see: fenced code blocks, shell scripts, Python strings. A token is
# checked up to its first placeholder character (*, ?, <, {, $), so
# `configs/Custom/x/maskrcnn_${BB}.py` still verifies that the directory
# exists even though the filename is templated.
PATHLIKE = re.compile(
    r'(?<![\w/.-])((?:configs/Custom|configs/_base_|tools|'
    r'mmdet/models|palm_inference|docker)/'
    r'[A-Za-z0-9_${}*?<>./-]+)')
PLACEHOLDER = re.compile(r'[*?<{$]')
DOC_PATH_EXT = ('.md', '.sh', '.py', '.txt', '.yaml', '.yml')
# Files a documented command generates rather than the repository shipping
# them. Referring to one by its intended path is correct, not a dead link.
GENERATED = {
    'configs/Custom/Feature_Analysis/config_feature_analysis.json',
}


def configs() -> list[Path]:
    return [f for f in FILES if f.suffix == '.py' and 'configs' in f.parts]


def config_inheritance():
    """Every _base_ reference resolves to a file that exists."""
    bad, n = [], 0
    for cfg in configs():
        m = re.search(r'''_base_\s*=\s*(\[.*?\]|['"][^'"]+['"])''',
                      read(cfg), re.S)
        if not m:
            continue
        for rel in QUOTED.findall(m.group(1)):
            if not rel.endswith('.py'):
                continue
            n += 1
            if not (cfg.parent / rel).resolve().exists():
                bad.append(f'{cfg.relative_to(ROOT)} -> {rel}')
    return f'{n} _base_ references', bad


def doc_links():
    """Every relative link in a markdown file points at something real."""
    bad, n = [], 0
    for md in FILES:
        if md.suffix != '.md':
            continue
        for text, link in re.findall(r'\[([^\]]+)\]\(([^)#]+)\)', read(md)):
            if link.startswith(('http', 'mailto:', '#')):
                continue
            n += 1
            if not (md.parent / link).resolve().exists():
                bad.append(f'{md.relative_to(ROOT)}: [{text}] -> {link}')
    return f'{n} relative links', bad


def doc_paths():
    """Every repo-path-like token in docs, scripts and docstrings exists."""
    bad, n = [], 0
    for f in FILES:
        if f.suffix not in DOC_PATH_EXT or f.name == 'validate_repo.py':
            continue
        for i, line in enumerate(read(f).splitlines(), 1):
            for token in PATHLIKE.findall(line):
                token = token.rstrip('.,;:>}')
                if token in GENERATED:
                    continue
                m = PLACEHOLDER.search(token)
                if m:
                    # Templated: require the deepest literal directory.
                    literal = token[:m.start()]
                    target = (ROOT / literal).parent if not literal.endswith('/') \
                        else ROOT / literal.rstrip('/')
                else:
                    target = ROOT / token
                n += 1
                if not target.exists():
                    bad.append(f'{f.relative_to(ROOT)}:{i} -> {token}')
    return f'{n} path tokens', bad


def custom_imports():
    """Every custom_imports module a config declares is a published file."""
    bad, n = [], 0
    for cfg in configs():
        m = re.search(r'custom_imports\s*=\s*dict\s*\((.*?)\)\s*$', read(cfg),
                      re.S | re.M)
        m = m and re.search(r'imports\s*=\s*\[(.*?)\]', m.group(1), re.S)
        if not m:
            continue
        for mod in QUOTED.findall(m.group(1)):
            # configs.* and the *_backbone wrappers must be published here;
            # anything else under mmdet.* ships with the installed package.
            if not (mod.startswith(('configs.', 'palm_inference.'))
                    or (mod.startswith('mmdet.')
                        and mod.rsplit('.', 1)[-1].endswith('_backbone'))):
                continue
            n += 1
            if not (ROOT / (mod.replace('.', '/') + '.py')).exists():
                bad.append(f'{cfg.relative_to(ROOT)} -> {mod}')
    return f'{n} custom imports', bad


def no_private_paths():
    bad, n = [], 0
    for f in FILES:
        if f.suffix not in TEXT:
            continue
        n += 1
        for i, line in enumerate(read(f).splitlines(), 1):
            if ALLOW_LINE.search(line):
                continue
            probe = ALLOW_SPAN.sub('', line)
            for kind, rx in PRIVATE:
                if rx.search(probe):
                    bad.append(f'{f.relative_to(ROOT)}:{i} [{kind}] '
                               f'{line.strip()[:90]}')
                    break
    return f'{n} text files', bad


def no_artefacts():
    bad = []
    for f in FILES:
        if any(fnmatch.fnmatch(f.name, pat) for pat in ARTEFACTS) \
                or '__pycache__' in f.parts:
            bad.append(str(f.relative_to(ROOT)))
    return 'checkpoints, imagery, caches', bad


CHECKS = [
    ('config inheritance', config_inheritance),
    ('documentation links', doc_links),
    ('documentation paths', doc_paths),
    ('custom imports resolve', custom_imports),
    ('no private paths', no_private_paths),
    ('no data artefacts', no_artefacts),
]


def main():
    print(f'validating {ROOT}  ({len(FILES)} files)\n')
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
