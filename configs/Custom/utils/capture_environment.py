#!/usr/bin/env python3
"""
Capture everything about a machine that stops being knowable once it is gone.
=============================================================================
WHY THIS EXISTS
  A results table is only reproducible if the environment that produced it can
  be reconstructed. Version pins in a requirements file are not that: they
  record what was ASKED FOR, not what was RESOLVED, and they say nothing about
  the CUDA toolkit, the driver, the compiled extensions, or which of several
  code paths a model actually took at runtime.

  That gap is normally an annoyance. Here it is terminal: the workstations that
  produced every number in the manuscript become unreachable shortly, and
  anything not written down before then is gone. A missing pin can be guessed
  later; a missing compiled-extension fact cannot.

  This project already has the exact case in view. A calibration run logged

      No module named 'selective_scan_vmamba_pt202'

  so VMamba fell back to its pure-PyTorch path instead of the fused CUDA
  kernel. Upstream states the consequence as SPEED and ships numerical-
  equivalence tests, claiming no accuracy difference -- so this is not evidence
  that anything is wrong. It is a fact that has to be RECORDED, because a
  reader who builds the kernel and sees any difference needs to know which
  path produced the published numbers, and nobody can establish that later.

  Note also that the extension name above is a LOCAL build tag. Upstream
  VMamba builds `selective_scan_cuda_oflex`. Probing one name would therefore
  report a false alarm on a machine that built it under another, which is why
  every known variant is probed and the warning fires only if none load.

WHAT IT RECORDS, AND WHY EACH MATTERS
  interpreter/platform  the floor everything else stands on
  pip freeze            RESOLVED versions, not requested ranges
  conda explicit        exact build strings -- conda's version alone is not
                        enough to reproduce a binary environment
  torch / CUDA / cuDNN  the numerical substrate; cuDNN algorithm selection can
                        move detection metrics in the third decimal
  GPU + driver          kernel availability and TF32 behaviour differ by
                        architecture, silently
  mmengine collect_env  the OpenMMLab-canonical block, including the compiler
                        MMCV's ops were built with
  compiled extensions   probed by IMPORT, not by reading a requirements file,
                        because the file cannot know whether the build worked
  library code paths    which implementation each SSM backbone would actually
                        select on this machine
  checkpoint SHA256     identity of every weight file; a filename is not an
                        identity, and several of these exist nowhere else

WHAT IT DELIBERATELY DOES NOT DO
  It does not sanitise. The output is a factual record of a private machine and
  will contain absolute paths, usernames and possibly hostnames. Review it
  before it goes anywhere public -- `--redact-home` masks the obvious cases but
  is a convenience, not a guarantee.

  It does not capture the Docker image. That cannot be seen from inside a
  container: run the companion commands printed at the end ON THE HOST.

USAGE
  # inside the training container
  python capture_environment.py --out env_capture/container \
      --checkpoints /workspace/mmdetection/checkpoints \
      --checkpoints /workspace/mmdetection/work_dirs

  # in each conda env on the host
  python capture_environment.py --out env_capture/ws1_gdal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Compiled CUDA extensions used by the SSM backbones. Probed by import: a
# requirements file records the intent to build, never the outcome, and a
# failed build degrades to a slower path SILENTLY rather than raising.
CUDA_EXTENSIONS = [
    'selective_scan_vmamba_pt202', 'selective_scan_vmamba', 'selective_scan',
    'selective_scan_cuda', 'selective_scan_cuda_core',
    'selective_scan_cuda_oflex', 'selective_scan_cuda_ndstate',
    'selective_scan_cuda_nrow', 'mamba_ssm', 'causal_conv1d',
    'natten', 'mmcv._ext', 'mmcv.ops',
]

# Packages whose exact version changes results rather than merely convenience.
KEY_PACKAGES = [
    'torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'mmpretrain',
    'timm', 'numpy', 'cv2', 'pycocotools', 'shapely', 'rasterio',
    'geopandas', 'pyogrio', 'einops', 'triton', 'ninja', 'transformers',
    # Import names, NOT distribution names: opencv-python imports as cv2 and
    # huggingface-hub as huggingface_hub. Probing the distribution name
    # records a present package as missing, which is worse than not probing
    # it -- the capture exists to be believed later.
    'huggingface_hub', 'safetensors',
]

CKPT_EXTS = {'.pth', '.pt', '.ckpt', '.safetensors', '.bin'}


def run(cmd, timeout=180):
    """Command output, or a recorded reason it produced none. Never raises:
    a capture that aborts halfway is worse than one with gaps, because the
    machine may not be available for a second attempt."""
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), timeout=timeout,
                           capture_output=True, text=True)
        out = (p.stdout or '') + (('\n[stderr]\n' + p.stderr) if p.stderr
                                  else '')
        return out.strip() or f'[no output, exit {p.returncode}]'
    except FileNotFoundError:
        return '[not installed]'
    except subprocess.TimeoutExpired:
        return f'[timed out after {timeout}s]'
    except Exception as exc:                                  # noqa: BLE001
        return f'[{type(exc).__name__}: {exc}]'


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


def _flush(out, record):
    """Persist what has been gathered so far. Cheap, and the difference
    between a partial capture and no capture at all."""
    try:
        (Path(out) / 'environment.json').write_text(
            json.dumps(record, indent=2), encoding='utf-8')
    except Exception as exc:                                  # noqa: BLE001
        print(f'  [WARN] could not write environment.json: {exc}')


def probe_imports(names):
    """Import each and record version/location. Import, not pip-list: the
    question is what this interpreter can actually load right now."""
    out = {}
    for n in names:
        try:
            mod = __import__(n, fromlist=['__version__'])
            out[n] = {
                'importable': True,
                'version': str(getattr(mod, '__version__', 'n/a')),
                'file': str(getattr(mod, '__file__', 'n/a')),
            }
        except Exception as exc:                              # noqa: BLE001
            out[n] = {'importable': False,
                      'error': f'{type(exc).__name__}: {exc}'}
    return out


def torch_block():
    try:
        import torch
    except Exception as exc:                                  # noqa: BLE001
        return {'error': f'torch not importable: {exc}'}
    d = {
        'version': torch.__version__,
        'cuda_compiled': torch.version.cuda,
        'cudnn': (torch.backends.cudnn.version()
                  if torch.backends.cudnn.is_available() else None),
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count()
        if torch.cuda.is_available() else 0,
        # These three change numerics, are rarely recorded, and default
        # differently across torch releases.
        'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
    }
    if d['cuda_available']:
        d['devices'] = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            d['devices'].append({
                'name': p.name, 'capability': f'{p.major}.{p.minor}',
                'total_memory_GB': round(p.total_memory / 1e9, 2),
                'multi_processor_count': p.multi_processor_count})
    return d


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--checkpoints', action='append', default=[],
                    help='directory to inventory with SHA256 (repeatable). '
                         'Hashing is slow on large trees; --no-hash records '
                         'sizes only.')
    ap.add_argument('--no-hash', action='store_true',
                    help='skip SHA256. Only for a first quick pass -- a '
                         'filename is not an identity, and several of these '
                         'weights exist nowhere else.')
    ap.add_argument('--label', default=None,
                    help='name for this environment, e.g. ws1-container')
    ap.add_argument('--redact-home', action='store_true',
                    help='mask the home directory in captured text. A '
                         'convenience, not a guarantee -- review before '
                         'publishing.')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or platform.node()
    print(f'capturing "{label}" -> {out}\n')

    # ---- text captures --------------------------------------------------
    commands = {
        'pip_freeze': [sys.executable, '-m', 'pip', 'freeze'],
        'pip_list': [sys.executable, '-m', 'pip', 'list', '--format=columns'],
        'conda_list_explicit': 'conda list --explicit',
        'conda_env_export': 'conda env export',
        'conda_info': 'conda info',
        'nvidia_smi': 'nvidia-smi',
        'nvidia_smi_query': ('nvidia-smi --query-gpu=name,driver_version,'
                             'memory.total,compute_cap --format=csv'),
        'nvcc_version': 'nvcc --version',
        'gcc_version': 'gcc --version',
        'os_release': ('cat /etc/os-release' if os.name != 'nt' else 'ver'),
        'uname': ('uname -a' if os.name != 'nt' else 'systeminfo'),
        'cpu_count': f'{sys.executable} -c "import os;print(os.cpu_count())"',
    }
    for name, cmd in commands.items():
        txt = run(cmd)
        if args.redact_home:
            home = str(Path.home())
            txt = txt.replace(home, '~').replace(home.replace('\\', '/'), '~')
        (out / f'{name}.txt').write_text(txt, encoding='utf-8')
        head = txt.splitlines()[0][:60] if txt.splitlines() else ''
        print(f'  {name:22s} {head}')

    # ---- mmengine's canonical block -------------------------------------
    try:
        from mmengine.utils.dl_utils import collect_env
        env = collect_env()
        (out / 'mmengine_collect_env.txt').write_text(
            '\n'.join(f'{k}: {v}' for k, v in env.items()), encoding='utf-8')
        print('  mmengine_collect_env   captured')
    except Exception as exc:                                  # noqa: BLE001
        (out / 'mmengine_collect_env.txt').write_text(f'[unavailable: {exc}]')
        print(f'  mmengine_collect_env   [unavailable: {exc}]')

    # ---- structured record ----------------------------------------------
    # Written after EVERY stage, not once at the end. Two of the stages can
    # take the interpreter down with them: hashing a checkpoint that a live
    # training job rotates mid-scan raises, and importing a CUDA extension
    # against a mismatched driver can abort() the process outright. A capture
    # that dies having written nothing is the one outcome that cannot be
    # retried once the machine is gone.
    record = {
        'label': label,
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'platform': {
            'python': sys.version, 'executable': sys.executable,
            'implementation': platform.python_implementation(),
            'system': platform.system(), 'release': platform.release(),
            'machine': platform.machine(), 'node': platform.node(),
            'in_container': Path('/.dockerenv').exists(),
        },
        'env_vars': {k: os.environ.get(k) for k in (
            'CUDA_HOME', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH',
            'PYTHONPATH', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV',
            'TORCH_CUDA_ARCH_LIST', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG') if k in os.environ},
        'complete': False,
    }
    _flush(out, record)                      # survives everything below

    record['torch'] = torch_block()
    _flush(out, record)

    record['key_packages'] = probe_imports(KEY_PACKAGES)
    _flush(out, record)

    # Most likely stage to abort the interpreter.
    record['cuda_extensions'] = probe_imports(CUDA_EXTENSIONS)
    _flush(out, record)

    built = [k for k, v in record['cuda_extensions'].items() if v['importable']]
    missing = [k for k, v in record['cuda_extensions'].items()
               if not v['importable']]
    record['cuda_extensions_summary'] = {'built': built, 'missing': missing}

    # ---- checkpoint inventory -------------------------------------------
    inv = []
    for root in args.checkpoints:
        rp = Path(root)
        if not rp.exists():
            inv.append({'root': str(rp), 'error': 'does not exist'})
            print(f'\n  [WARN] checkpoint root missing: {rp}')
            continue
        files = sorted(p for p in rp.rglob('*')
                       if p.suffix.lower() in CKPT_EXTS and p.is_file())
        print(f'\n  hashing {len(files)} weight file(s) under {rp} ...')
        for i, p in enumerate(files, 1):
            # A training job rotating a checkpoint between the scan and the
            # stat raises FileNotFoundError. Unguarded, that ended the whole
            # capture with nothing written.
            rec = {'root': str(rp), 'path': str(p)}
            try:
                st = p.stat()
                rec.update(relpath=str(p.relative_to(rp)), bytes=st.st_size,
                           mtime_utc=datetime.fromtimestamp(
                               st.st_mtime, timezone.utc).isoformat())
            except Exception as exc:                          # noqa: BLE001
                rec['stat_error'] = f'{type(exc).__name__}: {exc}'
                inv.append(rec)
                continue
            if not args.no_hash:
                try:
                    rec['sha256'] = sha256(p)
                except Exception as exc:                      # noqa: BLE001
                    rec['sha256_error'] = f'{type(exc).__name__}: {exc}'
            inv.append(rec)
            if i % 20 == 0:
                record['checkpoints'] = inv
                _flush(out, record)
            if i % 10 == 0 or i == len(files):
                print(f'    {i}/{len(files)}', end='\r')
        print()
    record['checkpoints'] = inv
    record['checkpoint_total_bytes'] = sum(
        r.get('bytes', 0) for r in inv if 'bytes' in r)
    record['complete'] = True

    _flush(out, record)

    # ---- report ----------------------------------------------------------
    print('\n' + '=' * 70)
    print(f'wrote {out}/environment.json  + {len(commands) + 1} text files')
    n_ck = sum(1 for r in inv if 'bytes' in r)
    print(f'checkpoints inventoried : {n_ck} '
          f'({record["checkpoint_total_bytes"] / 1e9:.1f} GB)')
    print(f'CUDA extensions BUILT   : {", ".join(built) or "NONE"}')
    print(f'CUDA extensions missing : {len(missing)}')
    # Upstream VMamba builds the extension as `selective_scan_cuda_oflex`
    # (kernels/selective_scan, MODES=["oflex"]). Names like
    # `selective_scan_vmamba_pt202` are LOCAL build tags, so testing for one
    # name alone reports a false alarm on a machine that built it under
    # another. The question is whether ANY variant is importable.
    scan_built = [k for k in built if k.startswith('selective_scan')]
    if not scan_built:
        print()
        print('*' * 70)
        print('No selective_scan CUDA extension is importable here, under any')
        print('of the names probed. VMamba-family backbones therefore fall')
        print('back to the pure-PyTorch path in csms6s.py.')
        print()
        print('Upstream states the consequence as SPEED and ships')
        print('numerical-equivalence tests; it claims no accuracy difference.')
        print('So this is a fact to RECORD, not evidence that results are')
        print('wrong -- but record it, because a reader who builds the kernel')
        print('and sees any difference needs to know which path produced the')
        print('published numbers.')
        print('*' * 70)
    else:
        print(f'selective_scan variant(s) built : {", ".join(scan_built)}')
    print()
    print('STILL TO CAPTURE -- these cannot be seen from in here.')
    print('Run ON THE DOCKER HOST, not inside the container:')
    print('  docker ps -a --no-trunc                    > docker_ps.txt')
    print('  docker inspect <container>                 > docker_inspect.json')
    print('  docker image inspect <image>               > docker_image.json')
    print('  docker history --no-trunc <image>          > docker_history.txt')
    print('  docker save <image> | gzip > image.tar.gz   # the real fallback')
    print('The image tarball is the only capture that survives a lost')
    print('Dockerfile, a deleted registry tag, and a pinned dependency that')
    print('gets yanked from PyPI.')
    print('=' * 70)


if __name__ == '__main__':
    main()
