#!/usr/bin/env python3
"""
Self-test for the handover package. Run this FIRST, before anything else.
=============================================================================
WHO THIS IS FOR
  Someone who has been handed this project and needs to know whether their
  machine is set up correctly -- without reading any code, and without anyone
  to ask.

  Every check prints PASS or FAIL. Every FAIL prints what is wrong in plain
  language and the exact command that fixes it. Nothing here changes anything
  on the machine; it only looks and reports.

WHY IT EXISTS
  Without it, each of these problems reaches the user as a Python traceback
  thirty lines long, usually in the middle of a job that has already been
  running for an hour:

    - the GPU is not visible inside the container
    - the GPU is a different generation from the one the kernels were built
      for, so the model loads and then fails at the first convolution
    - a compiled CUDA extension is missing, so a model silently runs ten
      times slower on a fallback path
    - the imagery folder is laid out differently than the pipeline expects
    - the checkpoint file is missing or truncated

  A traceback tells a programmer where to look. It tells everyone else
  nothing. This exists so that the answer arrives in the first minute
  rather than after a day of guessing.

THE CHECK THAT MATTERS MOST
  GPU ARCHITECTURE. CUDA extensions are compiled for specific GPU
  generations. If they were built on one machine and run on a newer one, the
  failure is silent until the moment a kernel is launched -- so it appears
  mid-run, long after everything looked fine. This test compares the GPU
  present against the architectures each extension was actually built for,
  and says so before any time is wasted.

USAGE
  python handover_selftest.py
  python handover_selftest.py --data /data/my_imagery --checkpoint /models/x.pth
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PASS, FAIL, WARN, INFO = 'PASS', 'FAIL', 'WARN', 'INFO'
RESULTS = []
# Imagery the pipeline can read. Kept in one place so the message the user
# sees and the check itself can never disagree.
IMAGE_EXTS = {'.tif', '.tiff', '.img', '.jp2', '.vrt'}


def report(status, title, detail='', fix=''):
    RESULTS.append((status, title, detail, fix))
    mark = {PASS: '  OK  ', FAIL: ' FAIL ', WARN: ' WARN ', INFO: ' INFO '}[status]
    print(f'[{mark}] {title}')
    if detail:
        for line in str(detail).splitlines():
            print(f'         {line}')
    if fix and status in (FAIL, WARN):
        print('         HOW TO FIX:')
        for line in fix.splitlines():
            print(f'           {line}')
    print()


def check_python():
    v = sys.version_info
    if v[:2] < (3, 8):
        report(FAIL, f'Python {v.major}.{v.minor}',
               'This project needs Python 3.8 or newer.',
               'You are probably running the wrong python. Inside the\n'
               'container the right one is /usr/bin/python3 or the conda\n'
               'environment named in the documentation.')
    else:
        report(PASS, f'Python {v.major}.{v.minor}.{v.micro}')


def check_in_container():
    inside = Path('/.dockerenv').exists()
    if inside:
        report(PASS, 'Running inside the Docker container')
    else:
        report(WARN, 'NOT running inside the Docker container',
               'You appear to be on the host machine, not inside the '
               'container.\nThe project is only supported inside the '
               'container: the host is\nmissing the compiled parts and the '
               'exact library versions.',
               'Start the container first -- the docker run command is at the\n'
               'end of docker/Dockerfile.reconstructed -- then run this\n'
               'test again from inside it.')


def check_torch():
    try:
        import torch
    except ImportError:
        report(FAIL, 'PyTorch is not installed',
               'Nothing can run without it.',
               'You are outside the container, or the environment was not\n'
               'built correctly. Rebuild from docker/Dockerfile.reconstructed.')
        return None
    report(PASS, f'PyTorch {torch.__version__}',
           f'built against CUDA {torch.version.cuda}')
    return torch


def check_gpu(torch):
    if torch is None:
        # Silence would read as "this check passed". Say why it was skipped.
        report(INFO, 'GPU check skipped',
               'Cannot check the GPU because PyTorch is not available.')
        return None
    if not torch.cuda.is_available():
        report(FAIL, 'No GPU is visible to this container',
               'PyTorch cannot see any NVIDIA GPU. Training and inference '
               'will not run.',
               '1. On the HOST, check the GPU exists:   nvidia-smi\n'
               '2. If that fails, install the NVIDIA driver.\n'
               '3. If it works on the host but not here, the container was\n'
               '   started without GPU access. Stop it and start it again\n'
               '   WITH the --gpus all option, exactly as the\n'
               '   documentation shows.')
        return None
    n = torch.cuda.device_count()
    p = torch.cuda.get_device_properties(0)
    cc = f'{p.major}.{p.minor}'
    report(PASS, f'GPU visible: {p.name}',
           f'{n} device(s), compute capability {cc}, '
           f'{p.total_memory / 1e9:.1f} GB memory')
    if p.total_memory < 10e9:
        report(WARN, 'This GPU has less memory than the one used to build '
                     'the project',
               f'{p.total_memory / 1e9:.1f} GB available; the original work '
               f'used 24 GB.\nLarge images may fail with an '
               f'"out of memory" error.',
               'If a run fails with CUDA out of memory, lower the batch\n'
               'size as described in the documentation. Nothing else\n'
               'needs to change.')
    return p


def check_cuda_arch(torch, props):
    """The check that prevents a mid-run failure nobody can diagnose.

    An extension compiled for one GPU generation raises `no kernel image is
    available for execution on the device` on a newer one -- at the moment a
    kernel first launches, not at import. So everything looks correct until
    an hour into a job.
    """
    if torch is None or props is None:
        report(INFO, 'GPU architecture check skipped',
               'Cannot check GPU compatibility without a working PyTorch '
               'and GPU.\nThis is the check that catches "no kernel image '
               'is available"\nfailures, so re-run it once the GPU is '
               'visible.')
        return
    have = f'sm_{props.major}{props.minor}'
    try:
        built = torch.cuda.get_arch_list()
    except Exception:                                          # noqa: BLE001
        report(WARN, 'Could not read the list of supported GPU architectures')
        return
    if have in built:
        report(PASS, f'PyTorch supports this GPU ({have})',
               f'built for: {", ".join(built)}')
    elif any(b.startswith('compute_') for b in built):
        report(WARN, f'PyTorch has no exact code for this GPU ({have})',
               f'built for: {", ".join(built)}\n'
               f'It contains forward-compatible (PTX) code, so it should '
               f'still work,\nbut the first run will be slower while that '
               f'code is compiled.',
               'Nothing to do. If a run fails with "no kernel image is\n'
               'available", report that message to the project author.')
    else:
        report(FAIL, f'PyTorch cannot run on this GPU ({have})',
               f'built for: {", ".join(built)}\n'
               f'This GPU is a different generation. Jobs will fail with\n'
               f'"no kernel image is available for execution on the device",\n'
               f'usually after running for some time.',
               'This cannot be fixed by settings. The container image needs\n'
               'to be rebuilt for this GPU. Report this whole message to\n'
               'the project author.')


def check_extensions():
    """Compiled CUDA kernels. A missing one is not an error -- the code falls
    back to a slower path silently -- so it has to be reported, or a user
    concludes the hardware is slow."""
    # "required" means a backbone wrapper refuses to build without it.
    wanted = {
        'mmcv._ext': ('required', 'The core operations used by every model.'),
        'selective_scan_cuda_oflex': (
            'required', 'VMamba, EfficientVMamba and MSVMamba kernel.'),
        'selective_scan_cuda_core': ('required', 'GroupMamba kernel.'),
        'selective_scan_cuda_oflex_rh': ('required', 'Spatial-Mamba kernel.'),
        'dwconv2d': ('required', 'Spatial-Mamba depthwise convolution.'),
        'selective_scan_cuda': ('optional', 'Alternative fast Mamba kernel.'),
        'mamba_ssm': ('required', 'MambaVision kernel and support library.'),
        'causal_conv1d': ('required', 'Mamba support library.'),
    }
    missing_required, missing_optional = [], []
    for mod, (need, what) in wanted.items():
        try:
            __import__(mod)
            report(PASS, f'{mod} loaded', what)
        except Exception as exc:                              # noqa: BLE001
            (missing_required if need == 'required'
             else missing_optional).append(mod)
            if need == 'required':
                report(FAIL, f'{mod} is MISSING', f'{what}\n{exc}',
                       'The environment is incomplete. Rebuild it from\n'
                       'docker/Dockerfile.reconstructed -- the missing piece is\n'
                       'one of the compiled kernels it builds.')
    if missing_optional:
        report(WARN, f'{len(missing_optional)} optional fast kernel(s) missing',
               ', '.join(missing_optional) + '\n'
               'The models will still run and give the same answers, but\n'
               'the Mamba-family models will be slower.',
               'Nothing needs fixing to get correct results. If speed\n'
               'matters, report this to the project author.')
    return not missing_required


def check_kernels_actually_run(torch, props):
    """Launch real kernels. Comparing the GPU against torch.cuda.get_arch_list()
    is NOT sufficient and it was wrong to imply otherwise: that list describes
    PyTorch only. mmcv._ext and the selective_scan extensions are compiled
    separately, and a CUDA extension imports cleanly and then dies at the first
    kernel launch -- which is exactly the failure this test exists to catch.
    The only way to know is to launch one.
    """
    if torch is None or props is None:
        report(INFO, 'Kernel launch test skipped', 'Needs a working GPU.')
        return
    try:
        x = torch.randn(1, 3, 64, 64, device='cuda')
        torch.nn.Conv2d(3, 8, 3).cuda()(x)
        torch.cuda.synchronize()
        report(PASS, 'PyTorch can actually run on this GPU',
               'A real convolution executed successfully.')
    except Exception as exc:                                  # noqa: BLE001
        report(FAIL, 'PyTorch FAILS to run on this GPU', str(exc),
               'The software was built for a different GPU generation.\n'
               'The container image must be rebuilt for this machine.\n'
               'Send this whole message to the project author.')
        return
    try:
        import torch as _t
        from mmcv.ops import nms
        boxes = _t.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]],
                          device='cuda')
        nms(boxes, _t.tensor([0.9, 0.8], device='cuda'), 0.5)
        _t.cuda.synchronize()
        report(PASS, 'Detection operations run on this GPU',
               'mmcv GPU operations executed successfully.')
    except Exception as exc:                                  # noqa: BLE001
        report(FAIL, 'Detection operations FAIL on this GPU', str(exc),
               'mmcv was compiled for a different GPU generation. Models\n'
               'will load and then fail once they start working.\n'
               'The container image must be rebuilt. Send this to the author.')
    report(INFO, 'Note on what this proves',
           'PyTorch and mmcv kernels were launched and worked. The\n'
           'Mamba-family kernels cannot be tested this way and are only\n'
           'exercised once a real model runs -- so also complete the sample\n'
           'run in the documentation before trusting a large job.')


def check_extension_cubins(props):
    """Inspect what GPU code each compiled extension ACTUALLY contains.

    This is the definitive check, and the reason the earlier version of this
    file was wrong. torch.cuda.get_arch_list() describes PyTorch; every
    compiled extension is built separately and can target something else
    entirely. Measured on the original machine, mmcv carried eleven
    architectures up to sm_86 while dwconv2d carried sm_75 alone -- so the
    build looked healthy and one file would still have failed on any GPU
    newer than Turing.

    CUDA's rule: a cubin built for X.y runs on a device X.z when z >= y, and
    never across a major version. Absent PTX -- and none of these extensions
    carried any -- there is no JIT fallback to rescue a mismatch.
    """
    exts = []
    for pat in ('selective_scan', 'dwconv', 'mamba_ssm', 'causal_conv1d',
                '_ext'):
        for root in sys.path:
            if not root or not Path(root).is_dir():
                continue
            exts += [p for p in Path(root).rglob(f'*{pat}*.so')]
    exts = sorted(set(exts))
    if not exts:
        report(INFO, 'No compiled extensions found to inspect')
        return
    if shutil.which('cuobjdump') is None:
        report(INFO, 'Cannot inspect compiled extensions',
               'cuobjdump is not installed, so the GPU code inside each\n'
               'extension cannot be listed. This check is skipped.')
        return
    dev_major, dev_minor = (props.major, props.minor) if props else (None, None)
    bad = []
    for so in exts:
        try:
            out = subprocess.run(['cuobjdump', '--list-elf', str(so)],
                                 capture_output=True, text=True, timeout=60)
            archs = sorted({int(a[3:]) for a in
                            __import__('re').findall(r'sm_(\d+)', out.stdout)})
            ptx = bool(__import__('re').findall(r'compute_\d+', subprocess.run(
                ['cuobjdump', '--list-ptx', str(so)], capture_output=True,
                text=True, timeout=60).stdout))
        except Exception:                                     # noqa: BLE001
            continue
        if not archs:
            continue
        line = (f'{so.name}: ' + ' '.join(f'sm_{a}' for a in archs)
                + (' +PTX' if ptx else ''))
        if dev_major is None:
            report(INFO, 'Extension GPU code', line)
            continue
        # Same major generation, and cubin minor <= device minor.
        ok = any(a // 10 == dev_major and a % 10 <= dev_minor for a in archs)
        if ok or ptx:
            report(PASS, f'{so.name} supports this GPU', line)
        else:
            bad.append(line)
    if bad:
        report(FAIL, f'{len(bad)} compiled extension(s) CANNOT run on this GPU',
               '\n'.join(bad) + f'\n\nThis GPU is sm_{dev_major}{dev_minor}. '
               f'None of the code inside those\nfiles targets it, and they '
               f'carry no forward-compatible PTX.',
               'These specific files must be rebuilt for this GPU. Everything\n'
               'else is fine. Send this list to the project author -- it names\n'
               'exactly which ones, so the fix is small.')


def check_mmdet():
    # These are imported at module scope by palm_inference_pipeline.py, so a
    # machine missing any of them passes a framework-only check and then dies
    # on the first line of the pipeline. That is the most likely real failure.
    for mod in ('mmengine', 'mmcv', 'mmdet', 'rasterio', 'geopandas',
                'shapely', 'cv2', 'pyproj', 'pycocotools'):
        try:
            m = __import__(mod)
            report(PASS, f'{mod} {getattr(m, "__version__", "?")}')
        except Exception as exc:                              # noqa: BLE001
            report(FAIL, f'{mod} is not installed', str(exc),
                   'The environment is incomplete or you are not inside the\n'
                   'container. Rebuild from docker/Dockerfile.reconstructed.')
            return False
    return True


def check_checkpoint(path):
    if not path:
        report(INFO, 'No checkpoint given to test',
               'Re-run with --checkpoint <file.pth> to test loading one.')
        return
    p = Path(path)
    if not p.exists():
        report(FAIL, 'Checkpoint file not found', str(p),
               'Check the path. On Windows, a folder name containing spaces\n'
               'must be wrapped in double quotes.')
        return
    if p.is_dir():
        report(FAIL, 'That is a folder, not a model file', str(p),
               'Give the path to a single file ending in .pth, not the\n'
               'folder that contains it. Look inside that folder for a\n'
               'file whose name ends in .pth.')
        return
    size = p.stat().st_size
    if size < 1_000_000:
        report(FAIL, 'Checkpoint file is too small to be valid',
               f'{p} is only {size:,} bytes.',
               'The file did not copy completely. Copy it again and compare\n'
               'the size against the documentation.')
        return
    try:
        import torch
        torch.load(str(p), map_location='cpu', weights_only=False)
        report(PASS, 'Checkpoint loads correctly',
               f'{p.name}, {size / 1e6:.0f} MB')
    except Exception as exc:                                  # noqa: BLE001
        report(FAIL, 'Checkpoint file is damaged', f'{p}\n{exc}',
               'The file is corrupted, probably from an incomplete copy.\n'
               'Copy it again from the original drive.')


def check_data(path):
    if not path:
        report(INFO, 'No data folder given to test',
               'Re-run with --data <folder> to check your imagery layout.')
        return
    d = Path(path)
    if not d.exists():
        report(FAIL, 'Data folder not found', str(d),
               'Check the path. Inside the container you must use the path\n'
               'the folder was mounted AT, not its path on the host.\n'
               'Example: if you started the container with\n'
               '  -v D:\\my_images:/data\n'  # noqa: leakscan (illustrative example path)
               'then use /data here, not D:\\my_images.')  # noqa: leakscan (illustrative example path)
        return
    imgs = [p for p in d.rglob('*') if p.suffix.lower() in IMAGE_EXTS]
    if not imgs:
        report(FAIL, 'No images found in the data folder',
               f'{d} contains no files ending in '
               f'{", ".join(sorted(IMAGE_EXTS))}',
               'Check that you pointed at the right folder, and that the\n'
               'images are GeoTIFF files. JPEG and PNG are not usable:\n'
               'they carry no map coordinates.')
        return
    folders = {p.parent for p in imgs}
    report(PASS, f'{len(imgs)} image file(s) found',
           f'in {len(folders)} folder(s) under {d}')
    try:
        import rasterio
    except ImportError:
        report(FAIL, 'The imagery library (rasterio) is not installed',
               'Your images cannot be checked, and the pipeline cannot run.',
               'This is a software problem, NOT a problem with your images.\n'
               'You are outside the container, or the environment was not\n'
               'built correctly. Rebuild from docker/Dockerfile.reconstructed.')
        return
    unreadable = []
    for p in imgs[:5]:
        try:
            with rasterio.open(p) as src:
                if src.crs is None:
                    unreadable.append(f'{p.name}: has no map coordinates')
        except Exception as exc:                              # noqa: BLE001
            unreadable.append(f'{p.name}: {exc}')
    if unreadable:
        report(FAIL, 'Some images cannot be used', '\n'.join(unreadable),
               'Every image must be a GeoTIFF carrying map coordinates.\n'
               'An image without them cannot be turned into a map, because\n'
               'nothing says where on Earth it is.')
    else:
        report(PASS, 'Sample images are readable and carry map coordinates')


def check_disk(path='/workspace'):
    p = Path(path)
    if not p.exists():
        report(INFO, f'Disk space check skipped',
               f'{path} does not exist on this machine, so free space was '
               f'not checked.\nPass --workdir <folder> to check the drive '
               f'you will actually write to.')
        return
    free = shutil.disk_usage(p).free
    if free < 20e9:
        report(WARN, f'Low free disk space on {path}',
               f'{free / 1e9:.1f} GB free.',
               'Results and temporary files need room. Free some space or\n'
               'write results to a different, larger drive.')
    else:
        report(PASS, f'Disk space on {path}', f'{free / 1e9:.1f} GB free')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=None,
                    help='folder of imagery to check (optional)')
    ap.add_argument('--checkpoint', default=None,
                    help='model file to test loading (optional)')
    ap.add_argument('--workdir', default='/workspace')
    args = ap.parse_args()

    print('=' * 68)
    print(' SELF-TEST — checking that this machine can run the project')
    print('=' * 68)
    print(f' machine : {platform.node()}  ({platform.system()})')
    print(f' time    : {__import__("datetime").datetime.now():%Y-%m-%d %H:%M}')
    print('=' * 68)
    print()

    check_python()
    check_in_container()
    torch = check_torch()
    props = check_gpu(torch)
    check_cuda_arch(torch, props)
    check_mmdet()
    check_extensions()
    check_kernels_actually_run(torch, props)
    check_extension_cubins(props)
    check_checkpoint(args.checkpoint)
    check_data(args.data)
    check_disk(args.workdir)

    fails = [r for r in RESULTS if r[0] == FAIL]
    warns = [r for r in RESULTS if r[0] == WARN]
    print('=' * 68)
    print(f' {len(RESULTS)} checks: {len(RESULTS) - len(fails) - len(warns)} '
          f'passed, {len(warns)} warning(s), {len(fails)} failure(s)')
    print('=' * 68)
    if fails:
        print('\n THIS MACHINE IS NOT READY. Fix these first:\n')
        for i, (_, title, _, fix) in enumerate(fails, 1):
            print(f'  {i}. {title}')
            for line in (fix or 'See the message above.').splitlines():
                print(f'       {line}')
            print()
        print(' If you cannot resolve these, send this entire output to the')
        print(' project author. It contains everything needed to diagnose it.')
    elif warns:
        print('\n Ready to use, with notes above. Nothing here stops a run;')
        print(' the warnings describe things that may be slower or need a')
        print(' setting changed if a job fails.')
    else:
        print('\n Everything passed. This machine is ready.')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
