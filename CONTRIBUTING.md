# Contributing

This repository accompanies a manuscript. It is published for
reproducibility rather than as an actively developed tool, so the most
useful contributions are corrections.

## Reporting a problem

Open an issue with:

1. The full command you ran
2. The exact error text, not a paraphrase
3. Output of `python configs/Custom/utils/handover_selftest.py`
4. Output of `nvidia-smi`

Those four identify almost any problem here without further exchange.

## Before opening an issue about installation

Most installation problems are one of three things, all covered in the
README:

- torch installed after mmcv. mmcv 2.1.0 compiles against whatever torch
  it finds, so the installation order in the README is required.
- A GPU-architecture mismatch. `no kernel image is available for
  execution on the device` means the compiled kernels were built for a
  different card. Rebuild with a wider `TORCH_CUDA_ARCH_LIST`.
- Missing `/opt` trees. The backbone wrappers contain no architecture
  code and expect the upstream projects at fixed paths.

Run both checks before reporting:

```bash
python configs/Custom/utils/handover_selftest.py   # does it import?
python configs/Custom/utils/smoke_build_models.py  # do the models run?
```

## Pull requests

CI runs `python tools/validate_repo.py`, which checks that every config's
`_base_` chain resolves, that documentation links and referenced paths
exist, that `custom_imports` modules are published, that no user home
paths or usernames have crept in, and that no data artefacts are
committed. It needs only the standard library, so run it locally first.

Please do not commit checkpoints, imagery or annotations. `.gitignore`
covers the usual extensions, and CI fails if one appears anyway.
