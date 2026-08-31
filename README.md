# Date-palm instance segmentation with state-space backbones

Code for the experiments reported in the accompanying manuscript: a
benchmark of CNN, transformer and state-space (Mamba-family) backbones for
date-palm crown instance segmentation in very-high-resolution imagery, and
the country-scale inventory built from the best of them.

This repository is a **fork of [MMDetection](https://github.com/open-mmlab/mmdetection) 3.3.0**
with project-specific configs, backbone wrappers and tooling added.

---

## What is here

| path | what it is |
|---|---|
| `configs/Custom/_base_palm/` | shared dataset, schedule and model bases |
| `configs/Custom/maskrcnn_palm_stagec/` | the 11-backbone benchmark |
| `configs/Custom/maskrcnn_palm_staged/` | the transfer / budget matrix |
| `configs/Custom/maskrcnn_palm_finetune_hn/` | hard-negative adaptation and the deployed config |
| `configs/Custom/Finetune_HN/` | hard-negative mining, threshold calibration, evaluation |
| `configs/Custom/Evaluation/`, `Feature_Analysis/` | the analyses behind the reported numbers |
| `configs/Custom/utils/` | dataset construction, inference pipeline, environment checks |
| `mmdet/models/backbones/` | wrappers for the SSM backbones |
| `palm_inference/` | the tiled, resumable, georeferenced inference pipeline |
| `weights.yaml` | every pretrained weight, by official source and SHA256 |
| `THIRD_PARTY.md` | upstream projects and their licences |

## What is NOT here, and why

**No imagery and no annotations.** The data is licensed to the project and
cannot be redistributed.

**No trained checkpoints.** Every config needed to retrain them is here.

**No pretrained weights.** `weights.yaml` records each one by its official
source and its SHA256, so the identical file can be obtained from its
author and verified. Several were renamed locally during the work —
**match them by hash, not by filename.**

**Not the Google Earth acquisition tooling.** Withheld at the authors'
discretion; the imagery it retrieves is subject to the provider's terms.
Nothing in the modelling code depends on how imagery was obtained — the
experiments reproduce from any imagery of comparable resolution.

See `WITHHELD.md` for the complete list.

---

## The backbone wrappers contain no architecture code

`mmdet/models/backbones/*_backbone.py` adapt upstream implementations to
MMDetection's registry. The architectures themselves live in the original
projects and are **expected at fixed paths**:

```
/opt/vmamba  /opt/spatial_mamba  /opt/groupmamba  /opt/efficientvmamba
```

Those paths are hard-coded and the directory names are load-bearing.
`THIRD_PARTY.md` gives each project's repository and the exact commit this
work used.

---

## Reproducing the environment

This is the part that is genuinely difficult, so it is stated plainly.

Three dependencies compile CUDA extensions against a specific
torch/CUDA pair: **`mmcv 2.1.0`** (source-only on PyPI), **`mamba-ssm
2.2.4`** and **`causal-conv1d 1.4.0`**, plus the SSM projects' own
`selective_scan` and `dwconv2d` kernels. A version mismatch anywhere in
that chain fails at import or, worse, at the first GPU kernel launch.

`docker/Dockerfile.reconstructed` is the recipe, pinned to the exact
commits used. **Order matters** — torch is installed before mmcv, because
mmcv compiles against whatever torch is present.

The environment was:

| | |
|---|---|
| base | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| python / torch | 3.10.12 / 2.1.0+cu121 |
| mmengine / mmcv / mmdet | 0.10.1 / 2.1.0 / 3.3.0 |
| mmpretrain | 1.2.0 |
| mamba-ssm / causal-conv1d | 2.2.4 / 1.4.0 |
| GPU used | NVIDIA TITAN RTX (sm_75), 24 GB |

Two scripts check an installation, and you want both:

```bash
python configs/Custom/utils/handover_selftest.py     # does it import?
python configs/Custom/utils/smoke_build_models.py    # do the models RUN?
```

The first proves imports. The second builds every model and pushes a
tensor through it — a backbone can import cleanly and still fail on its
first forward pass, because importing touches Python and a kernel launch
touches the GPU.

**A note on GPU architecture.** The `dwconv2d` kernel as originally built
carried `sm_75` only. Rebuild the kernels with a wider
`TORCH_CUDA_ARCH_LIST` (the Dockerfile does this) or newer cards fail with
`no kernel image is available for execution on the device`.

---

## Running inference

```bash
python -m palm_inference.run_inference \
  --input-root /path/to/geotiffs \
  --output-root /path/to/output \
  --config-file configs/Custom/maskrcnn_palm_finetune_hn/maskrcnn_spatialmamba_s_deploy.py \
  --checkpoint /path/to/checkpoint.pth \
  --tile-size 1024 --overlap 256 \
  --score-thr 0.30 --postprocess
```

**The command-line defaults are not the deployment settings.** They
default to 512 / 128 / 0.35; the deployment used **1024 / 256 / 0.30**.
A run that omits them succeeds and produces a plausible map that is not
the reported configuration. Always pass all three, and always
`--postprocess` — without it, palms straddling tile boundaries are counted
twice.

Imagery must be georeferenced GeoTIFF at roughly 15 cm/px. At 1 m/px a
crown is a few pixels across and will not be detected; that is a property
of the data, not a setting.

---

## Building datasets

`configs/Custom/utils/TILING_README.md` documents the full pipeline:
mosaic + reference polygons → 512 px tiles + LabelMe JSON → COCO.

Annotation format: [LabelMe](https://github.com/wkentaro/labelme).

Three policies materially affect results and are explained there: empty
tiles belong in train and never in val/test; background is capped at 30%
with a fixed seed; and `filter_empty_gt`'s only symptom is a lower image
count in the log.

---

## Citation

If you use this code, please cite the manuscript (details to follow on
publication) and the upstream projects listed in `THIRD_PARTY.md`.

MMDetection itself:

```bibtex
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and others},
  journal = {arXiv preprint arXiv:1906.07155},
  year    = {2019}
}
```

## Licence

Apache-2.0, inherited from MMDetection (see `LICENSE`). Upstream backbone
projects carry their own licences — see `THIRD_PARTY.md`, and note in
particular that **MambaVision is NVIDIA non-commercial**.
