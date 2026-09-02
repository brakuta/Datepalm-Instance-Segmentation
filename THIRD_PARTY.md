# Third-party projects and their licences

Every upstream dependency whose code or weights this work uses, with the
exact commit used and the licence as read from that commit.

Licences below were read from each project's own `LICENSE` file in the
clone this project actually used, not from a project page that may have
changed since. Where none was found, that is stated rather than assumed.

Last verified: 2026-08-31.

---

## Framework

| project | version | licence |
|---|---|---|
| [MMDetection](https://github.com/open-mmlab/mmdetection) | 3.3.0 (this fork) | Apache-2.0 |
| [MMEngine](https://github.com/open-mmlab/mmengine) | 0.10.1 | Apache-2.0 |
| [MMCV](https://github.com/open-mmlab/mmcv) | 2.1.0 | Apache-2.0 |
| [MMPreTrain](https://github.com/open-mmlab/mmpretrain) | 1.2.0 | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | 2.1.0+cu121 | BSD-3-Clause |
| [timm](https://github.com/huggingface/pytorch-image-models) | 1.0.15 | Apache-2.0 |

## State-space kernels

| project | version | licence |
|---|---|---|
| [mamba](https://github.com/state-spaces/mamba) | v2.2.4 | Apache-2.0 |
| [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | v1.4.0 | Apache-2.0 |

## Backbone architectures

These are loaded from clones at fixed paths; the wrappers in this
repository contain no architecture code.

### VMamba (`/opt/vmamba`)
- <https://github.com/MzeroMiko/VMamba>
- commit `2ed52ead062a51a64521ed3871d52914bf532876`
- MIT, Copyright (c) 2024 MzeroMiko

### Spatial-Mamba (`/opt/spatial_mamba`)
- <https://github.com/EdwardChasel/Spatial-Mamba>
- commit `f72ed9b2486a5931190912dcfa8b964033c80e8c`
- Apache-2.0
- *Provides the backbone of the deployed model.*

### GroupMamba (`/opt/groupmamba`)
- <https://github.com/Amshaker/GroupMamba>
- commit `c1800ed262e995de3c645348e7e0202e8385f334`
- MIT, Copyright (c) 2024 Abdelrahman Shaker

### EfficientVMamba (`/opt/efficientvmamba`)
- <https://github.com/TerryPei/EfficientVMamba>
- commit `0bc5ee288b402d648641f5494b73e9d152e0c62b`
- No licence file found at that commit.

> The clone contains no `LICENSE` at commit `0bc5ee28`, while the other
> three projects all carry one. A repository published without a licence is
> all-rights-reserved by default: public visibility does not by itself
> grant permission to use, modify or redistribute.
>
> This is recorded as a fact about that commit, not as a criticism, and it
> does not affect the validity of experiments that used the code. Anyone
> building on this should check the project's current page (a licence may
> have been added since) and describe the terms as the authors state them.

### MambaVision (installed via `pip install mambavision`)
- Weights: `nvidia/MambaVision-S-1K` on HuggingFace, resolved when the
  model is built.
- **NVIDIA Source Code License — NON-COMMERCIAL.**

> **This restriction applies to MambaVision only.** It does not extend to
> the other backbones and must not be described as if it did. Weights are
> not redistributed here; obtain them from NVIDIA under that licence.

### MambaOut (via `timm`)
- Weights: `timm/mambaout_small.in1k`, resolved when the model is built.

> MambaOut is not a state-space model, despite the name. It is the
> ablation that *removes* the SSM, and belongs with the CNN / gated-conv
> baselines rather than the Mamba family.

### Also present in the environment, not used by any reported result
`/opt/vim` ([hustvl/Vim](https://github.com/hustvl/Vim), commit
`dd0358ad…`), `/opt/vssd`
([YuHengsss/VSSD](https://github.com/YuHengsss/VSSD), commit `c10de2f8…`),
`/opt/msvmamba`
([YuHengsss/MSVMamba](https://github.com/YuHengsss/MSVMamba), commit
`c3940191…`).

## CNN and transformer baselines

ResNet-50/101 come from torchvision (BSD-3-Clause) at run time. Swin,
ConvNeXt and PVTv2 start from standard MMPreTrain/timm ImageNet
checkpoints (Apache-2.0), which this project then stripped to
backbone-only form; see `weights.yaml`, which distinguishes upstream
files from derived ones.

## Data formats

[LabelMe](https://github.com/wkentaro/labelme) (annotation), and
[COCO](https://cocodataset.org) (training format).

---

## Weights are not redistributed

No pretrained or trained weights are published in this repository.
`weights.yaml` records each one by official source and, for the files
central to the reported results, by SHA256, which is what lets a
reader confirm they have the identical file. Several were renamed locally,
so the hash is the identity and the filename is not.
