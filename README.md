# Date-palm instance segmentation across sensors and scales

[![validate](https://github.com/brakuta/Datepalm-Instance-Segmentation/actions/workflows/validate.yml/badge.svg)](https://github.com/brakuta/Datepalm-Instance-Segmentation/actions/workflows/validate.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Code for benchmarking Mask R-CNN with eleven interchangeable backbones
(CNN, transformer and state-space/Mamba families) on date-palm crown
instance segmentation, from 5 cm UAV imagery to 30 cm WorldView-3
satellite imagery, and for the hard-negative adaptation used to run the
best model as a country-scale inventory. The repository is a fork of
[MMDetection](https://github.com/open-mmlab/mmdetection) 3.3.0: it adds
project configs, backbone wrappers and tooling, and vendors the MMDetection
train/test entry points unchanged.

The imagery, annotations and trained checkpoints are not distributed
(see [What is not included](#9-what-is-not-included)). Everything needed
to rebuild the environment, prepare data in the expected format, retrain
every model and reproduce the evaluation is included.

## Contents

1. [Quick start](#1-quick-start)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Data layout](#4-data-layout)
5. [Data preparation](#5-data-preparation)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Inference with the deployed model](#8-inference-with-the-deployed-model)
9. [What is not included](#9-what-is-not-included)
10. [Repository checks](#10-repository-checks)
11. [Citation and licence](#11-citation-and-licence)

## 1. Quick start

```bash
# 1. clone
git clone https://github.com/brakuta/Datepalm-Instance-Segmentation.git
cd Datepalm-Instance-Segmentation

# 2. build the environment (1-2 hours; kernel compilation dominates)
docker build -f docker/Dockerfile.reconstructed -t mamba-mmdet:rebuilt .

# 3. start the container with your data mounted
docker run --gpus all -it --shm-size=16g \
    -v /path/to/datasets:/workspace/datasets \
    -v /path/to/checkpoints:/workspace/mmdetection/checkpoints \
    mamba-mmdet:rebuilt

# 4. verify the installation (inside the container)
python configs/Custom/utils/handover_selftest.py     # imports and versions
python configs/Custom/utils/smoke_build_models.py    # builds every model, runs a forward pass

# 5. train a model (see section 6 for the full config list)
python tools/train.py configs/Custom/1_single_sensor_uav_5cm/maskrcnn_r50_uav5cm.py

# 6. test a trained model
python tools/test.py configs/Custom/1_single_sensor_uav_5cm/maskrcnn_r50_uav5cm.py \
    work_dirs/maskrcnn_r50_uav5cm/best_coco_segm_mAP_50_iter_XXXX.pth
```

Steps 5 and 6 need the datasets in place first (sections 4 and 5) and the
pretrained backbone weights listed in `weights.yaml`. Step 4 works without
data and should be run before anything else.

## 2. Repository layout

```
configs/Custom/
  1_single_sensor_uav_5cm/    experiment 1: backbone benchmark on UAV 5 cm
  2_pooled_15cm_ge_aerial/    experiment 2: Google Earth + aerial pooled at 15 cm
  3_unified_multisource/      experiment 3: one model on all three sources
  4_satellite_wv3_30cm/       experiment 4: WorldView-3 30 cm transfer
  5_deployment_finetune/      experiment 5: hard-negative adaptation, deployed model

  _base_palm/                 shared dataset, schedule, hook and sampler definitions,
                              inherited by every experiment config
  utils/                      dataset building, installation checks, inference helpers
  Evaluation/                 metrics engine, per-model evaluation, result compilation
  Feature_Analysis/           representation-level analysis (CKA, Bures, separability)
  Finetune_HN/                hard-negative mining and threshold calibration
  tools_staged/               experiment 4 tooling (budget manifests, stem inflation)

mmdet/models/backbones/       backbone wrappers; contain no architecture code
palm_inference/               tiled, resumable, georeferenced inference pipeline
docker/                       Dockerfile.reconstructed, the environment recipe
tools/                        train.py, test.py (vendored from MMDetection),
                              install_backbones.py, validate_repo.py
.github/workflows/            CI

README.md          this page
RESULTS.md         how to regenerate the result tables
THIRD_PARTY.md     upstream projects, pinned commits and licences
WITHHELD.md        files deliberately not published, and why
weights.yaml       every model weight by source and SHA256
requirements.txt   pinned versions; read the ordering note inside
CITATION.cff  CONTRIBUTING.md  LICENSE
```

Every folder has its own README with details specific to that part.

Each experiment folder corresponds to one part of the study. The
manuscript and some historical documents refer to them by internal stage
names:

| folder | internal name |
|---|---|
| `1_single_sensor_uav_5cm` | `maskrcnn_palm` (Stage A) |
| `2_pooled_15cm_ge_aerial` | `maskrcnn_palm_ms15` (Stage B) |
| `3_unified_multisource` | `maskrcnn_palm_stagec` (Stage C) |
| `4_satellite_wv3_30cm` | `maskrcnn_palm_staged` (Stage D) |
| `5_deployment_finetune` | `maskrcnn_palm_finetune_hn` |

## 3. Installation

### 3.1 Requirements

| component | version |
|---|---|
| GPU | CUDA-capable, 24 GB VRAM used in this work (TITAN RTX, sm_75) |
| base image | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| Python / PyTorch | 3.10.12 / 2.1.0+cu121 |
| mmengine / mmcv / mmdet / mmpretrain | 0.10.1 / 2.1.0 / 3.3.0 / 1.2.0 |
| mamba-ssm / causal-conv1d | 2.2.4 / 1.4.0 |

Three dependencies compile CUDA extensions against a specific torch/CUDA
pair: `mmcv 2.1.0` (source-only on PyPI), `mamba-ssm 2.2.4` and
`causal-conv1d 1.4.0`, plus the `selective_scan` and `dwconv2d` kernels
from the SSM projects themselves. A version mismatch fails at import, or
at the first GPU kernel launch.

### 3.2 Docker build (recommended)

```bash
docker build -f docker/Dockerfile.reconstructed -t mamba-mmdet:rebuilt .
```

[`docker/Dockerfile.reconstructed`](docker/Dockerfile.reconstructed) is
pinned to the exact commits used in this work and performs all of the
steps below in the right order, including cloning the upstream backbone
repositories to `/opt` and compiling their kernels. The `docker run`
command with the expected mount points is documented at the end of the
file and in the quick start above.

### 3.3 Manual installation

Follow this order. It matters, because mmcv compiles against whatever
torch it finds:

```bash
# 1. torch first, from the cu121 index
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. the OpenMMLab stack and pure-python dependencies
pip install -r requirements.txt

# 3. the SSM kernels, from git at a tag, against the installed torch
git clone https://github.com/Dao-AILab/causal-conv1d.git && cd causal-conv1d \
    && git checkout v1.4.0 && pip install --no-build-isolation . && cd ..
git clone https://github.com/state-spaces/mamba.git && cd mamba \
    && git checkout v2.2.4 && pip install --no-build-isolation . && cd ..
pip install mambavision

# 4. the upstream backbone repositories at their pinned commits
#    (paths are hard-coded in the wrappers; the directory names matter)
#    /opt/vmamba /opt/spatial_mamba /opt/groupmamba /opt/efficientvmamba
#    See THIRD_PARTY.md for each repository and commit, and
#    docker/Dockerfile.reconstructed for the kernel build commands.

# 5. copy the backbone wrappers into the installed mmdet package
python tools/install_backbones.py
```

Step 5 is required: the configs import the wrappers as
`mmdet.models.backbones.*`, so the wrapper files must sit inside the
installed mmdet package. A plain `pip install mmdet` does not know about
them, and without this step every Mamba-family config fails to load.
The Docker build performs this step automatically.

`tools/train.py` and `tools/test.py` are vendored unchanged from
MMDetection 3.3.0, so training commands run from the repository root
without a separate MMDetection checkout.

### 3.4 Verifying the installation

```bash
python configs/Custom/utils/handover_selftest.py     # imports, versions, GPU visibility
python configs/Custom/utils/smoke_build_models.py    # builds each model, one forward pass
```

Run both. The first proves the imports; the second pushes a tensor
through every model, which is what catches a kernel compiled for the
wrong GPU architecture. If you see `no kernel image is available for
execution on the device`, rebuild the kernels with a wider
`TORCH_CUDA_ARCH_LIST` (the Dockerfile sets
`7.5;8.0;8.6;8.9;9.0+PTX`).

### 3.5 Pretrained backbone weights

Training starts from ImageNet checkpoints. [`weights.yaml`](weights.yaml)
records every file: its official source, its SHA256 where captured, and
whether it is an unchanged upstream file or one derived locally (for
example, classifier heads stripped, or a 3-channel stem widened to 8
bands). Derived files can be regenerated with
`configs/Custom/utils/strip_backbone_checkpoint.py` and
`configs/Custom/tools_staged/inflate_stem_to_nband.py`. Several files
were renamed during the work, so match them by hash rather than filename.
MambaVision and MambaOut weights are fetched from HuggingFace when the
model is first built and need network access at that moment.

## 4. Data layout

All datasets are COCO-format instance segmentation sets with a single
class, `DatePalm`. The dataset configs in `configs/Custom/_base_palm/`
expect the following trees under one root (the original work used
`/workspace/datasets/COCO/`; either mount your data there or edit
`data_root` in the dataset configs and keep
`configs/Custom/Evaluation/sensor_registry.py` consistent with it):

```
<COCO root>/
  UAV_5cm/                          # experiment 1; also pooled into experiment 3
    train_UAV/  val_UAV/  test_UAV/           # 1024 x 1024 tiles
    Annotations/train_UAV.json  val_UAV.json  test_UAV.json
  GE_15cm/                          # experiments 2, 3, 5
    train_GE/  val_GE/  test_GE/              # 512 x 512 tiles
    Annotations/train_GE.json  val_GE.json  test_GE.json
  Aerial_15cm/                      # experiments 2, 3
    train_aerial/  val_aerial/  test_aerial/
    Annotations/train_aerial.json  val_aerial.json  test_aerial.json
  Sat_30cm/                         # experiment 4 (real WorldView-3)
    train_sat/  val_sat/  test_sat/
    Annotations/train_sat.json  val_sat.json  test_sat.json
  GE_30sim/                         # experiment 4 (simulated 30 cm pre-training)
    train/  val/  test/
    Annotations/GE_30sim_train.json  GE_30sim_val.json  GE_30sim_test.json
```

Each dataset config header states the tile counts and provenance of its
corpus. Before training, check three paths in your copies of the configs:
`data_root` in the dataset files, `pretrained=` in the per-backbone
configs (where the ImageNet weights sit), and `work_dir` (where runs
write; experiments 1 and 2 use relative `./work_dirs/`, the others ship
with the original machine's absolute paths).

## 5. Data preparation

The tiling pipeline that produced these datasets is documented in
[`configs/Custom/utils/TILING_README.md`](configs/Custom/utils/TILING_README.md)
and runs in three steps:

1. Orthomosaic plus reference polygons in, image tiles plus
   [LabelMe](https://github.com/wkentaro/labelme) JSON out
   (`image_vector_to_labelme_pipeline.py`). One job file per corpus;
   tile size, overlap and band selection are derived from each mosaic's
   ground sample distance. `jobs_example.json` is a template.
2. LabelMe to COCO conversion (`labelme2coco_palm.py`).
3. Verification of counts and geometry against the source polygons.

Three dataset-construction policies affect results and are explained in
that document: empty (palm-free) tiles are allowed in training sets only,
never in validation or test; background tiles are capped at 30% of a
training set with a fixed seed; and `filter_empty_gt=True` in a config
silently drops empty tiles, so the image count in the training log is the
place to notice it.

For experiment 4, `configs/Custom/tools_staged/` holds the extra
preparation steps: building the simulated 30 cm pre-training corpus,
nested annotation-budget manifests (`build_budget_manifests.py`), and
widening pretrained stems for the 8-band multispectral runs
(`inflate_stem_to_nband.py`).

## 6. Training

Each experiment folder holds one config per backbone. The backbones
compared are ResNet-50/101 and ConvNeXt-T (CNN); Swin-T/S and PVTv2-B2
(transformer); VMamba, Spatial-Mamba, GroupMamba, EfficientVMamba and
MambaVision (state-space); and MambaOut, which removes the state-space
component and is included as a control, not as a Mamba model. Training is
always:

```bash
python tools/train.py <config> [--work-dir <dir>]
```

### Experiment 1: single-sensor benchmark, UAV 5 cm (18 configs)

The controlled backbone comparison: same data, same schedule, same
detector, only the backbone changes. Both size variants of most families.

```
maskrcnn_{r50,r101,swin_t,swin_s,convnext_t,pvtv2_b2}_uav5cm.py
maskrcnn_{vmamba,spatialmamba,groupmamba,mambaout,mambavision}_{t,s}_uav5cm.py
maskrcnn_efficientvmamba_{s,b}_uav5cm.py      # EfficientVMamba sizes are S and B
```

### Experiment 2: pooled 15 cm, Google Earth + aerial (10 configs)

Trains on both 15 cm sources together and validates on Google Earth only.
The aerial test split is held out and scored afterwards through an
evaluation-only config, so the pooling question is not answered by the
validation set that selected the checkpoints.

### Experiment 3: unified multi-source model (11 configs)

UAV 5 cm, Google Earth 15 cm and aerial 15 cm pooled into one training
set, with source-local batch construction (each batch is drawn from one
source). The deployed model comes from this comparison.

### Experiment 4: WorldView-3 30 cm transfer (24 configs)

Four config families, named by suffix:

| suffix | purpose |
|---|---|
| `_ge30sim_stage1` | pre-training on simulated 30 cm imagery (GE 15 cm downsampled with PSF blur and sensor noise) |
| `_staged_ft` | fine-tuning on real WorldView-3 across a nested annotation-budget ladder |
| `_staged_full` | the full-budget reference point |
| `_staged_ms` | 8-band multispectral WorldView-3 instead of RGB |

The budget ladder is nested and seeded, so each smaller subset is
contained in the next larger one. Read
[`STAGE_D_README.md`](configs/Custom/4_satellite_wv3_30cm/STAGE_D_README.md)
in that folder before running these; `tools_staged/run_staged_matrix.sh`
drives the full matrix.

### Experiment 5: deployment and hard-negative adaptation (4 configs)

Adapts the experiment 3 model against the false positives found in
country-scale operation (palm-like shrubs, ghaf, acacia), with the
original positive data replayed alongside the negatives so recall is
preserved. `maskrcnn_spatialmamba_s_deploy.py` is the deployed
configuration. This is an operational adaptation; the benchmark
checkpoints and their reported numbers are unaffected. Supporting tools
(hard-negative tile mining, threshold recalibration, false-positive
evaluation) are in `configs/Custom/Finetune_HN/`.

Two practical notes that apply across experiments. Seeds are not fixed in
the configs; each run draws a seed and writes it only to its own log, so
reproducing a specific run needs that log. Training budgets differ
between experiments; check the schedule a config inherits before
comparing numbers across them.

## 7. Evaluation

Per-model evaluation and table compilation live in
[`configs/Custom/Evaluation/`](configs/Custom/Evaluation/), with their
own README. The usual sequence:

```bash
# score one trained model on the sensors it applies to
python configs/Custom/Evaluation/evaluate_model.py \
    --config <config> --checkpoint <ckpt> --sensors UAV GE Aerial

# compile per-experiment tables from the run logs
python configs/Custom/Evaluation/compile_results.py --results-dir <dir>

# cross-sensor transfer matrix
python configs/Custom/Evaluation/build_manifest.py ...
python configs/Custom/Evaluation/run_cross_transfer.py --manifest <manifest>
python configs/Custom/Evaluation/compile_cross_transfer.py --manifest <manifest>
```

`sensor_registry.py` in that folder is the single source of truth for
sensor names, annotation paths and evaluation protocols; keep it
consistent with your `data_root`. [`RESULTS.md`](RESULTS.md) explains how
the manuscript tables are regenerated and what to check before comparing
numbers across experiments. Result tables themselves will be added there
on publication.

## 8. Inference with the deployed model

`palm_inference/` is a tiled, resumable, georeferenced inference
pipeline: it tiles large GeoTIFFs, runs batched FP16 inference, merges
and de-duplicates detections across tile boundaries, and writes
GeoPackage. An interrupted run resumes from its manifest.

```bash
python -m palm_inference.run_inference \
  --input-root /path/to/geotiffs \
  --output-root /path/to/output \
  --config-file configs/Custom/5_deployment_finetune/maskrcnn_spatialmamba_s_deploy.py \
  --checkpoint /path/to/checkpoint.pth \
  --tile-size 1024 --overlap 256 \
  --score-thr 0.30 --postprocess
```

Three settings to check before trusting an output map:

1. The CLI defaults (tile 512, overlap 128, threshold 0.35) are not the
   deployment settings. Deployment used 1024 / 256 / 0.30, as above.
   Omitting the flags succeeds and produces a different map.
2. Without `--postprocess`, palms straddling tile boundaries are counted
   twice.
3. The model was trained at roughly 15 cm/px. At 1 m/px a crown is a few
   pixels across and will not be detected. Resolution matters more than
   any flag.

## 9. What is not included

- **Imagery, annotations and trained checkpoints.** The imagery is
  licensed to the project and cannot be redistributed. Every config
  needed to retrain is here.
- **Pretrained weights.** Recorded in [`weights.yaml`](weights.yaml) by
  source and hash instead of being redistributed; see section 3.5.
- **The Google Earth acquisition tooling.** The imagery it retrieves is
  subject to the provider's terms. Nothing in the modelling code depends
  on how imagery was obtained. External very-high-resolution basemap
  imagery was used for testing only, never for training.

The full list, including internal handover files that other documents
mention, is in [`WITHHELD.md`](WITHHELD.md).

## 10. Repository checks

CI runs on every push, needs no GPU, torch or network, and finishes in
seconds. The same check runs locally:

```bash
python tools/validate_repo.py
```

It verifies that every config's `_base_` chain resolves, that paths named
in documentation, shell scripts and docstrings exist, that
`custom_imports` modules are published, that no private absolute paths or
data artefacts have been committed, and that markdown links are intact.
Run it before opening a pull request; `CONTRIBUTING.md` has the rest.

## 11. Citation and licence

Until the manuscript is published, please cite this repository (see
`CITATION.cff`), along with MMDetection and the upstream backbone
projects listed in [`THIRD_PARTY.md`](THIRD_PARTY.md):

```bibtex
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and others},
  journal = {arXiv preprint arXiv:1906.07155},
  year    = {2019}
}
```

The repository is Apache-2.0, inherited from MMDetection
([`LICENSE`](LICENSE)). Upstream backbones carry their own terms; note in
particular that MambaVision is released by NVIDIA under a non-commercial
licence, which applies to MambaVision only. Wrappers in
`mmdet/models/backbones/` adapt the upstream implementations, which are
expected as clones under `/opt/`; `THIRD_PARTY.md` records each repository
and the exact commit used.
