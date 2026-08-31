# Date-palm instance segmentation across sensors and scales

Mask R-CNN with eleven interchangeable backbones — CNN, transformer and
state-space (Mamba-family) — trained and evaluated on date-palm crown
delineation in imagery from **5 cm UAV down to 30 cm satellite**, and the
adaptation that turned the best of them into a country-scale inventory.

A fork of [MMDetection](https://github.com/open-mmlab/mmdetection) 3.3.0
with project configs, backbone wrappers and tooling added.

---

## Start here: which experiment do you want?

The work is five experiments, each answering a different question. **Pick
the row you care about** — you do not need the others.

| | question | imagery | folder |
|---|---|---|---|
| **1** | Which backbone is best when the sensor is fixed? | UAV 5 cm | [`1_single_sensor_uav_5cm/`](configs/Custom/1_single_sensor_uav_5cm) |
| **2** | Does pooling two 15 cm sources help? | GE + aerial, 15 cm | [`2_pooled_15cm_ge_aerial/`](configs/Custom/2_pooled_15cm_ge_aerial) |
| **3** | One model for all sensors, or one per sensor? | UAV 5 cm + GE + aerial | [`3_unified_multisource/`](configs/Custom/3_unified_multisource) |
| **4** | How far does it carry to 30 cm, and at what labelling cost? | WorldView-3 30 cm | [`4_satellite_wv3_30cm/`](configs/Custom/4_satellite_wv3_30cm) |
| **5** | Surviving contact with a whole country | GE 15 cm, national | [`5_deployment_finetune/`](configs/Custom/5_deployment_finetune) |

**Every folder has its own README** giving the configs, which to run
first, and the design decisions that are easy to get wrong. Start there,
not here.

Dataset definitions live in
[`configs/Custom/_base_palm/`](configs/Custom/_base_palm): `dataset_uav_5cm`,
`dataset_MS15_pooled`, `dataset_UAV_GE_Aerial_pooled_C`,
`dataset_sat_30cm_staged`, `dataset_ge30sim`.

<details>
<summary>Mapping to the internal stage names</summary>

The manuscript and the authors' notes use stage letters. The folders were
renamed here because `stagec` and `staged` differ by one letter and mean
unrelated things, and `ms15` means nothing to a reader.

| public folder | internal |
|---|---|
| `1_single_sensor_uav_5cm` | `maskrcnn_palm` — Stage A |
| `2_pooled_15cm_ge_aerial` | `maskrcnn_palm_ms15` — Stage B |
| `3_unified_multisource` | `maskrcnn_palm_stagec` — Stage C |
| `4_satellite_wv3_30cm` | `maskrcnn_palm_staged` — Stage D |
| `5_deployment_finetune` | `maskrcnn_palm_finetune_hn` |

File contents are unchanged; only folder names differ, and cross-folder
`_base_` references were rewritten to match.

</details>

---

### 1 — Single-sensor benchmark (UAV 5 cm)

The controlled comparison: same data, same schedule, same detector, only
the backbone changes. Both size variants of most families, so scale is a
visible axis rather than a confound.

```
maskrcnn_{r50,r101,swin_t,swin_s,convnext_t,pvtv2_b2}_uav5cm.py
maskrcnn_{vmamba,spatialmamba,groupmamba,efficientvmamba,mambaout,mambavision}_{t,s}_uav5cm.py
```

### 2 — Pooled 15 cm (Google Earth + aerial)

Trains on Google Earth 15 cm and aerial 15 cm together, validating on GE
only. Aerial is held out and scored afterwards through a separate
eval-only config — so "does pooling help?" is answered without the
validation set choosing the answer.

### 3 — Unified multi-source model

UAV 5 cm, GE 15 cm and aerial 15 cm pooled, with **source-local batch
construction**: batches are drawn from one source at a time, for page-cache
locality across three mounts. Eleven backbones. This is the comparison the
deployed model came from.

### 4 — Satellite transfer (WorldView-3, 30 cm)

The hardest question here, and the one with the most machinery. Four
config families:

| suffix | what it is |
|---|---|
| `_ge30sim_stage1` | pre-train on **simulated** 30 cm — GE 15 cm downsampled with PSF blur and sensor noise, 19,472 tiles, crowns ~17 px, matching real WV-3 scale |
| `_staged_ft` | fine-tune on real WV-3 across an **annotation-budget** ladder |
| `_staged_full` | the full-budget reference point |
| `_staged_ms` | **multispectral** — 8-band WV-3 rather than RGB |

The budget ladder is nested and seeded (`tools_staged/build_budget_manifests.py`),
so the 5% subset is contained in the 10%, and "more labels help this much"
is a measurement rather than an artefact of which tiles were drawn.

Multispectral runs widen a 3-channel ImageNet stem to 8 channels
(`tools_staged/inflate_stem_to_nband.py`) so they still start pretrained.

Read [`STAGE_D_README.md`](configs/Custom/4_satellite_wv3_30cm/STAGE_D_README.md)
before running any of these.

### 5 — Deployment and hard-negative adaptation

A benchmark model applied to a whole country meets terrain the training
set never contained: palm-like shrubs, ghaf, acacia. `_finetune_hn`
adapts against exactly those false positives, with the original positives
**replayed alongside** so recall does not quietly collapse.

`maskrcnn_spatialmamba_s_deploy.py` is the deployed configuration. It
raises the per-image detection cap, and its header explains why that
matters only in dense plantations and why validation could not have shown
it.

**This is an operational adaptation, not part of the benchmark.** The
Stage C checkpoints and their reported numbers are untouched.

Supporting tools in [`configs/Custom/Finetune_HN/`](configs/Custom/Finetune_HN):
hard-negative tile mining, threshold re-calibration, and an evaluation that
measures false-positive suppression directly — because COCO mAP is close
to blind to it on tiles with no ground truth.

---

## The backbones

| family | models |
|---|---|
| CNN | ResNet-50, ResNet-101, ConvNeXt-T |
| Transformer | Swin-T/S, PVTv2-B2 |
| State-space | VMamba, Spatial-Mamba, GroupMamba, EfficientVMamba, MambaVision |
| SSM ablation | **MambaOut** — *removes* the SSM; a control, not a Mamba model |

Wrappers live in `mmdet/models/backbones/` and **contain no architecture
code**. They adapt upstream implementations expected at fixed paths —
`/opt/vmamba`, `/opt/spatial_mamba`, `/opt/groupmamba`,
`/opt/efficientvmamba`. Those paths are hard-coded and the directory names
are load-bearing. [`THIRD_PARTY.md`](THIRD_PARTY.md) gives each project's
repository and the exact commit used.

---

## Running the deployed model

```bash
python -m palm_inference.run_inference \
  --input-root /path/to/geotiffs \
  --output-root /path/to/output \
  --config-file configs/Custom/5_deployment_finetune/maskrcnn_spatialmamba_s_deploy.py \
  --checkpoint /path/to/checkpoint.pth \
  --tile-size 1024 --overlap 256 \
  --score-thr 0.30 --postprocess
```

`palm_inference/` is a tiled, **resumable**, georeferenced pipeline:
it tiles large rasters, runs batched inference, merges and de-duplicates
across tile boundaries, and writes GeoPackage. An interrupted run resumes
from its manifest.

**Three things that silently give wrong answers:**

- **The CLI defaults are not the deployment settings.** They are
  512 / 128 / 0.35; deployment used **1024 / 256 / 0.30**. Omitting them
  succeeds and produces a plausible map that is not the reported one.
- **Without `--postprocess`**, palms straddling tile boundaries are
  counted twice.
- **Resolution matters more than any setting.** Trained at ~15 cm/px; at
  1 m/px a crown is a few pixels across and will not be found.

---

## Reproducing the environment

This is the genuinely hard part, so it is stated plainly.

Three dependencies compile CUDA extensions against a specific torch/CUDA
pair — **`mmcv 2.1.0`** (source-only on PyPI), **`mamba-ssm 2.2.4`**,
**`causal-conv1d 1.4.0`** — plus the SSM projects' own `selective_scan`
and `dwconv2d` kernels. A mismatch anywhere fails at import or, worse, at
the first GPU kernel launch.

[`docker/Dockerfile.reconstructed`](docker/Dockerfile.reconstructed) is the
recipe, pinned to the exact commits used. **Order matters**: torch is
installed before mmcv, because mmcv compiles against whatever torch it
finds.

| | |
|---|---|
| base | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| python / torch | 3.10.12 / 2.1.0+cu121 |
| mmengine / mmcv / mmdet / mmpretrain | 0.10.1 / 2.1.0 / 3.3.0 / 1.2.0 |
| mamba-ssm / causal-conv1d | 2.2.4 / 1.4.0 |
| GPU used | TITAN RTX (sm_75), 24 GB |

**Verify with both, not one:**

```bash
python configs/Custom/utils/handover_selftest.py     # does it import?
python configs/Custom/utils/smoke_build_models.py    # do the models RUN?
```

The first proves imports. The second builds every model and pushes a
tensor through it. A backbone can import cleanly and fail on its first
forward pass — importing touches Python, a kernel launch touches the GPU.

**GPU architecture:** `dwconv2d` as originally built carried `sm_75` only.
Rebuild the kernels with a wider `TORCH_CUDA_ARCH_LIST` (the Dockerfile
does) or newer cards fail with `no kernel image is available for execution
on the device`.

---

## Building datasets

[`configs/Custom/utils/TILING_README.md`](configs/Custom/utils/TILING_README.md)
documents the pipeline: mosaic + reference polygons → 512 px tiles +
[LabelMe](https://github.com/wkentaro/labelme) JSON → COCO. One job file
per corpus; tile size, overlap and band selection derive from each
mosaic's own GSD.

Three policies materially affect results and are explained there: empty
tiles belong in **train only**, never val or test; background is capped at
30% with a fixed seed; and `filter_empty_gt`'s only symptom is a lower
image count in the log.

---

## What is not here

**No imagery, annotations or trained checkpoints.** The data is licensed
to the project. Every config needed to retrain is here.

**No pretrained weights.** [`weights.yaml`](weights.yaml) records each by
official source and **SHA256**, and separates *upstream* files from
*derived* ones produced locally that exist nowhere online. Several were
renamed during the work — **match by hash, not filename**.

**Not the Google Earth acquisition tooling.** Withheld; the imagery is
subject to the provider's terms. Nothing in the modelling code depends on
how imagery was obtained.

Full list and reasons: [`WITHHELD.md`](WITHHELD.md).

---

## Citation and licence

Please cite the manuscript (details on publication), MMDetection, and the
upstream backbone projects in [`THIRD_PARTY.md`](THIRD_PARTY.md).

```bibtex
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and others},
  journal = {arXiv preprint arXiv:1906.07155},
  year    = {2019}
}
```

Apache-2.0, inherited from MMDetection ([`LICENSE`](LICENSE)). Upstream
backbones carry their own terms — note in particular that **MambaVision is
NVIDIA non-commercial**.
