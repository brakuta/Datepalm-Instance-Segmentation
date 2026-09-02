# Stage D: WorldView-3 30 cm feasibility (v4)

> This is the historical Stage D design memo, kept as it was written during
> the work. The folder itself ships the full config matrix (5
> `*_ge30sim_stage1`, 11 `*_staged_ft`, 4 `*_staged_full` and 4
> `*_staged_ms` configs, 24 in total) regardless of the narrower subset
> this memo proposes reporting. Where the memo and this folder's
> `README.md` disagree about what exists, the `README.md` is current.

Supersedes v3 (four backbones × three arms + budget curve). v4 frames
Stage D as a feasibility experiment, not a full study: two arms, four
backbones, eight runs, one table. The scope cut was deliberate. Every
additional experimental axis demands its own methodological documentation and
validation, not just GPU hours.

## 1. The one question

*Does the cross-resolution model built in Stages A–C transfer to 30 cm
WorldView-3, and is 30 cm workable at all for date-palm instance
segmentation?*

Everything that does not answer that question is out.

## 2. Matrix: 4 backbones × 2 arms

| Backbone | Family |
|---|---|
| ConvNeXt-T | CNN |
| Swin-S | Transformer (strongest Transformer in Stages A–C) |
| SpatialMamba-S | SSM |
| MambaVision-S | Hybrid |

| Arm | Initialisation | Adaptation regime | Config |
|---|---|---|---|
| **b0** | ImageNet (natural images) | full training: nothing frozen, lr 1e-4, backbone lr_mult 0.1, 60k iters | `maskrcnn_<bb>_staged_full.py` |
| **cf** | Stage C `best_GE_*` (cross-resolution RS) | full training, **identical to b0** | same config + `load_from=` |
| *c* | Stage C `best_GE_*` | constrained fine-tune: `frozen_stages=2`, head lr 2e-5, backbone lr_mult 0.01, 40k iters | `maskrcnn_<bb>_staged_ft.py` + `load_from=` |

Plus zero-shot Stage C → WV-3 (no training; evaluation only).

## 3. What the arms actually compare

b0 vs cf is an initialisation comparison, not a fine-tuning one. Both
arms train every weight, on the same data, with the same schedule, samplers,
caps and budget. The only difference is where the optimiser starts: natural
images or a cross-resolution remote-sensing model. Arm cf is a warm start,
not a fine-tune: nothing is frozen and the learning rate is the full 1e-4.

Only arm *c* is fine-tuning in the strict sense: stem and first two stages
frozen, the rest moving at 2e-5 × 0.01, the model largely preserved.

So:

* b0 vs cf isolates the initialisation, recipe held fixed.
* c vs cf isolates the adaptation regime, initialisation held fixed.

The question b0-vs-cf asks is the standard one in remote sensing (does a
domain-matched initialisation beat ImageNet?), and unlike the earlier framing
it does not depend on any recipe being the right choice.

### Report iterations-to-best, not only the best score

With every weight free for ~33 epochs on 3,636 tiles, the influence of the
initialisation partly washes out; b0 ≈ cf on final accuracy is a plausible and
reportable outcome, meaning *at this dataset size a cross-resolution prior
confers no accuracy advantage over ImageNet*.

But the prior may still pay in convergence speed. ConvNeXt-T b0 reached its
best at iteration 32,400. If cf peaks far earlier at a comparable score, the
prior bought a real reduction in training cost, measured under a matched
recipe, which is what makes it defensible. The earlier cost claim was not:
arm c was a different recipe, and it was not even faster (5-7 hours either
way, against b0's 5-7).

Record `best score` and `best iteration` for every run.

### The first result, and why arm c is no longer the default

| ConvNeXt-T | best segm mAP@50 | at iter | stopped |
|---|---|---|---|
| b0 | **0.8020** | 32,400 | ~46,800 |
| c | 0.7690 | 28,800 | ~38,400 |

The constrained fine-tune lost by 3.3 points and was not cheaper. That is not
evidence against the prior: Stage C learned crowns at 40–120 px and WV-3
crowns span ~20 px, so the early layers encode the wrong scale statistics and
arm c freezes precisely the layers that needed to change. Arm cf exists to
remove that explanation. Arm c is kept for ConvNeXt-T only, as the
illustrative case of why the conservative recipe fails.

## 3b. Training budget: the answer to "why not 120k like Stage C?"

Stage C ran 120,000 iterations at per-GPU batch 1 = 120,000 samples.
Stage D arm B0 runs 60,000 iterations at per-GPU batch 2 = 120,000 samples.
The exposure is identical; only the batch size, and therefore the iteration
count, differs. With samples seen recorded alongside iterations, the question
does not arise.

Both stages use effective batch 4 and are governed by EarlyStopping on
`coco/segm_mAP_50`; the selected iteration is recorded for every run.

Stage C carries a disclosed non-uniformity: it is not internally uniform.
All four backbones saw 120,000 samples at batch 1, but
`accumulative_counts` was 2 for ConvNeXt-T, Swin-S and MambaVision-S and 4 for
SpatialMamba-S, giving effective batch 2 versus 4, and 60,000 versus 30,000
optimiser steps. The comment in `schedule_stagec.py` claims effective batch 4
throughout; the built configs disagree, because
`dataset_UAV_GE_Aerial_pooled_C.py` sets `batch_size = 1` and the schedule's
`_BATCH` never reaches the dataloader. The runs are published and cannot be
redone, so this stands as a per-backbone memory accommodation with equal
samples and unequal optimiser steps, a detail that cuts against the
convenient direction, since SpatialMamba-S is the strongest model. Stage D is
uniform precisely so this does not recur.

## 4. Cut from v3, and why

| Cut | Reason |
|---|---|
| **Arm S** (GE-30sim simulation prior) | The stage-1 arm is a **simulation prior**, not a fine-tune: it requires full documentation of the simulation (resampling, PSF, noise, codec) and a defence of its realism, which is deferred to companion work. The trained `*_ge30sim_stage1` checkpoints become that work's starting material: not wasted, just not here. |
| **Budget curve (BU)** | The annotation-cost study is a separate contribution; it reads as a second paper inside this one. |
| **Backward evaluation / forgetting note** | Another axis, another table. |
| **Simulation sensitivity** | Already deferred in v3; stays deferred. |

## 5. The split, as built

Regenerated 4 Aug 2026 by `image_vector_to_labelme_pipeline.py` (512 px, 50%
training overlap, 30% background in train, seed 20260804) from the refined
reference vectors, over the Ajman and Kalba WorldView-3 mosaics pooled into
one contrast-stretch group:

| split | tiles | annotations | of which iscrowd | unique crowns | empty |
|---|---|---|---|---|---|
| train | 3,636 | 149,990 | 7,345 | 37,068 | 1,090 |
| val | 407 | 13,343 | 610 | 12,733 | 0 |
| test | 413 | 14,886 | 741 | 14,145 | 0 |

There are 63,946 distinct reference crowns. Quote that, not the 178k
polygons; 50% overlap writes each training crown up to four times.

Supersedes the 2,413 figure in `schedule_staged_ft.py` and the 267/142/124 in
the header of `dataset_sat_30cm_staged.py`; both are stale. Confirm before
launching (`<data_root>` is the `data_root` set in
`_base_palm/dataset_sat_30cm_staged.py`):

```bash
for s in train val test; do
  python -c "import json;d=json.load(open(
    '<data_root>/Annotations/${s}_sat.json'));
    print('${s}', len(d['images']), 'images', len(d['annotations']), 'anns')"
done
```

Tile density drove the detection cap: up to 622 crowns in a training tile, 333
in test, 327 in val, with 31 tiles above the Stage C base cap of 300. The
eight configs of this memo's matrix (the four selected backbones in the
`_staged_ft` and `_staged_full` arms) raise `test_cfg.rcnn.max_per_img` to
1000 so the cap cannot decide recall, and the four `_staged_ms` configs
inherit the raised cap from their `_staged_full` bases (twelve configs in
total carry it; `maskrcnn_spatialmamba_s_ge30sim_stage1.py` also sets 1000
inside its own `test_cfg`). The `_staged_ft` configs for the seven remaining
benchmark backbones keep the base value, the shared base is untouched, and
the Stage C benchmark still reports at 300.

Also confirm the SpatialMamba ImageNet weights exist, or arm B0 starts from
random init and becomes "from scratch" without any error being raised:

```bash
ls -la <checkpoints_dir>/spatialmamba/spatialmamba_small_in1k.pth
# and in the run log, expect "Successfully load ckpt", not "Failed loading"
```

## 5b. Confirm batch 2 fits, once per backbone

Every backbone runs at per-GPU batch 2 with `accumulative_counts=2` (effective
batch 4). The batch-1 accommodation SpatialMamba-S carried from Stage C was
sized for 1024 px tiles on a 24 GB TITAN; WV-3 tiles are 512 px, a quarter of
the pixels, so it no longer applies. Confirm rather than assume, because
an OOM 30 hours into a run costs more than a two-minute check.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for BB in spatialmamba_s mambavision_s swin_s convnext_t; do
  echo "== $BB"
  timeout 600 python tools/train.py \
    configs/Custom/4_satellite_wv3_30cm/maskrcnn_${BB}_staged_full.py \
    --cfg-options train_cfg.max_iters=60 train_cfg.val_interval=100000 \
                  default_hooks.checkpoint.interval=100000 \
    2>&1 | tail -3
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
done
```

Order matters: SpatialMamba-S first, then MambaVision-S, the two heaviest. If
either reports CUDA out of memory, that one backbone falls back to batch 1 +
`accumulative_counts=4` at `max_iters=120_000`, which preserves 120,000
samples and effective batch 4; only the number of optimiser steps changes
(30,000 either way, since accumulation doubles with the halved batch). Say so
and the fallback goes into that config.

## 6. Run: both arms

`run_staged_matrix.sh` is the single entry point and is v4-aware: arm `b0`
resolves to `maskrcnn_<bb>_staged_full.py` with no `load_from`, arm `c` to
`maskrcnn_<bb>_staged_ft.py` with the Stage C `best_GE` prior injected. Arms
`s` and `bu` are refused outright rather than running a cut arm with no
warning.

```bash
# see exactly what would launch, resolve every checkpoint, write nothing
bash configs/Custom/tools_staged/run_staged_matrix.sh all b0 --dry-run
bash configs/Custom/tools_staged/run_staged_matrix.sh all c  --dry-run

# then, in order
bash configs/Custom/tools_staged/run_staged_matrix.sh all b0
bash configs/Custom/tools_staged/run_staged_matrix.sh all c
```

The arm-C dry run is the real test of the prior inventory: the runner resolves
each Stage C checkpoint by glob and refuses on a missing or ambiguous match,
so if all four resolve there, arm C cannot fail on a bad path hours later.

Stage C `best_GE` checkpoints, one per backbone in that backbone's Stage C
work directory:

| Backbone | checkpoint |
|---|---|
| ConvNeXt-T | `best_GE_segm_mAP_50_iter_85001.pth` |
| Swin-S | `best_GE_segm_mAP_50_iter_95001.pth` |
| SpatialMamba-S | `best_GE_segm_mAP_50_iter_75001.pth` |
| MambaVision-S | `best_GE_segm_mAP_50_iter_95001.pth` |

The Stage C prior is the GE-selected checkpoint (satellite-like selection
target); that selection choice is part of the recorded protocol.

### Two checks in the first minute of the first run

* the dataset line must read 3,636 train images. Lower means
  `filter_empty_gt` discarded the 1,090 background tiles and the whole
  background-supervision decision was reverted with no message in the log.
* for SpatialMamba-S, the log must say `Successfully load ckpt`, not
  `Failed loading`. Its ImageNet weights load through the backbone's
  `pretrained` path rather than `init_cfg`, so a missing file degrades to
  random initialisation without an error; arm B0 would become "from
  scratch" and the arm-C comparison would be meaningless.

## 7. Evaluation

Locked protocol: `PalmBenchmarkMetric`, segm mAP@50, `max_dets 500`, F1 at the
F1-optimal threshold, on the refined WV-3 test set. Earlier single-stage WV-3
numbers and all June/July Stage D fine-tunes predate the refined ground truth
and are superseded: not compared, not shown.

## 8. Reporting

One table, twelve rows: 4 backbones × {zero-shot, B0, C}. Columns: segm
mAP@50, F1, and wall-clock training time (the cost half of the claim). Stage
D is reported compactly, as a feasibility finding rather than a full study,
with the table carrying the result.

Recipe-selection note: the freeze regime and loss variant were selected on a
preliminary annotation version (June runs) and held fixed for all reported
arms; never re-tuned.

## 9. Per-backbone accommodations (carried verbatim from Stage C)

| Config | Accommodation |
|---|---|
| convnext_t, swin_s | none (batch 2) |
| mambavision_s | reduced RPN sampler/proposal caps; `compute_flops=False`; `cudnn_benchmark=False` |
| spatialmamba_s | batch 1 + `accumulative_counts=4` (effective batch 4) |

These are identical in the `_full` and `_ft` configs; the two arms differ
only in schedule base and `frozen_stages`.
