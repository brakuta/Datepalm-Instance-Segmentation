# Stage C Training Pipeline: Diagnosis and Production Redesign

> **Historical record.** This document is the engineering record of the
> Stage C redesign, written while the work was being done and kept in that
> form. It is retained because `RESULTS.md` relies on it when interpreting
> cross-stage comparisons, and because the defect register explains settings
> that would otherwise look arbitrary. Paths have been updated to the
> published folder names; the analysis is otherwise as written at the time.

**Scope.** This document analyses the reported Stage C failure modes (severe
slowdown, host hangs, and system-memory spillover), identifies their root
cause, and specifies a production-grade redesign of the Stage C training
configuration. It also records latent configuration defects discovered during
the audit and documents one design decision that affected published results;
Section 4 records how it was resolved, and the resolved configuration is what
shipped.

---

## 1. Summary of the failure mode

Three symptoms were reported: training that is markedly slower than Stage A/B,
intermittent host freezes, and resident system memory approaching capacity
(VMamba-S reached 63.3 / 63.7 GB before the host froze; SpatialMamba-S
degraded to approximately 8.7 s/iter with spillover). These are not three
independent problems. They are the successive stages of a single mechanism:
under full FP32 at 1024×1024 with per-GPU batch 2, the heavy state-space
backbones exceed the 24 GB device memory of the TITAN RTX, after which the
WSL2/NVIDIA driver silently relocates device allocations into shared
(system) memory rather than raising an out-of-memory error.

The consequences follow directly. Every relocated allocation is thereafter
accessed across the PCIe bus, which inflates iteration time by roughly an
order of magnitude (the observed 2 → 16+ s/iter). When the spilled device
memory and the DataLoader working set together approach the 64 GB physical
limit, the host itself becomes unresponsive. The mechanism is therefore
memory-bound, not compute-bound, and no amount of kernel-level tuning resolves
it while the working set exceeds device memory.

---

## 2. Root-cause analysis

### 2.1 Primary cause

The primary cause is the locked decision to train Stage C in full FP32 at
per-GPU batch 2. This combination places the activation memory of the deepest
SSM backbones (notably VMamba-S and SpatialMamba-S, the latter with a
21-block third stage) above 24 GB once the first validation pass has
fragmented the allocator. Only two classes of intervention remove the
spillover at its source: reducing the numerical precision of activations
(mixed precision), or reducing the per-step working set (smaller per-GPU
batch). Both are adopted below, selectable through a single switch. Device
control-panel settings and allocator flags are defensive measures only; they
do not reduce the working set and therefore cannot be the primary remedy.

### 2.2 A contradiction in the locked precision decision

The FP32 decision was justified in the internal working notes (not published)
on two grounds: (i) it
avoids an FP16/FP32 dtype mismatch in `batched_nms` for the pure-SSM
backbones, and (ii) it matches Stage B for cross-stage comparability. The
second justification is contradicted by the Stage B artifacts themselves. The
Stage B schedule, `schedule_unified_MS_80k.py`, specifies:

```
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=1e-4, ...),
    ...
)
```

Stage B therefore trained under mixed precision (float16), with AdamW at
1e-4, gradient accumulation of 2 (effective batch 4), a backbone-aware
`paramwise_cfg`, and a cosine schedule. Because Stage B completed every
backbone in the 10-model set — including VMamba-S, SpatialMamba-S,
GroupMamba-S and MambaVision-S — on the same 24 GB devices under this
condition, mixed precision is demonstrably VRAM-feasible for all Stage C
backbones, and the `batched_nms` concern (justification i) was evidently
resolved or did not arise in the Stage B code path. Consequently, the Stage C
FP32 decision does not preserve comparability with Stage B; it breaks it.

**Resolution.** The Stage B per-backbone configs have since been examined. None
of the eleven training configs (ResNet-50, ConvNeXt-T, Swin-S, PVT-v2-B2,
MambaVision-S, VMamba-S, SpatialMamba-S, GroupMamba-S, MambaOut-S,
EfficientVMamba-B, and the VMamba-S scaling check) assigns `optim_wrapper`;
every one inherits `schedule_unified_MS_80k.py` unchanged. The two apparent
matches found by a naive text search are comment lines only ("reduce
`accumulative_counts` ... via a per-config `optim_wrapper` override"), not
assignments. The Stage B training condition is therefore unambiguously
`AmpOptimWrapper` with `dtype='float16'` across the entire cohort, including
every state-space backbone. This confirms the recommendation: `amp_fp16` is the
comparability-correct setting, and the working notes' assertion that FP32
matches Stage B is incorrect. The residual uncertainty flagged in the first
revision is closed; no further confirmation is required.

### 2.3 The benchmark-validity problem beneath the symptoms

Independently of memory, the Stage C v2 runtime diverges from Stage B on
several axes that are not memory-related and that compromise the cross-stage
comparison the benchmark is designed to support. These divergences, all
silent, are summarised below.

| Axis | Stage B (and Stage A) | Stage C v2 | Consequence |
|---|---|---|---|
| Optimizer | AdamW, lr 1e-4 | SGD, lr 0.02 | Different optimiser family; SGD 0.02 with full weight decay is unsuitable for pretrained SSM/transformer backbones |
| `paramwise_cfg` | backbone `lr_mult`=0.1; zero-decay on norm/bias and Mamba `A_log`/`D`/`dt_proj` | none | Backbone fine-tuned at full LR; norm/SSM parameters incorrectly decayed |
| Precision | AMP float16 | FP32 | Primary cause of the spillover; also a comparability break |
| Effective batch | 4 (batch 2 × accumulate 2) | 2 (no accumulation) | Different gradient statistics |
| LR schedule | Linear warmup 1.5k + Cosine to 1e-6 | Linear warmup 1k + MultiStep at 90k/110k | Different decay shape |
| `clip_grad` | max_norm 1.0 | max_norm 35 | Different gradient regularisation |
| `auto_scale_lr` | disabled | enabled, base 16 | Silently rescales the configured LR by 0.125 |
| Within Stage C | — | two configs override to AMP+AdamW; the rest use FP32+SGD | Backbones trained under different conditions in the same matrix |

The redesign realigns every one of these axes, so that Stage C becomes
comparable to Stage B and internally consistent across backbones.

---

## 3. Redesign specification

### 3.1 A single precision switch

`runtime_palm_stagec.py` now exposes one module-level variable:

```python
PRECISION = 'amp_fp16'   # 'amp_fp16' (recommended) | 'fp32'
```

- **`amp_fp16`** reproduces the Stage B condition exactly: `AmpOptimWrapper`
  with `dtype='float16'`, per-GPU batch 2, `accumulative_counts=2` (effective
  batch 4). This is the recommended setting. The TITAN RTX is a Turing device
  and supports float16 tensor cores but not bfloat16; float16 is therefore the
  correct and only AMP dtype here, and it is the dtype Stage B used.
- **`fp32`** retains full FP32 but sets per-GPU batch to 1 and
  `accumulative_counts=4`, preserving an effective batch of 4. Halving the
  per-GPU batch approximately halves activation memory, which is expected to
  bring the heavy SSM backbones inside 24 GB without altering precision.

In both modes the optimizer (AdamW, 1e-4), `paramwise_cfg`, gradient clipping
(1.0), schedule (cosine), and effective batch (4) are identical. The two modes
are therefore mutually comparable and each is comparable to Stage B on every
axis except numerical precision.

Under `fp32`, the runtime deep-merges `batch_size=1` and sampler
`chunk_size=4` over the dataset definition. Widening the chunk to the
accumulation window keeps each optimiser step drawn from a single source,
preserving the page-cache locality that `SensorBalancedSamplerN` was built to
exploit and avoiding cross-source micro-batches within an accumulation window.

### 3.2 Optimizer, schedule, and effective batch

The optimizer is unified to AdamW (lr 1e-4, weight decay 0.05) with the Stage
B `paramwise_cfg`, applied to all backbones. The Mamba-specific zero-decay
keys are no-ops for non-Mamba backbones. The schedule is Linear warmup over
1500 iterations followed by CosineAnnealing to `eta_min=1e-6` over the 120k
horizon, matching the Stage B shape. `auto_scale_lr` is disabled so the
configured learning rate is used verbatim. `clip_grad` is set to max_norm 1.0,
the Stage B value.

The 120k iteration ceiling and the EarlyStopping policy (patience 10 on the
mean of the two sensor mAP@50 scores, validation every 5000 iterations) are
retained from the Stage C design intent.

### 3.3 Per-backbone memory accommodations

The policy is to retain only those per-backbone accommodations that were also
present in Stage B, so that each backbone remains comparable to its own Stage
B run, and to remove accommodations introduced solely to contain the Stage C
FP32 overflow. Examination of the Stage B per-backbone configs shows that
**MambaVision-S is the only backbone with a Stage B accommodation**; every
other Stage B config trains at the base full proposal counts.

- **Retained** (present in Stage B): MambaVision-S only
  (`rpn.sampler.num`=128, `nms_pre`=500, `max_per_img`=512, together with
  `cudnn_benchmark=False`).
- **Removed** (Stage C only): SpatialMamba-S and PVT-v2-B2 both revert to the
  base full proposal counts. The Stage C PVT-v2-B2 config described its
  accommodation as "retained from Stage B", but the Stage B PVT-v2-B2 config
  has no `train_cfg` block and ran at full proposals; the accommodation was
  therefore Stage-C-only and is removed.

Under the recommended `amp_fp16` mode the Stage B memory conditions are
reproduced, so the Stage-C-only accommodations are unnecessary as well as
non-comparable.

### 3.4 Environment and data loading

The multiprocessing start method is `fork` with `opencv_num_threads=1`,
correctly nested under `env_cfg['mp_cfg']`. `fork` is the method that Stage
A/B effectively used (their top-level `mp_start_method='spawn'` key was
mis-placed and ignored by MMEngine, so both stages ran under `fork` and
completed); the OpenCV thread clamp is the genuine improvement over Stage A/B.
`cudnn_benchmark` is enabled because the input shape is constant.

DataLoader settings are consolidated into a single per-machine block.
Validation and test loaders use `persistent_workers=False` so their workers
are released between the 5000-iteration validation passes; training workers
remain persistent. `serialize_data=True` is retained on every sub-dataset,
which packs the annotation index into a shared-memory buffer and is the
correct mitigation for per-worker index duplication.

---

## 4. Decision resolved

This section previously requested confirmation of the authoritative Stage B
precision condition. The Stage B per-backbone configs have now been examined
and resolve it: all inherit `AmpOptimWrapper(float16)` with no per-config
override (Section 2.2). The delivered default `PRECISION='amp_fp16'` therefore
reproduces the Stage B condition exactly and requires no change. The `fp32`
mode is retained only as a contingency; if it is ever selected, the effective
batch, optimizer, paramwise, and schedule remain identical to the AMP mode, so
comparability is preserved either way.

---

## 5. Defect register

The following defects were identified and corrected in the revised files.

1. **Invalid hook keyword (run-blocking).** The v2 runtime passed
   `save_last=False` to `PerSensorBestCheckpointHook`, whose constructor does
   not accept that keyword. Any config inheriting the runtime hook stack
   without redeclaring it (ResNet-50, ResNet-101, Swin-S) would raise a
   `TypeError` at runner construction. The keyword has been removed.

2. **Malformed `env_cfg` override (silent).** `maskrcnn_mambavision_s_stagec.py`
   placed `mp_start_method='spawn'` at the top level of `env_cfg`. MMEngine
   reads this key only from `env_cfg['mp_cfg']`, so the setting was ignored and
   `opencv_num_threads` reverted to its unlimited default, re-introducing cv2
   thread oversubscription for that backbone. The override is rewritten with a
   correctly nested `mp_cfg` while retaining the intended
   `cudnn_benchmark=False`.

3. **Optimizer/precision inconsistency within the matrix (benchmark
   contaminant).** `maskrcnn_efficientvmamba_s_stagec.py` and
   `maskrcnn_mambaout_t_stagec.py` overrode the optimizer to AMP+AdamW while
   the remaining Stage C backbones used FP32+SGD. Both overrides are removed;
   precision and optimizer are now centralised.

4. **Harmful learning-rate override.** `maskrcnn_r101_stagec.py` set
   `optim_wrapper=dict(optimizer=dict(lr=0.02))`. Under the revised AdamW
   runtime this would have set the AdamW learning rate to 0.02. Removed.

5. **`auto_scale_lr` rescaling (silent).** Enabled with `base_batch_size=16`
   against an effective batch of 2, it scaled the configured learning rate by
   0.125. Disabled, matching Stage B.

6. **Unusable `worker_init_fn` config (run-blocking).** The dataset passed
   `worker_init_fn=dict(type='function', qualname=...)` to the train
   dataloader. MMEngine's `Runner.build_dataloader` constructs its own
   per-worker init function for seeding and does not resolve a user-supplied
   `worker_init_fn` from the config; the residual dict is forwarded to the
   PyTorch `DataLoader`, which fails `assert callable(worker_init_fn)` at train
   loop construction (open-mmlab/mmengine issue #933). The key has been
   removed. The cv2/OMP/MKL thread clamp it was meant to apply is provided
   instead by `env_cfg.mp_cfg.opencv_num_threads=1` (runtime) and the
   `OMP_NUM_THREADS` / `MKL_NUM_THREADS` shell exports in the launch command.
   The `dataloader_worker_init` entry remains in `custom_imports` and is inert
   (it registers an unreferenced function); it is retained without effect.

7. **AMP `batched_nms` dtype mismatch (run-blocking under `amp_fp16`).** Under
   `AmpOptimWrapper(float16)` the RPN post-NMS write-back
   `scores_after_nms[mask[keep]] = dets[:, -1]` raised `RuntimeError: Index put
   requires the source and destination dtypes match, got Half for the
   destination and Float for the source`. This is the FP16/FP32 `batched_nms`
   interaction cited in the internal working notes (not published) as the
   original motivation for the
   FP32 lock; it surfaced once the corrected runtime restored AMP. Notably,
   none of the Stage B per-backbone configs, the schedule, or the runtime
   contained any NMS or autocast guard, yet Stage B trained these same SSM
   backbones under AMP to completion — indicating the mismatch is build- and
   path-dependent rather than intrinsic to AMP. The fix is a minimal,
   reversible monkey-patch (`nms_fp32_guard.py`, loaded via `custom_imports`)
   that upcasts the NMS boxes/scores to float32, runs `batched_nms` with
   autocast disabled, and casts the result back to the input dtype, so the
   write-back is dtype-consistent. It changes no model config, backbone, or
   detector code, and is a no-op fast-path under `PRECISION='fp32'`. It is
   loaded in both precision modes so the configuration is identical across
   modes. All eleven published Stage C configs are covered: nine declare it in
   their (replaced) `custom_imports`; ResNet-50 and ResNet-101 inherit it from
   the runtime's `custom_imports`.

---

## 6. Per-file change log

**Revised — shared:**

- `runtime_palm_stagec.py` — PRECISION switch; AdamW + Stage B `paramwise_cfg`;
  `accumulative_counts`; cosine schedule; `auto_scale_lr` disabled; `clip_grad`
  1.0; `save_last` removed; FP32 dataloader deep-merge override;
  `nms_fp32_guard` added to `custom_imports`.
- `dataset_UAV_GE_Aerial_pooled_C.py` — consolidated per-machine block; AMP
  default batch 2 with `chunk_size` tracking `batch_size`; val/test
  `persistent_workers=False`; unusable `worker_init_fn` key removed (defect 6);
  **train augmentation aligned to Stage B**
  (Resize `keep_ratio=True` + Pad, horizontal *and* vertical `RandomFlip`,
  `PhotoMetricDistortion` restored). v2 had dropped vertical flip and
  photometric jitter, which diverged from Stage B and removed the
  augmentation most relevant to cross-sensor radiometric robustness.

**New — shared:**

- `nms_fp32_guard.py` — AMP-safe `batched_nms` wrapper (defect 7). Loaded via
  `custom_imports` in the runtime (inherited by ResNet-50 and ResNet-101) and
  re-declared in the other nine published per-backbone configs'
  `custom_imports`.

**Revised — per-backbone:**

- `maskrcnn_mambavision_s_stagec.py` — `env_cfg` corrected; Stage B
  accommodation retained.
- `maskrcnn_spatialmamba_s_stagec.py` — Stage-C-only accommodation removed.
- `maskrcnn_pvtv2_b2_stagec.py` — Stage-C-only accommodation removed (Stage B
  ran full proposals).
- `maskrcnn_r101_stagec.py` — lr override removed.

**No change required** (inherit the corrected runtime; model blocks already
consistent): `maskrcnn_r50_stagec.py`, `maskrcnn_convnext_t_stagec.py`,
`maskrcnn_swin_s_stagec.py`, `maskrcnn_vmamba_s_stagec.py`,
`maskrcnn_groupmamba_s_stagec.py`, `maskrcnn_efficientvmamba_b_stagec.py`,
`maskrcnn_mambaout_s_stagec.py`. Note that ResNet-50 and Swin-S were
previously unusable because they inherited the invalid `save_last` keyword
(defect 1); the runtime fix restores them without per-file edits.

At the time of the audit the working tree also contained two variant configs
outside the reported matrix (which uses EfficientVMamba-B and MambaOut-S);
their AMP/AdamW overrides were removed for hygiene (defect 3), but the
variants themselves were not carried into the published set. The published
folder `3_unified_multisource/` ships exactly eleven Stage C configs, one per
benchmark backbone:

```
maskrcnn_{r50,r101,convnext_t,swin_s,pvtv2_b2}_stagec.py
maskrcnn_{vmamba_s,spatialmamba_s,groupmamba_s,
          efficientvmamba_b,mambavision_s,mambaout_s}_stagec.py
```

---

## 7. Launch and verification

**Launch (single config):**

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python tools/train.py \
  configs/Custom/3_unified_multisource/maskrcnn_<backbone>_stagec.py
```

`expandable_segments:True` reduces allocator fragmentation and is a defensive
complement to the working-set reduction, not a substitute for it. The
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` exports clamp BLAS thread pools to one
per process; together with `env_cfg.mp_cfg.opencv_num_threads=1` in the runtime
they replace the per-worker thread clamp that the (unusable) custom
`worker_init_fn` was intended to provide — see defect 6 in the register above.

**Defence in depth (historical aside — original workstation only).** The
training hosts were Windows/WSL2 workstations, where the NVIDIA driver spills
device allocations into shared system memory instead of raising an
out-of-memory error. On that environment, the NVIDIA Control Panel option
*Manage 3D Settings → CUDA — Sysmem Fallback Policy → Prefer No Sysmem
Fallback* converts any residual overflow into a fast, explicit out-of-memory
error rather than a silent host-freezing spill. With the working set inside
24 GB this path is never exercised, and the setting is irrelevant on native
Linux hosts; it is recorded here as part of the original environment.

**Verification checklist.**

1. During the first 200 iterations, confirm in the host GPU monitor that
   *Shared GPU memory usage* remains at 0. Any non-zero value indicates the
   working set still exceeds device memory.
2. Confirm steady-state iteration time is in the expected single-digit
   seconds-per-iteration range, not the 16+ s/iter spillover regime.
3. After training, inspect `benchmark_record.json` for `peak_cuda_memory_mb`
   below the 24 GB device limit and a `sec_per_iter` consistent with Stage B.
4. Confirm the run builds without `TypeError` (validates the hook fix) and
   that `best_UAV_*` and `best_GE_*` checkpoints are written at validation.
