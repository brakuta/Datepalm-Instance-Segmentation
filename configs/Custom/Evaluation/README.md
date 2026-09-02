# Evaluation

A source-agnostic test-set evaluation pipeline. The same scripts
evaluate any trained Mask R-CNN model on any test source (UAV, Google
Earth, Aerial, or WorldView-3 satellite) and report mAP and F-score
for both the bounding-box and instance-segmentation tracks, plus a
computational-efficiency profile per model.

Folder location: `configs/Custom/Evaluation/`

> Before running anything here: these scripts operate
> on trained checkpoints and prepared COCO-format test sets, neither of
> which is distributed with this repository (see `WITHHELD.md` and
> `weights.yaml` at the repository root). Dataset paths default to
> `/workspace/datasets/COCO/...` in `sensor_registry.py`; edit the
> `_COCO_ROOT` constant there to point at your own data. A few
> workflow scripts (`make_stagec_manifest.py`, `splice_uav8.py`,
> `verify_pkl_metrics.py`) additionally carry `/workspace/mmdetection/...`
> constants from the machines this work ran on; adapt those before use.


## 1. Quick start

Evaluate one trained model on one or more test sets, then compile the
per-model CSVs into cross-backbone tables:

```bash
# 1. One model, one or more sensors -> one CSV per sensor:
python configs/Custom/Evaluation/evaluate_model.py \
    --config      configs/Custom/2_pooled_15cm_ge_aerial/maskrcnn_r50_ms15.py \
    --checkpoint  work_dirs/Stage_B/maskrcnn_r50_ms15/best_coco_segm_mAP_50_iter_25000.pth \
    --sensors     GE Aerial \
    --results-dir results/stage_b

# 2. Aggregate every per-model CSV into wide-form tables (CSV + XLSX):
python configs/Custom/Evaluation/compile_results.py \
    --results-dir results/stage_b
```

The compiler can also drive the runner for backbones whose CSVs are
missing, so a whole experiment is one resumable command:

```bash
python configs/Custom/Evaluation/compile_results.py \
    --results-dir results/stage_b \
    --run-missing \
    --config-dir  configs/Custom/2_pooled_15cm_ge_aerial \
    --work-root   work_dirs/Stage_B \
    --backbones   maskrcnn_r50_ms15 maskrcnn_swin_s_ms15 maskrcnn_vmamba_s_ms15
```

Models already evaluated are skipped, so re-running as more training
finishes is safe. Compiled tables land in `<results-dir>/compiled/`.

For the cross-resolution / cross-sensor transfer experiment (zero-shot
evaluation of every single-source model on every other domain):

```bash
# 1. Build the experiment manifest by scanning the training work_dirs:
python configs/Custom/Evaluation/build_manifest.py \
    --stage-a-dir work_dirs/Stage_A \
    --stage-b-dir work_dirs/Stage_B \
    --out cross_transfer.json

# 2. Evaluate every (model, sensor) cell; resumable, one CSV per cell:
python configs/Custom/Evaluation/run_cross_transfer.py --manifest cross_transfer.json

# 3. Compile the matrices, transfer gaps, and figures:
python configs/Custom/Evaluation/compile_cross_transfer.py --manifest cross_transfer.json
```


## 2. Files

Eighteen Python files. Three are supporting modules that are imported,
not run; four are the main entry points; the rest are manifest builders,
secondary compilers, and checks that supported specific experiments.

### Supporting modules (imported by the others; not run in normal use)

- `sensor_registry.py`: central registry of every test set and its
  paths. The single source of truth for dataset locations, and the only
  file to edit when a path changes or a source is added.
- `metrics_engine.py`: the metric engine. Defines
  `PalmBenchmarkMetric`, `patch_config`, and the CSV writer shared by
  every evaluator, so all numbers in the paper come from one metric
  implementation. It also has its own CLI (`--config`, `--checkpoint`,
  `--ann-file`, `--img-prefix`) for a raw single evaluation outside the
  registry, but the runner below is the normal route.
- `efficiency.py`: the efficiency profiler (params, FLOPs,
  latency, FPS, peak VRAM) under a fixed single-image fp32 regime.
  Called by `evaluate_model.py`; documents why FLOPs are flagged as an
  undercount for the Mamba backbones.

### Main entry points

- `evaluate_model.py`: per-model runner. Evaluates one model on
  one or more sensors (`--sensors`, keys from the registry), and writes
  one accuracy CSV per sensor plus one efficiency CSV per model.
  `--sensor-checkpoints SENSOR=PATH ...` expresses the experiment 3
  (unified multi-source) diagonal protocol, where UAV-derived sensors
  take the `best_UAV_*` checkpoint and GE-derived sensors `best_GE_*`.
- `compile_results.py`: cross-backbone compiler. Ingests the
  per-sensor CSVs, transposes each into one wide row per backbone, and
  writes per-sensor tables, a combined table, and an XLSX workbook to
  `<results-dir>/compiled/`. With `--run-missing` it first invokes
  `evaluate_model.py` for any backbone lacking CSVs, locating each
  checkpoint under `--work-root` (see section 7).
- `run_cross_transfer.py`: zero-shot transfer driver. Evaluates
  each single-source model across its row of the transfer matrix, as
  declared in a manifest JSON (`--manifest`;
  `--write-example-manifest PATH` emits a template). Reuses the metric
  stack verbatim; its one addition is injecting the no-resize test
  pipeline for the resampled sensors (see the registry), so transfer
  and in-domain numbers are produced by identical code. Resumable:
  cells whose CSV exists are skipped unless `--overwrite` is passed.
- `compile_cross_transfer.py`: analyser for the transfer
  experiment. Reads the per-cell CSVs named by the manifest
  (`--manifest`, optionally `--results-dir`, `--no-figures`) and writes
  the models-by-sensors matrices, per-model transfer gaps with the
  matched-GSD decomposition, and the degradation figures.

### Manifest builders (cross-transfer experiment)

The transfer experiment is defined by a manifest JSON rather than
hardcoded lists, so adding or removing a model is a manifest edit, not
a code change. These scripts write or amend manifests:

- `build_manifest.py`: auto-generates the full manifest by
  scanning the experiment 1 (UAV) and experiment 2 (pooled 15 cm)
  work_dirs, resolving each run's best checkpoint and dumped config,
  and inferring the backbone label and family from the run-dir name.
  Unresolvable runs are written as `UNRESOLVED__...` placeholders and
  listed, so a missing run is visible in the manifest rather than
  absent from it.
- `make_stagec_manifest.py`: adds the experiment 3 (unified
  multi-source) model under the diagonal checkpoint protocol, expressed
  as two manifest rows per backbone (`Stage_C_U` with `best_UAV_*`,
  `Stage_C_G` with `best_GE_*`).
- `make_ge30sim_manifest.py`: derives a `GE_30sim`-only manifest
  from the full one, so the one missing matrix column can be evaluated
  without touching the existing cells or fighting relative-path resume
  state.
- `add_ge30sim.py`: one-shot, idempotent patch that added the
  `GE_30sim` sensor to `sensor_registry.py` and to the manifest matrix.
  Kept as a record of how that entry entered the registry; the shipped
  registry already contains it.
- `splice_uav8.py`: one-shot repair that restored eight UAV-trained
  backbones which had been dropped from the manifest, reconstructing
  their entries from the on-disk work_dir layout. Kept for provenance.

### Validation and experiment 4 compilers

- `evaluate_validation.py`: evaluates trained experiment 1/2/3
  checkpoints on their *validation* splits under the same metric and
  protocol as the test evaluation, because model selection happened on
  validation and that number is usually reported alongside the test
  score. Takes
  `--stage A=work_dirs/Stage_A B=work_dirs/Stage_B C=work_dirs/Stage_C`
  and is resumable.
- `compile_validation.py`: assembles the validation CSVs into a
  summary CSV, per-experiment wide tables, and LaTeX tables.
- `compile_stage_d.py`: experiment 4 (satellite) compiler. Exists
  because the b0 and cf arms share one config file, so
  `compile_results.py` would collapse them into one row; this compiler
  keys the arm on the results subdirectory
  (`results/stage_d/{zeroshot,b0,cf,ms}/`) instead.

### Checks and audits

- `audit_predictions.py`: ground-truth audit visualiser. Runs a
  detector over an annotated set and renders TP (green) / FP (red) /
  FN (blue) polygon outlines per image, plus a per-image
  `audit_counts.csv` sorted by FP count, so suspected annotation gaps
  and genuine false alarms can be judged by eye.
- `verify_pkl_metrics.py`: recomputes mAP directly from saved
  prediction PKLs (`tools/test.py --out` output) and diffs the result
  against the compiled matrix. A PKL written from the wrong checkpoint
  or split reproduces the wrong mAP here and is caught.
- `bench_dataloader_stagec.py`: pre-launch sanity check for the
  experiment 3 training dataloader. Iterates it without a model forward
  and reports data_time, per-source quota adherence, batch homogeneity,
  and worker memory growth, so a slow or misconfigured loader is found
  before GPU hours are committed.


## 3. Reported metrics

For every model and test set, the pipeline reports the following for
both the `bbox` and `segm` tracks.

| Metric | Description |
|--------|-------------|
| `mAP@50` | Mean AP at IoU 0.50. Primary metric. |
| `mAP@[.5:.95]` | COCO mAP over IoU 0.50–0.95. Supplementary. |
| F1 / P / R (optimal) | At the F1-maximising score threshold. |
| F1 / P / R (fixed) | At the locked threshold, score = 0.05. |
| TP / FP / FN | Counts at the fixed threshold. |

In the compiled tables the `segm_*` and `bbox_*` column blocks sit side
by side, so both tracks appear in one row per model. The efficiency CSV
adds params, GFLOPs (flagged where undercounted), latency, FPS, and
peak VRAM per model.


## 4. How the pieces fit

```
        sensor_registry.py            (test-set paths)
                 |  imported by
                 v
   evaluate_model.py  --uses-->  metrics_engine.py + efficiency.py
        (per-model runner)             (metric engine, profiler)
                 |  invoked by (--run-missing)
                 v
   compile_results.py  -->  results/<stage>/compiled/
        (cross-backbone compiler)      (CSV + XLSX tables)

   build_manifest.py --> cross_transfer.json --> run_cross_transfer.py
        (manifest)                                    |
                                                      v
   compile_cross_transfer.py  -->  matrices, gaps, figures
```

The compiler runs the per-model runner for any model lacking results,
then aggregates every per-sensor CSV into the final tables. The runner
uses the metric engine. Every evaluator reads test-set paths from the
registry and writes the same long-form CSV schema, which is why the
compilers can consume each other's cells interchangeably.


## 5. Test sources

All sources are declared in `sensor_registry.py`, with paths derived
from the `_COCO_ROOT` constant (default `/workspace/datasets/COCO`;
edit it to your dataset location).

| Key | Experiment | GSD | Notes |
|-----|------------|-----|-------|
| `UAV` | 1 (single-sensor UAV) | 5 cm | |
| `GE` | 2 (pooled 15 cm) | 15 cm | |
| `Aerial` | 2 (pooled 15 cm) | 15 cm | |
| `Sat` | 4 (satellite WV-3) | 30 cm | RGB |
| `SatMS` | 4 (satellite WV-3) | 30 cm | 8-band multispectral arm |
| `UAV_15sim` | transfer | 15 cm | UAV test resampled; no-resize pipeline |
| `UAV_30sim` | transfer | 30 cm | UAV test resampled; no-resize pipeline |
| `GE_30sim` | transfer | 30 cm | GE test resampled; no-resize pipeline |

The three `*sim` entries carry an optional `test_pipeline` key: their
tiles are written at their true coarse pixel size, and the standard
1024 px resize would enlarge them and erase the resolution effect being
measured. `run_cross_transfer.py` injects that no-resize pipeline (and
batch size 1) automatically; `evaluate_model.py` ignores the key and is
unaffected.

Stage naming: the internal stage letters A/B/C/D used in work_dir and
registry labels correspond to the published experiment folders
`1_single_sensor_uav_5cm`, `2_pooled_15cm_ge_aerial`,
`3_unified_multisource`, and `4_satellite_wv3_30cm` respectively.


## 6. Command-line options (runner and compiler)

| Option | Meaning |
|--------|---------|
| `--sensors` | Which test sets to evaluate (registry keys). |
| `--device` | Compute device. See section 8. |
| `--ckpt-pattern` | Glob to select a checkpoint. See section 7. Compiler only. |
| `--num-workers` | DataLoader workers; use `0` if RAM-constrained. |
| `--results-dir` | Where CSVs and compiled tables are written. |
| `--max-dets` | Per-image detection cap for both COCO passes. Use one value benchmark-wide. |
| `--applied-score-thr` | Validation-selected operating threshold for the reported F1; omitted = swept on the test set. |
| `--no-efficiency` | Runner only: skip the efficiency profile. |

Run any script with `-h` for its full option list; every flag is
documented there.


## 7. Checkpoint selection

When run with `--run-missing`, the compiler locates each model's
checkpoint inside `<work-root>/<config_stem>/`. Resolution order:

| Priority | Rule |
|----------|------|
| 1 | If `--ckpt-pattern` is given, use that glob. |
| 2 | Else match `best_*.pth`; if several, take highest iter. |
| 3 | Fallback: highest-iter `iter_*.pth` or `epoch_*.pth`. |

This handles every experiment with no code change:

- Single-checkpoint runs (experiments 1, 2, 4): one `best_*` checkpoint,
  named after the monitored metric key; selected directly.
- Unified multi-source runs (experiment 3): two per-sensor
  checkpoints, `best_UAV_*` and `best_GE_*`; the highest-iteration one
  is chosen and a warning lists both. Pass `--ckpt-pattern "best_GE_*"`
  to pick one explicitly, or use `evaluate_model.py
  --sensor-checkpoints` to express the full diagonal in one launch.


## 8. GPU acceleration

Inference runs on the GPU. Device selection is explicit; the scripts
never fall back to the CPU on their own, since a CPU run is one to two
orders of magnitude slower and is almost never intended.

| `--device` | Behaviour |
|------------|-----------|
| `auto` (default) | Use the GPU; abort if none is visible. |
| `cuda` / `cuda:N` | Target a specific GPU; abort if unavailable. |
| `cpu` | Force CPU. Debugging only. |

The runner pins the MMEngine config to the resolved device and enables
`cudnn_benchmark`, so cuDNN picks the fastest convolution algorithms
for the fixed 1024x1024 inference input (the cross-transfer driver
disables it for the variable-size no-resize sensors, where autotuning
cannot help). The device, GPU name, and memory are printed at the
start of every run.


## 9. Adding a new source

1. Open `sensor_registry.py`, copy any sensor block, and set its four
   fields: `stage`, `ann_file`, `img_prefix`, `label`. Add the optional
   `test_pipeline` field only for a set that must be evaluated at its
   true (non-1024) tile size.
2. Pass the new key via `--sensors`, or add it to a manifest matrix for
   the transfer experiment.

No other file changes; every evaluator and compiler reads the
registry.


## 10. Protocol and notes

| Item | Value |
|------|-------|
| Fixed score threshold | `0.05` (locked) |
| IoU threshold | `0.50` (locked) |
| Inference input size | 1024 x 1024 (native sensors) |

Additional notes:

- The scripts import their siblings via
  `sys.path.insert(0, <this folder>)`, so they can be launched from any
  working directory but must stay together in this folder.
- The runner and compilers contain no hardcoded dataset paths; all
  dataset paths live in `sensor_registry.py`. Paths are not validated
  at import time; each script checks at run time and skips a missing
  test set with a clear message.
- The GE test annotation must be a correctly-named COCO file,
  `test_GE.json`. Confirm this before running GE evaluation.

Before comparing numbers across experiments, three things are easy to
miss:

1. Training budget. The experiments did not all run the same number of
   iterations. Check the schedule each config inherits
   (`_base_palm/schedule_*.py`) before comparing across experiments;
   `_base_palm/STAGE_C_REDESIGN.md` documents a case where two
   experiments were compared while training under different precision
   and optimiser settings.
2. MambaOut is not a state-space model. It is the ablation with the SSM
   removed; counting it among the Mamba family inflates that family's
   spread and misreads the control as a result.
3. The detection cap. The deployed config raises `max_per_img`. The cap
   binds only in dense plantations, and no validation tile was that
   dense, so a metric computed on validation cannot show its effect
   either way.

For experiment 4, `configs/Custom/tools_staged/summarize_stage_d.py`
compiles the satellite-transfer scores from the run logs.


## 11. Troubleshooting

Each entry below is a message the scripts may print, followed by its
cause and fix.

**`no CUDA device available`**
The container does not see the GPU. Check `nvidia-smi`, and start the
container with `--gpus all`.

**`test annotation not found` / `image dir missing`**
The sensor's `ann_file` or `img_prefix` in `sensor_registry.py` is
wrong for your machine, or the dataset is not prepared yet. Edit
`_COCO_ROOT` (or the entry) in the registry.

**`unknown sensor(s) [...]`**
The requested `--sensors` key is not declared in `sensor_registry.py`.
The message lists the valid keys; add an entry if the source is new.

**`multiple best_* checkpoints`**
The work_dir holds several `best_*` files (an experiment-3 run). Pass
`--ckpt-pattern` to choose one explicitly.

**`work_dir not found`**
`--work-root` does not contain a `<config_stem>/` directory for that
backbone. Check the root and the spelling of `--backbones` against the
actual run-dir names.

**`glob ... is ambiguous, matched N files`**
A checkpoint glob (e.g. in a manifest or `--sensor-checkpoints`)
matched more than one file. Tighten it until exactly one matches.

**`a run lock exists`**
`run_cross_transfer.py` refuses to start while another driver appears
active on the same results dir. If none is running, delete the named
`.run.lock` file and retry.
