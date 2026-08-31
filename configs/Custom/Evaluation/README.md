# Evaluation — Date Palm Detection Benchmark

A source-agnostic, production-grade test-set evaluation pipeline. The
same scripts evaluate any trained Mask R-CNN model on any test source —
UAV, Google Earth, Aerial, or WorldView-3 satellite — and report mAP
and F-score for both the bounding-box and instance-segmentation tracks.

Folder location: `configs/Custom/Evaluation/`

> **Viewing this file:** the tables below render correctly in any
> Markdown viewer. In VS Code, press `Ctrl+Shift+V` to open the
> rendered preview. In the plain-text editor view, tables appear as
> raw `|` characters — that is expected.


## 1. Quick start

Evaluate every backbone of a stage and compile the result tables, in
one command:

```bash
# Run the launcher for the stage you want, from anywhere:
bash configs/Custom/Evaluation/run_stage_a.sh   # UAV 5 cm
bash configs/Custom/Evaluation/run_stage_b.sh   # GE + Aerial 15 cm
bash configs/Custom/Evaluation/run_stage_d.sh   # WorldView-3 30 cm
bash configs/Custom/Evaluation/run_evaluation_stagec.sh   # Stage C
```

Each per-stage launcher already contains that stage's backbone list and
paths; edit those variables inside the file if your folders differ.

Results are written to `results/<stage>/compiled/` as CSV and XLSX.


## 2. Files

The folder has six files. Only two are edited in normal use.

| File | Edited? |
|------|---------|
| `sensor_registry.py` | to add a source or fix a path |
| `run_stage_a.sh` / `run_stage_b.sh` / `run_stage_d.sh` | to set a stage's backbones |
| `run_evaluation_stagec.sh` | Stage C dual-checkpoint passes |
| `_run_common.sh` | shared launch logic (not edited) |
| `metrics_engine.py` | no |
| `evaluate_model.py` | no |
| `compile_results.py` | no |
| `README.md` | no |

Roles:

- **`sensor_registry.py`** — central registry of every test set and its
  paths. The single source of truth for dataset locations.
- **`metrics_engine.py`** — the metric engine. Defines
  `PalmBenchmarkMetric` and the helper functions. Not run directly.
- **`evaluate_model.py`** — per-model runner. Evaluates one
  model on one or more sensors; writes one CSV per sensor.
- **`compile_results.py`** — cross-backbone compiler.
  Aggregates per-model CSVs into wide-form tables; can also drive the
  runner for models that have no CSVs yet.
- **`run_stage_a.sh`**, **`run_stage_b.sh`**, **`run_stage_d.sh`** —
  one launcher per stage. Each sets only its own sensors, paths, and
  backbone list, then sources the shared `_run_common.sh`.
- **`run_evaluation_stagec.sh`** — dedicated Stage C launcher (two
  checkpoint passes: best_UAV_* and best_GE_*).
- **`_run_common.sh`** — shared path resolution, pre-flight checks, and
  compiler invocation. Sourced by the per-stage launchers; not run or
  edited directly.


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
by side, so both tracks appear in one row per model.


## 4. How the pieces fit

```
        sensor_registry.py            (test-set paths)
                 |  imported by
                 v
   evaluate_model.py  --uses-->  metrics_engine.py
        (per-model runner)                (metric engine)
                 |  invoked by
                 v
   compile_results.py  -->  results/<stage>/compiled/
        (cross-backbone compiler)        (CSV + XLSX tables)
                 ^
        run_stage_*.sh                  (drives the sequence)
```

Each `run_stage_*.sh` calls the compiler. The compiler runs the per-model
runner for any model lacking results, then aggregates every per-sensor
CSV into the final tables. The runner uses the metric engine. Both the
runner and the compiler read test-set paths from the registry.


## 5. Usage

Three entry points, from most to least automated.

### 5.1 Whole stage — one command

Run the launcher for your stage:

```bash
bash configs/Custom/Evaluation/run_stage_b.sh
```

Each launcher (`run_stage_a.sh`, `run_stage_b.sh`, `run_stage_d.sh`,
`run_evaluation_stagec.sh`) carries that stage's sensors, config
directory, work_dir root, and backbone list. To change which backbones
run, edit the `BACKBONES` array inside that one file. Models already
evaluated are skipped, so a launcher is safe to re-run as more training
finishes.

### 5.2 A single model — manual

```bash
python configs/Custom/Evaluation/evaluate_model.py \
    --config      <path/to/config.py> \
    --checkpoint  <path/to/checkpoint.pth> \
    --sensors     UAV GE Aerial Sat \
    --results-dir results/run1
```

### 5.3 Compile only — CSVs already exist

```bash
python configs/Custom/Evaluation/compile_results.py \
    --results-dir results/run1
```

### Command-line options

| Option | Meaning |
|--------|---------|
| `--sensors` | Which test sets to evaluate (registry keys). |
| `--device` | Compute device. See section 8. |
| `--ckpt-pattern` | Glob to select a checkpoint. See section 7. |
| `--num-workers` | DataLoader workers; use `0` if RAM-constrained. |
| `--results-dir` | Where CSVs and compiled tables are written. |

`--sensors`, `--device`, `--num-workers`, `--results-dir` apply to both
the runner and the compiler. `--ckpt-pattern` applies to the compiler.


## 6. Test sources

All four sources are declared in `sensor_registry.py`, with paths under
`/workspace/datasets/COCO/`.

| Key | Stage | GSD |
|-----|-------|-----|
| `UAV` | Stage A | 5 cm |
| `GE` | Stage B | 15 cm |
| `Aerial` | Stage B | 15 cm |
| `Sat` | Stage D | 30 cm |

Per-source paths (relative to `/workspace/datasets/COCO/`):

```
UAV     Annotations/test_UAV.json      test_UAV/      (in UAV_5cm/)
GE      Annotations/test_GE.json       test_GE/       (in GE_15cm/)
Aerial  Annotations/test_aerial.json   test_aerial/   (in Aerial_15cm/)
Sat     Annotations/test_sat.json      test_sat/      (in Sat_30cm/)
```


## 7. Checkpoint selection

When run with `--run-missing`, the compiler locates each model's
checkpoint inside its work_dir. Resolution order:

| Priority | Rule |
|----------|------|
| 1 | If `--ckpt-pattern` is given, use that glob. |
| 2 | Else match `best_*.pth`; if several, take highest iter. |
| 3 | Fallback: highest-iter `iter_*.pth` or `epoch_*.pth`. |

This handles every stage with no code change:

- **Stage B** — one `best_coco_segm_mAP_50_*` checkpoint; selected
  directly.
- **Stage C** — two per-sensor checkpoints, e.g. `best_UAV_*` and
  `best_GE_*`; the highest-iteration one is chosen and a warning lists
  both.
- **Stage D** — one `best_*` checkpoint; selected directly.

To pick a specific checkpoint when several `best_*` files exist:

```bash
--ckpt-pattern "best_GE_*"
```


## 8. GPU acceleration

Inference runs on the GPU. Device selection is explicit — there is no
silent CPU fallback, since a CPU run is one to two orders of magnitude
slower and is almost never intended.

| `--device` | Behaviour |
|------------|-----------|
| `auto` (default) | Use the GPU; abort if none is visible. |
| `cuda` / `cuda:N` | Target a specific GPU; abort if unavailable. |
| `cpu` | Force CPU. Debugging only. |

The runner pins the MMEngine config to the resolved device and enables
`cudnn_benchmark`, so cuDNN picks the fastest convolution algorithms
for the fixed 1024x1024 inference input. The device, GPU name, and
memory are printed at the start of every run.

Set the `DEVICE` variable in the stage launcher (default `auto`), or
pass `--device` on the command line.


## 9. Adding a new source

1. Open `sensor_registry.py`, copy any sensor block, and set its four
   fields: `stage`, `ann_file`, `img_prefix`, `label`.
2. Pass the new key via `--sensors`, or add it to a stage profile in
   the relevant `run_stage_*.sh`.

No other file changes — the runner and compiler both read the registry.


## 10. Protocol and notes

| Item | Value |
|------|-------|
| Fixed score threshold | `0.05` (locked) |
| IoU threshold | `0.50` (locked) |
| Inference input size | 1024 x 1024 |

Additional notes:

- `evaluate_model.py` imports the metric engine via
  `from metrics_engine import ...`; the two files must therefore sit in
  the same directory (they do, in this folder).
- The runner and compiler contain no hardcoded dataset paths; all paths
  live in `sensor_registry.py`.
- The GE test annotation must be a correctly-named COCO file,
  `test_GE.json`. Confirm this before running GE evaluation.


## 11. Troubleshooting

Each entry below is a message the scripts may print, followed by its
cause and fix.

**`no CUDA device available`**
The container does not see the GPU. Check `nvidia-smi`, and start the
container with `--gpus all`.

**`config directory not found`**
`CONFIG_DIR` in the stage launcher you ran is wrong. Edit it to the
real path.

**`work_dir root not found`**
`WORK_ROOT` in the active stage profile is wrong. Edit it.

**`test annotation not found`**
The sensor's `ann_file` in `sensor_registry.py` is wrong, or the file
does not exist yet.

**`multiple best_* checkpoints`**
The work_dir holds several `best_*` files. Pass `--ckpt-pattern` to
choose one explicitly.

**`config missing, cannot evaluate`**
A name in `BACKBONES` has no matching `<name>.py` in `CONFIG_DIR`.
Check the spelling against the actual config filenames.
