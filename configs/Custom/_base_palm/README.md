# `_base_palm/`: shared building blocks

Nothing in this folder is run directly. Every experiment config inherits
from these files.

## Dataset definitions

One dataset file per corpus:

| file | used by |
|---|---|
| `dataset_uav_5cm.py` | `1_single_sensor_uav_5cm` |
| `dataset_MS15_pooled.py` | `2_pooled_15cm_ge_aerial` |
| `dataset_UAV_GE_Aerial_pooled_C.py` | `3_unified_multisource` |
| `dataset_ge30sim.py` | `4_satellite_wv3_30cm` (simulated-30 cm pre-training) |
| `dataset_sat_30cm_staged.py` | `4_satellite_wv3_30cm` (real WorldView-3) |

Each header states the corpus, tile counts and provenance. Read the
header before changing a path; several headers explain why a setting is
what it is.

### Paths to edit before training

The dataset files ship with the original container's absolute paths,
which an outside reader does not have. Before training, point the
following at your own data (built per
`configs/Custom/utils/TILING_README.md`):

1. `data_root`, and the per-source roots in
   `dataset_UAV_GE_Aerial_pooled_C.py` (which name the same trees under
   `/root/datasets/` rather than `/workspace/datasets/`; the Docker image
   links the two): where the COCO trees live. Keep
   them consistent with `configs/Custom/Evaluation/sensor_registry.py`;
   if they diverge, training and evaluation read different data and no
   error is raised.
2. `pretrained=` / `init_cfg` checkpoint paths in the per-experiment
   configs: where the ImageNet weights sit. `weights.yaml` identifies
   each file; some configs use relative `checkpoints/...`, others the
   original absolute paths.
3. `work_dir` in the experiment 3 and 4 configs, where runs write, and
   `load_from` in the experiment 4 and 5 configs, which name the
   experiment 3 checkpoints they start from. Experiments 1 and 2 use
   relative `./work_dirs/`.

## Schedules

The `schedule_*.py` files set the iteration budget, optimiser,
precision and LR schedule. Comparability across experiments depends on
these matching; `STAGE_C_REDESIGN.md` documents a case where they did
not match and nothing flagged it.

## Models

`_base_maskrcnn_palm_stagec.py` here, and `_base_maskrcnn_palm.py` and
`_base_maskrcnn_palm_ms15.py` in `1_single_sensor_uav_5cm/`, define the detector
with `backbone=None` and `neck=None`, so each experiment config
supplies its own. This is why every config needs `_delete_=True` on
those keys: without it MMEngine tries to merge a dict into `None` and
refuses to build the config.

## Hooks and samplers

| file | what it does |
|---|---|
| `sensor_balanced_sampler.py`, `sensor_balanced_sampler_n.py` | per-sensor rebalancing (experiment 2) and source-local batch construction (experiment 3) for the pooled datasets |
| `per_sensor_best_checkpoint_hook.py` | keeps the best checkpoint per sensor, not just overall |
| `mean_sensor_metric_hook.py` | reports the mean across sensors rather than a total dominated by the largest source |
| `nms_fp32_guard.py` | forces NMS to FP32 under mixed precision, where it is dtype-unsafe |
| `mem_probe_hook.py`, `benchmark_logging_hook.py` | memory and throughput instrumentation |
| `loading_multispectral.py`, `ms_pipelines.py`, `ms_data_preprocessor.py` | the >3-channel path |

## Seeds and reproducibility

The experiment 1 runtime (`runtime_palm.py`) leaves `randomness.seed`
unset: each run draws its own seed and writes it only into that run's
log, so reproducing a specific run needs that log, or
`--cfg-options randomness.seed=<value>` at launch. The other runtimes
fix `seed = 0`.
