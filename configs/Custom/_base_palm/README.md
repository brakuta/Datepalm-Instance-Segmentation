# `_base_palm/` — shared building blocks

Nothing here is run directly. Every experiment config inherits from these.

## Datasets — one per experiment

| file | used by |
|---|---|
| `dataset_uav_5cm.py` | `1_single_sensor_uav_5cm` |
| `dataset_MS15_pooled.py` | `2_pooled_15cm_ge_aerial` |
| `dataset_UAV_GE_Aerial_pooled_C.py` | `3_unified_multisource` |
| `dataset_ge30sim.py` | `4_satellite_wv3_30cm` (simulated-30 cm pre-training) |
| `dataset_sat_30cm_staged.py` | `4_satellite_wv3_30cm` (real WorldView-3) |

Each header states the corpus, tile counts and provenance. **Read the
header before changing a path** — several document why a setting is what
it is.

### Paths you must edit

The dataset files ship with the original container's absolute paths, and
an outside reader has none of that layout. Before training, point these at
your own data (built per `configs/Custom/utils/TILING_README.md`):

- **`data_root`** (and the per-source roots in
  `dataset_UAV_GE_Aerial_pooled_C.py`) — where the COCO trees live. Keep
  them consistent with `configs/Custom/Evaluation/sensor_registry.py`, or
  training and evaluation will silently read different data.
- **`pretrained=` / `init_cfg` checkpoint paths** in the per-experiment
  configs — where the ImageNet weights sit (`weights.yaml` identifies each
  file; some configs use relative `checkpoints/...`, others the original
  absolute paths).
- **`work_dir`** in the Stage 3/4 configs — where runs write; folders 1–2
  use relative `./work_dirs/`.

## Schedules

`schedule_*.py` — iteration budget, optimiser, precision, LR schedule.
Cross-stage comparability depends on these matching; `STAGE_C_REDESIGN.md`
documents a case where they silently did not.

## Models

`_base_maskrcnn_palm_stagec.py` and siblings define the detector with
`backbone=None` and `neck=None`, so each experiment config supplies its
own. **This is why every config needs `_delete_=True` on those keys** —
without it MMEngine tries to merge a dict into `None` and refuses to build
the config at all.

## Hooks and samplers

| file | what it does |
|---|---|
| `sensor_balanced_sampler*.py` | source-local batch construction for the pooled datasets |
| `per_sensor_best_checkpoint_hook.py` | keeps the best checkpoint per sensor, not just overall |
| `mean_sensor_metric_hook.py` | reports the mean across sensors, so one dominant source cannot carry the number |
| `nms_fp32_guard.py` | forces NMS to FP32 under mixed precision, where it is dtype-unsafe |
| `mem_probe_hook.py`, `benchmark_logging_hook.py` | memory and throughput instrumentation |
| `loading_multispectral.py`, `ms_pipelines.py`, `ms_data_preprocessor.py` | the >3-channel path |

## The one thing that will cost you a day

`randomness.seed` is `None`. Each run draws its own seed and writes it
**only into that run's log**. If you need to reproduce a run, the log is
the sole record.
