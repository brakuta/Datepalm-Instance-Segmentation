# `utils/`: dataset construction, inference, environment checks

## Building datasets

| file | what it does |
|---|---|
| `image_vector_to_labelme_pipeline.py` | mosaic + reference polygons → tiles (512 px by default; `--set TILE_SIZE=1024` for the experiment 1–3 layout) + LabelMe JSON |
| `labelme2coco_palm.py` | tiles → COCO, straight into the layout the dataset configs read |
| `jobs_example.json` | the job file to copy and edit; keep your edited copy, since it is the record of how a dataset was built |
| `labels.txt` | the class definition; match it exactly when extending a dataset |
| `TILING_README.md` | tiling policies and procedure; read it before tiling anything |

`image_vector_to_labelme_pipeline.py` reads its default input and output
roots from `PALM_DATA_ROOT` and `PALM_OUTPUT_DIR`; set those rather than
editing the file.

## Running inference

| file | what it does |
|---|---|
| `palm_inference_pipeline.py` | an earlier single-file inference pipeline, kept for reference; the national run recorded in the handover used `palm_inference/` |
| `COUNTRY_SCALE_MAPPING_WRITEUP.md` | how the national inventory was actually produced |

## Checking an installation

| file | what it does |
|---|---|
| `handover_selftest.py` | checks that the environment imports and reports versions |
| `smoke_build_models.py` | builds each model and pushes a tensor through it |
| `capture_environment.py` | records resolved versions, CUDA/driver, compiled-extension status and weight hashes |

Run both of the first two. A backbone can import cleanly and still fail
on its first forward pass, because importing only runs Python code
while the forward pass launches the compiled GPU kernels.

## Analysis

`compare_runs.py`, `measure_postproc_recall.py`,
`crown_diameter_analysis.py`, `make_qualitative_grid.py`.
