# `utils/`: dataset construction, inference, environment checks

## Building datasets

| file | what it does |
|---|---|
| `image_vector_to_labelme_pipeline.py` | mosaic + reference polygons → 512 px tiles + LabelMe JSON |
| `labelme2coco_palm.py` | tiles → COCO, straight into the layout the dataset configs read |
| `jobs_example.json` | the job file to copy and edit; keep your edited copy, since it is the record of how a dataset was built |
| `labels.txt` | the class definition; match it exactly when extending a dataset |
| `TILING_README.md` | tiling policies and procedure; read it before tiling anything |

Set `PALM_DATA_ROOT` and `PALM_OUTPUT_DIR` rather than editing paths in
the file.

## Running inference

| file | what it does |
|---|---|
| `palm_inference_pipeline.py` | the configurable pipeline behind `palm_inference/` |
| `COUNTRY_SCALE_MAPPING_WRITEUP.md` | how the national inventory was actually produced |

## Checking an installation

| file | what it does |
|---|---|
| `handover_selftest.py` | checks that the environment imports and reports versions |
| `smoke_build_models.py` | builds each model and pushes a tensor through it |
| `capture_environment.py` | records resolved versions, CUDA/driver, compiled-extension status and weight hashes |

Run both of the first two. A backbone can import cleanly and still fail
on its first forward pass: importing exercises the Python side, a
kernel launch exercises the GPU.

## Analysis

`compare_runs.py`, `measure_postproc_recall.py`,
`crown_diameter_analysis.py`, `make_qualitative_grid.py`.
