# `utils/` — dataset construction, inference, environment checks

## Building datasets

| file | what it does |
|---|---|
| `image_vector_to_labelme_pipeline.py` | mosaic + reference polygons → 512 px tiles + LabelMe JSON |
| `labelme2coco_palm.py` | tiles → COCO, straight into the layout the dataset configs read |
| `jobs_example.json` | the job file to copy and edit; **keep yours** — it is the record of how a dataset was built |
| `labels.txt` | the class definition. Match it exactly when extending a dataset |
| **`TILING_README.md`** | **read this before tiling anything** |

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
| `handover_selftest.py` | does the environment **import**? |
| `smoke_build_models.py` | do the models **run**? Builds each and pushes a tensor through it |
| `capture_environment.py` | records resolved versions, CUDA/driver, compiled-extension status and weight hashes |

**Run both of the first two.** A backbone can import cleanly and fail on
its first forward pass — importing touches Python, a kernel launch touches
the GPU.

## Analysis

`compare_runs.py`, `measure_postproc_recall.py`,
`crown_diameter_analysis.py`, `make_qualitative_grid.py`.
