# What is not in this repository, and why

This repository contains the code for the experiments reported in
the manuscript. Some things are deliberately absent. They are listed
here so their absence is a decision on record rather than an
oversight a reader has to guess about.

## Google Earth acquisition tooling

Withheld at the authors' discretion. The imagery it retrieves is subject to the provider's terms of service. Nothing in the modelling code depends on how the imagery was obtained: the experiments reproduce from any imagery of comparable resolution.

- `download_ge_tiles.py`
- `probe_tile_throughput.py`
- `probe_xyz_coverage.py`
- `screen_grid_cells.py`
- `split_survey_ids.py`
- `check_download_coverage.py`
- `map_unavailable_tiles.py`
- `plan_redownload.py`
- `oman_project_WS1.json`
- `oman_project_WS2.json`
- `oman_project_README.md`

## Archive and handover tooling

Specific to retiring the machines this work ran on. Of no use to a reader, and it names private drives and hosts.

- `make_project_archive.py`
- `mirror_verify.py`
- `verify_backup.py`
- `reconcile_audits.py`
- `make_public_repo.py`
- `docs/handover/FACTS.yml` — the handover fact sheet, including the
  SHA256 identities of the trained checkpoints (which are themselves not
  published)
- `env_capture/environment.json` — the output of
  `configs/Custom/utils/capture_environment.py` on the original machine,
  including the hash of every weight file present there

## Internal working notes

Working context for the authors, superseded by the README.

- `PROJECT_CONTEXT_MERGED.md`
- `SESSION_CONTEXT.md`

## Internal launch scripts

Thin wrappers around the published Python entry points, written for the
authors' own work trees. The Python scripts they wrapped are all
published; the READMEs give the equivalent commands.

- `run_stage_a.sh`, `run_stage_b.sh`, `run_stage_d.sh`,
  `run_evaluation_stagec.sh`, `_run_common.sh` (evaluation drivers)
- `run_feature_analysis.sh` (feature-analysis driver)
- `make_stagec_pkls.sh`
- `clean_coco_degenerate.py`, `sensor_registry_crossres_additions.py`
  (one-off helpers absorbed into the published scripts)

## Data and trained weights

No imagery, annotations or trained checkpoints are published here.
The imagery is licensed to the project and cannot be redistributed.

Third-party pretrained weights are not redistributed either. See
`weights.yaml`, which records each one by its official source — the
pinned upstream repository or HuggingFace id — and, for the files
central to the reported results, by SHA256, so the exact file used
can be obtained from its source and verified. Where a hash or a
release-asset URL was not captured before the machines were retired,
the entry says `UNKNOWN` rather than guessing. Several files were
renamed locally during the work; match them by hash, not by filename.

`MambaVision` is released by NVIDIA under a non-commercial licence.
It is referenced and configured here; its weights must be obtained
from NVIDIA under that licence and are not redistributed. That
restriction applies to MambaVision alone and not to the other
backbones.
