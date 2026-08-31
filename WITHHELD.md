# What is not in this repository, and why

This repository contains the code for the experiments reported in
the paper. Some things are deliberately absent. They are listed
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

## Internal working notes

Working context for the authors, superseded by the README.

- `PROJECT_CONTEXT_MERGED.md`

## Data and trained weights

No imagery, annotations or trained checkpoints are published here.
The imagery is licensed to the project and cannot be redistributed.

Third-party pretrained weights are not redistributed either. See
`weights.yaml`, which records each one by its OFFICIAL download URL
and its SHA256, so the exact file used can be obtained from its
source and verified. Several were renamed locally during the work;
match them by hash, not by filename.

`MambaVision` is released by NVIDIA under a non-commercial licence.
It is referenced and configured here; its weights must be obtained
from NVIDIA under that licence and are not redistributed. That
restriction applies to MambaVision alone and not to the other
backbones.
