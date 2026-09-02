# `tools_staged/`: satellite-transfer tooling

Supports `4_satellite_wv3_30cm/` (experiment 4). Read that folder's
`README.md` for what the configs are, and its `STAGE_D_README.md` (the
design memo) before running the matrix.

| script | what it does |
|---|---|
| `build_budget_manifests.py` | builds nested, seeded annotation-budget subsets for a labelling-cost study (dropped from the reported experiment; see the folder README). Build once: rebuilding changes which tiles fall in each budget |
| `compute_band_stats.py` | per-band mean and std for the multispectral data preprocessor |
| `inflate_stem_to_nband.py` | widens a 3-channel ImageNet stem to N channels so multispectral runs still start pretrained |
| `select_stagec_checkpoint.py` | recommends the experiment 3 checkpoint to transfer from: the surviving checkpoint closest to the iteration with the best mean of the two per-sensor validation scores |
| `import_stage_checkpoints.py` | copies a run's checkpoints from a container's work directory to persistent storage (written for the original environment; paths are arguments) |
| `summarize_stage_d.py` | tabulates every run from its own logs: best score, peak iteration, where it stopped |
| `run_staged_matrix.sh`, `run_zeroshot_wv3.sh` | drive the matrix (arms `b0`, `c`, `cf`, `s`; see the script header) and the zero-shot WV-3 evaluation |

## Order

```
build_budget_manifests.py     # once, then frozen
*_ge30sim_stage1.py           # simulated-30 cm pre-training
*_staged_ft.py                # fine-tune across the ladder
summarize_stage_d.py          # read the results out of the logs
```
