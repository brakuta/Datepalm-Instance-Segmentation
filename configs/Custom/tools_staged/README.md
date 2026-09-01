# `tools_staged/`: satellite-transfer tooling

Supports `4_satellite_wv3_30cm/` (experiment 4). Read
`STAGE_D_README.md` first.

| script | what it does |
|---|---|
| `build_budget_manifests.py` | builds the nested, seeded annotation-budget subsets. Run once and freeze the output: rebuilding changes which tiles fall in each budget, and the ladder no longer measures what it did |
| `compute_band_stats.py` | per-band mean and std for the multispectral data preprocessor |
| `inflate_stem_to_nband.py` | widens a 3-channel ImageNet stem to N channels so multispectral runs still start pretrained |
| `select_stagec_checkpoint.py` | picks the checkpoint to transfer from, by best mean validation across sensors |
| `import_stage_checkpoints.py` | copies checkpoints off an ephemeral container overlay onto persistent storage |
| `summarize_stage_d.py` | tabulates every run from its own logs: best score, peak iteration, where it stopped |
| `run_staged_matrix.sh`, `run_zeroshot_wv3.sh` | drive the full matrix |

## Order

```
build_budget_manifests.py     # once, then frozen
*_ge30sim_stage1.py           # simulated-30 cm pre-training
*_staged_ft.py                # fine-tune across the ladder
summarize_stage_d.py          # read the results out of the logs
```
