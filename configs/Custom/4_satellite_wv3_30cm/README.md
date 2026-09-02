# Experiment 4: satellite transfer (WorldView-3, 30 cm)

*Internal name: Stage D. Datasets: `_base_palm/dataset_sat_30cm_staged.py`
(real WV-3) and `_base_palm/dataset_ge30sim.py` (simulated 30 cm).*

This experiment asks how far the approach carries to 30 cm satellite
imagery, where a crown is roughly 17 px across instead of 100, and
which initialisation transfers best.

**Read `STAGE_D_README.md` in this folder before running anything here.**
It is the most involved experiment in the repository. `STAGE_D_README.md`
is the historical design memo, kept as written during the work; this README
describes what the folder ships.

## Config families

| suffix | purpose |
|---|---|
| `_ge30sim_stage1` | pre-training on simulated 30 cm imagery: GE 15 cm downsampled 2× with PSF blur and sensor noise, 19,472 tiles. Crowns land at ~17 px, the real WV-3 scale, so the model meets the target scale with far more data than the real corpus has |
| `_staged_full` | full training on real WV-3 (60k iterations at batch 2, nothing frozen). `run_staged_matrix.sh` runs it from ImageNet weights (arm `b0`), from the experiment 3 checkpoint (arm `cf`) or from the simulated-30 cm checkpoint (arm `s`), so the arms differ only in initialisation |
| `_staged_ft` | fine-tuning on real WV-3 from the experiment 3 checkpoint with the stem and first two stages frozen (arm `c`) |
| `_staged_ms` | 8-band multispectral WV-3 rather than RGB |

Real WV-3 corpus: 3,636 train / 407 val / 413 test tiles, 63,946 distinct
reference crowns (counts from the regenerated 4 Aug ground truth; see
`STAGE_D_README.md`).

## Budget manifests

`tools_staged/build_budget_manifests.py` builds nested annotation-budget
subsets (the 5% subset is contained in the 10%, which is contained in
the 25%), drawn with a fixed seed and stratified by palm count, so that
the measured effect of adding labels does not depend on which tiles
happened to be drawn. The budget curve was dropped from the reported
study (see `STAGE_D_README.md`), and `run_staged_matrix.sh` refuses the
`bu` arm; the manifests remain usable by overriding a config's training
annotation file, for example
`--cfg-options train_dataloader.dataset.ann_file=Annotations/train_sat_b25.json`.
If you build them, build them once: rebuilding changes which tiles fall
in each budget.

## Multispectral stem

An ImageNet-pretrained stem takes 3 channels; WV-3 multispectral has 8.
`tools_staged/inflate_stem_to_nband.py` widens the stem so the run still
starts from pretrained weights instead of from scratch. Band statistics
come from `tools_staged/compute_band_stats.py`.

## Order of operations

1. `*_ge30sim_stage1.py`: simulated-30 cm pre-training (the prior for
   arm `s`)
2. `tools_staged/select_stagec_checkpoint.py`: choose the experiment 3
   checkpoint to transfer from (the prior for arms `c` and `cf`)
3. `tools_staged/run_staged_matrix.sh <b0|c|cf|s|all>`: run the arms;
   the script injects each arm's starting checkpoint
4. `summarize_stage_d.py`: tabulate every run from its own logs
