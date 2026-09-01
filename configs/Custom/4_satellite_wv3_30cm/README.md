# Experiment 4: satellite transfer (WorldView-3, 30 cm)

*Internal name: Stage D. Datasets: `_base_palm/dataset_sat_30cm_staged.py`
(real WV-3) and `_base_palm/dataset_ge30sim.py` (simulated 30 cm).*

This experiment asks how far the approach carries to 30 cm satellite
imagery, where a crown is roughly 17 px across instead of 100, and how
much annotation the transfer requires.

**Read `STAGE_D_README.md` in this folder before running anything here.**
It is the most involved experiment in the repository. `STAGE_D_README.md`
is the historical design memo, kept as written during the work; this README
describes what the folder ships.

## Config families

| suffix | purpose |
|---|---|
| `_ge30sim_stage1` | pre-training on simulated 30 cm imagery: GE 15 cm downsampled 2× with PSF blur and sensor noise, 19,472 tiles. Crowns land at ~17 px, the real WV-3 scale, so the model meets the target scale with far more data than the real corpus has |
| `_staged_ft` | fine-tuning on real WV-3 across a nested annotation-budget ladder |
| `_staged_full` | the full-budget reference point |
| `_staged_ms` | 8-band multispectral WV-3 rather than RGB |

Real WV-3 corpus: 3,636 train / 407 val / 413 test tiles, 63,946 distinct
reference crowns (counts from the regenerated 4 Aug ground truth; see
`STAGE_D_README.md`).

## Budget manifests

`tools_staged/build_budget_manifests.py` builds the annotation-budget
subsets once and freezes them. They are nested (the 5% subset is
contained in the 10%, which is contained in the 25%) and drawn with a
fixed seed, stratified by palm count. Without nesting and a fixed seed,
the measured effect of adding labels would partly reflect which tiles
happened to be drawn. Build the manifests once and do not rebuild them.

## Multispectral stem

An ImageNet-pretrained stem takes 3 channels; WV-3 multispectral has 8.
`tools_staged/inflate_stem_to_nband.py` widens the stem so the run still
starts from pretrained weights instead of from scratch. Band statistics
come from `tools_staged/compute_band_stats.py`.

## Order of operations

1. `build_budget_manifests.py`: run once, then freeze the manifests
2. `*_ge30sim_stage1.py`: simulated-30 cm pre-training
3. `*_staged_ft.py`: fine-tune across the budget ladder
4. `summarize_stage_d.py`: tabulate every run from its own logs
