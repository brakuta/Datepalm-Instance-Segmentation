# 4 — Satellite transfer (WorldView-3, 30 cm)

*Internal name: Stage D. Datasets: `_base_palm/dataset_sat_30cm_staged.py`
(real WV-3) and `_base_palm/dataset_ge30sim.py` (simulated 30 cm).*

**The questions.** How far does this carry to 30 cm satellite imagery —
where a crown is roughly 17 px across instead of 100? And how much
annotation does that actually take?

**Read `STAGE_D_README.md` in this folder before running anything here.**
It is the most involved experiment in the repository.

## Four config families

| suffix | what it does |
|---|---|
| `_ge30sim_stage1` | pre-train on **simulated** 30 cm: GE 15 cm downsampled 2× with PSF blur and sensor noise, 19,472 tiles. Crowns land at ~17 px — real WV-3 scale — so the model meets the target scale with far more data than the real corpus has |
| `_staged_ft` | fine-tune on real WV-3 across an **annotation-budget ladder** |
| `_staged_full` | the full-budget reference point |
| `_staged_ms` | **8-band multispectral** WV-3 rather than RGB |

Real WV-3 corpus: 3,636 train / 407 val / 413 test tiles, 63,946 crowns.

## The budget ladder is nested and seeded

`tools_staged/build_budget_manifests.py` builds the subsets once and
freezes them. They are **nested** — the 5% subset is contained in the 10%,
which is contained in the 25% — and drawn with a fixed seed, stratified by
palm count.

Without that, "more labels help this much" would partly measure which
tiles happened to be drawn. Build them once and do not rebuild.

## Multispectral needs a wider stem

An ImageNet-pretrained stem takes 3 channels; WV-3 multispectral has 8.
`tools_staged/inflate_stem_to_nband.py` widens it so the run still starts
from pretrained weights instead of from scratch. Band statistics come from
`tools_staged/compute_band_stats.py`.

## Order of operations

1. `build_budget_manifests.py` — once, then frozen
2. `*_ge30sim_stage1.py` — simulated-30 cm pre-training
3. `*_staged_ft.py` — fine-tune across the budget ladder
4. `summarize_stage_d.py` — tabulate every run from its own logs
