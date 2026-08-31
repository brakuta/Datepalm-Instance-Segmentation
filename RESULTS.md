# Results

**The results are in the manuscript, not here.** This page will carry the
headline tables once it is published; until then it would be a second copy
of numbers still under review, and a second copy is a copy that drifts.

What this repository gives you instead is everything needed to regenerate
them.

## Regenerating the tables

Every run writes its own metrics. Nothing below re-reads a spreadsheet.

```bash
# per-experiment scores, read out of each run's own logs
python configs/Custom/tools_staged/summarize_stage_d.py       # satellite transfer
python configs/Custom/Evaluation/compile_results.py           # benchmark tables
python configs/Custom/Evaluation/compile_cross_transfer.py    # cross-sensor transfer
```

`configs/Custom/Evaluation/` holds the metrics engine, the manifest
builders, and the per-model evaluation entry points. Its own README
describes them.

## Reading a comparison honestly

Three things affect the numbers and are easy to miss:

**Training budget.** Not every stage ran the same number of iterations.
Check the schedule each config inherits before comparing across stages —
`_base_palm/schedule_*.py`. `_base_palm/STAGE_C_REDESIGN.md` documents a
case where two stages were compared while training under different
precision and optimiser settings.

**MambaOut is not a state-space model.** It is the ablation with the SSM
removed. Counting it among the Mamba family inflates that family's spread
and misreads the control as a result.

**The detection cap.** The deployed config raises `max_per_img`. It binds
only in dense plantations, which no validation tile was — so a metric
computed on validation cannot show its effect either way.

## Reproducing a single number

1. Build the environment — `docker/Dockerfile.reconstructed`
2. Verify it — `handover_selftest.py`, then `smoke_build_models.py`
3. Obtain the pretrained weights — `weights.yaml`, matched by SHA256
4. Build the dataset — `configs/Custom/utils/TILING_README.md`
5. Train the config, then evaluate with the matching script above

Seeds are not in the configs: `randomness.seed` is `None` and the drawn
seed appears only in that run's log. Exact reproduction of a specific run
needs that log.
