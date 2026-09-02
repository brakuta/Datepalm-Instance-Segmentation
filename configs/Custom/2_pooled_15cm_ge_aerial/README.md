# Experiment 2: pooled 15 cm (Google Earth + aerial)

*Internal name: Stage B. Dataset: `_base_palm/dataset_MS15_pooled.py`.*

This experiment trains a single model on two pooled 15 cm sources, at a
coarser resolution than experiment 1, with one source held out of
validation so that its score is an independent measurement.

## Validation design

Training pools Google Earth 15 cm and aerial 15 cm. Validation uses
Google Earth only; aerial is held out and scored afterwards with
`configs/Custom/Evaluation/evaluate_model.py --sensors Aerial` against
the best GE-validation checkpoint.

The asymmetry is intentional. If aerial were in the validation set, the
checkpoint would be chosen partly by its aerial performance, and the
question of whether pooling helps on aerial would be answered by data
that had already influenced checkpoint selection. Holding aerial out
keeps that measurement independent.

## Configs

Ten backbones, one config each:

```
maskrcnn_{r50,convnext_t,swin_s,pvtv2_b2}_ms15.py
maskrcnn_{vmamba_s,spatialmamba_s,groupmamba_s,
          efficientvmamba_b,mambavision_s,mambaout_s}_ms15.py
```

Schedule: `_base_palm/schedule_unified_MS_80k.py`, mixed precision, AdamW,
cosine.

## Running one

```bash
python tools/train.py configs/Custom/2_pooled_15cm_ge_aerial/maskrcnn_r50_ms15.py
```

## Related experiments

Experiment 3 (`3_unified_multisource/`) adds UAV 5 cm to the pool: three
sources, one model.
