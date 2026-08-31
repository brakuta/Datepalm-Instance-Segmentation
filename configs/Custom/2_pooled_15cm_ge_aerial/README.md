# 2 — Pooled 15 cm (Google Earth + aerial)

*Internal name: Stage B. Dataset: `_base_palm/dataset_MS15_pooled.py`.*

**The question.** At a coarser resolution than Stage A, does training on
two 15 cm sources together beat training on one?

## The design decision that matters

Training pools Google Earth 15 cm **and** aerial 15 cm. Validation is
**Google Earth only**; aerial is held out and scored afterwards through a
separate eval-only config run against the best GE-validation checkpoint.

That asymmetry is deliberate. If aerial were in the validation set, the
checkpoint would be chosen partly by its aerial performance, and "does
pooling help on aerial?" would be answered by a set that had already seen
it. Holding it out keeps that measurement honest.

## The configs

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

## Where to go next

Adding UAV 5 cm to the pool — three sources, one model —
→ `3_unified_multisource/`
