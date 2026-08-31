# 3 — Unified multi-source model

*Internal name: Stage C. Dataset:
`_base_palm/dataset_UAV_GE_Aerial_pooled_C.py`.*

**The question.** One model for every sensor, or one model per sensor?

Trains on UAV 5 cm, Google Earth 15 cm and aerial 15 cm together — a 3×
resolution span within a single model. **This is the comparison the
deployed model came from.**

## Source-local batching

Batches are drawn from one source at a time rather than mixed. The three
corpora live on different mounts, and interleaving them destroys kernel
page-cache locality: the same epoch takes far longer for reasons that
never appear in a profiler as anything but I/O wait.

This changes throughput, not what the model sees over an epoch.

## The configs

Eleven backbones — the full set:

```
maskrcnn_{r50,r101,convnext_t,swin_s,pvtv2_b2}_stagec.py
maskrcnn_{vmamba_s,spatialmamba_s,groupmamba_s,
          efficientvmamba_b,mambavision_s,mambaout_s}_stagec.py
```

[`../_base_palm/STAGE_C_REDESIGN.md`](../_base_palm/STAGE_C_REDESIGN.md)
documents a real failure and its fix:
an earlier version trained in FP32 while claiming comparability with
Stage B, which had trained in mixed precision. Backbones were being
compared under different conditions inside the same matrix. **Read it
before drawing conclusions from any cross-stage comparison.**

## Per-backbone accommodations

Not every backbone fits a 24 GB card at the same batch size. The SSM
models use batch 1 with gradient accumulation 4; MambaVision additionally
needs a proposal cap and `cudnn_benchmark=False`. These are recorded per
config, and they affect memory, not the comparison.

## Running one

```bash
python tools/train.py configs/Custom/3_unified_multisource/maskrcnn_r50_stagec.py
```

## Where to go next

- Pushing to 30 cm satellite → `4_satellite_wv3_30cm/`
- Deploying it → `5_deployment_finetune/`
