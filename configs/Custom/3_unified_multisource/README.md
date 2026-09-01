# Experiment 3: unified multi-source model

*Internal name: Stage C. Dataset:
`_base_palm/dataset_UAV_GE_Aerial_pooled_C.py`.*

This experiment asks whether one model can serve every sensor, or
whether each sensor needs its own. It trains on UAV 5 cm, Google Earth
15 cm and aerial 15 cm together, a 3× resolution span within a single
model. The deployed model came from this comparison.

## Source-local batching

Batches are drawn from one source at a time rather than mixed. The three
corpora live on different mounts, and interleaving them destroys kernel
page-cache locality: the same epoch takes far longer, and the slowdown
appears in a profiler only as I/O wait. Source-local batching changes
throughput, not what the model sees over an epoch.

## Configs

Eleven backbones, the full set:

```
maskrcnn_{r50,r101,convnext_t,swin_s,pvtv2_b2}_stagec.py
maskrcnn_{vmamba_s,spatialmamba_s,groupmamba_s,
          efficientvmamba_b,mambavision_s,mambaout_s}_stagec.py
```

[`../_base_palm/STAGE_C_REDESIGN.md`](../_base_palm/STAGE_C_REDESIGN.md)
documents a failure and its fix: an earlier version of this experiment
trained in FP32 while claiming comparability with experiment 2 (Stage B),
which had trained in mixed precision, so backbones were being compared
under different conditions inside the same matrix. Read it before
drawing conclusions from any cross-experiment comparison.

## Per-backbone accommodations

Not every backbone fits a 24 GB card at the same batch size. The SSM
models use batch 1 with gradient accumulation 4; MambaVision additionally
needs a proposal cap and `cudnn_benchmark=False`. These settings are
recorded per config, and they affect memory, not the comparison.

## Running one

```bash
python tools/train.py configs/Custom/3_unified_multisource/maskrcnn_r50_stagec.py
```

## Related experiments

- Experiment 4 (`4_satellite_wv3_30cm/`): transfer to 30 cm satellite imagery
- Experiment 5 (`5_deployment_finetune/`): deployment and hard-negative adaptation
