# 1 — Single-sensor benchmark (UAV, 5 cm)

*Internal name: Stage A. Dataset: `_base_palm/dataset_uav_5cm.py`.*

**The question.** With the sensor fixed and everything else held constant,
which backbone segments date-palm crowns best?

This is the controlled comparison. Same imagery, same schedule, same
detector, same augmentation — only the backbone changes. Where a family
publishes two sizes, both are here, so model scale is a visible axis
rather than a hidden confound.

## The configs

```
maskrcnn_{r50,r101}_uav5cm.py                    CNN baselines
maskrcnn_{convnext_t}_uav5cm.py                  modern CNN
maskrcnn_{swin_t,swin_s,pvtv2_b2}_uav5cm.py      transformers
maskrcnn_{vmamba,spatialmamba,groupmamba,
          efficientvmamba,mambavision}_{t,s}_uav5cm.py   state-space
maskrcnn_mambaout_{t,s}_uav5cm.py                SSM ABLATION -- see below
```

**MambaOut is not a state-space model.** It is the architecture with the
SSM removed, included as a control. If it performs comparably to the SSM
models, that is evidence about what the SSM is contributing — which is the
point of including it. Do not count it as a Mamba family member.

## Running one

```bash
python tools/train.py configs/Custom/1_single_sensor_uav_5cm/maskrcnn_r50_uav5cm.py
```

Start with `maskrcnn_r50_uav5cm.py`. It is the fastest, has no compiled
CUDA extensions in its path, and if it fails the problem is your data or
install rather than an SSM kernel.

## Where to go next

- Pooling more sensors → `2_pooled_15cm_ge_aerial/`
- All sensors in one model → `3_unified_multisource/`
