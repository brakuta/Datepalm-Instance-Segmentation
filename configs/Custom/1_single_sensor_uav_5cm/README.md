# Experiment 1: single-sensor benchmark (UAV, 5 cm)

*Internal name: Stage A. Dataset: `_base_palm/dataset_uav_5cm.py`.*

This experiment asks which backbone segments date-palm crowns best when
the sensor is fixed and everything else is held constant. It is the
controlled comparison: same imagery, same schedule, same detector, same
augmentation, only the backbone changes. Where a family publishes two
sizes, both are included, so model scale is an explicit axis rather than
a hidden confound.

## Configs

```
maskrcnn_{r50,r101}_uav5cm.py                    CNN baselines
maskrcnn_convnext_t_uav5cm.py                    modern CNN
maskrcnn_{swin_t,swin_s,pvtv2_b2}_uav5cm.py      transformers
maskrcnn_{vmamba,spatialmamba,groupmamba,
          mambavision}_{t,s}_uav5cm.py                   state-space
maskrcnn_efficientvmamba_{s,b}_uav5cm.py         state-space (sizes S and B; no T)
maskrcnn_mambaout_{t,s}_uav5cm.py                SSM ABLATION -- see below
```

MambaOut is not a state-space model. It is the architecture with the
SSM removed, included as a control: if it performs comparably to the SSM
models, that is evidence about what the SSM contributes. It should not be
counted as a member of the Mamba family when results are grouped.

## Running one

```bash
python tools/train.py configs/Custom/1_single_sensor_uav_5cm/maskrcnn_r50_uav5cm.py
```

Start with `maskrcnn_r50_uav5cm.py`. It is the fastest, has no compiled
CUDA extensions in its path, and if it fails the problem is in your data
or installation rather than an SSM kernel.

## Related experiments

- Experiment 2 (`2_pooled_15cm_ge_aerial/`): pooling two 15 cm sources
- Experiment 3 (`3_unified_multisource/`): all sensors in one model
