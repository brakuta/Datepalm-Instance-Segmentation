# `Finetune_HN/`: hard-negative mining and calibration

Supports `5_deployment_finetune/` (experiment 5). Two longer documents
sit beside these scripts and cover the method in detail:

- `README_hard_negative_finetune.md`: suppressing false positives on
  palm-like shrubs, ghaf and acacia
- `README_false_negative_finetune.md`: the counterpart, recovering
  missed palms

## Scripts

| script | what it does |
|---|---|
| `make_aoi_tiles.py` | cuts 1024 px tiles restricted to areas of interest given as polygons |
| `make_hard_negative_coco.py` | builds a valid COCO dataset of unlabelled tiles from those areas |
| `calibrate_threshold.py` | re-derives the operating threshold after adaptation, on both axes |
| `eval_hard_negatives.py` | measures false-positive suppression directly, on tiles containing no palms |
| `validation_sample.py` | turns a detection map into a citable national estimate |

## Required steps after fine-tuning

1. Re-derive the operating threshold. A fine-tuned model does not
   inherit the old operating point.
2. Measure both axes: false-positive suppression and recall. Standard
   COCO mAP is close to blind to false-positive suppression, because a
   tile with no ground truth contributes no true positives. A model
   that reports fewer false positives while finding fewer palms has not
   improved; it has moved the failure to a place the metric does not
   show.
