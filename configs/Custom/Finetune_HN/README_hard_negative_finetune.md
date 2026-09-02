# Hard-negative fine-tuning: false-positive suppression runbook

*HN = hard negative. The companion runbook,
`README_false_negative_finetune.md`, covers the FN (false-negative,
missed-palm) round.*

Goal: adapt the deployed Stage C Spatial-Mamba-S (best_GE) model to reject
desert false positives (palm-like shrubs, ghaf/acacia) without touching the
benchmark or losing recall. This is an operational deployment model, kept
separate from the Stage C benchmark checkpoints.

## The misconception this resolves
"You cannot train instance segmentation with background" is not true for
Mask R-CNN. On a tile with zero annotations, every RPN anchor and RoI
proposal is an unmatched negative, so the tile is pure "nothing here is a
palm" supervision for the classification heads; box/mask losses just get no
positives. The benchmark's `filter_empty_gt=True` only skips empty tiles;
it is a data choice, not a framework limit. We flip it to `False` on the
hard-negative source only.

## Step 0: Radiometry (nothing to reconcile for GE)
GE_train was NOT contrast-stretched (the tiling pipeline stretched
WorldView-3 only), and the GE country inference pipeline also does not stretch
uint8 imagery. Training, negatives, and inference are therefore all raw uint8
and mutually consistent. Generate the negatives with `--stretch none` (the
default) so they match. Do not stretch GE negatives. (`--stretch match-train`
exists only for a WV-3-style workflow whose positive tiles were stretched.)

## Step 1: Generate hard negatives from selected images (image-selection workflow)
Select tens of 1×1 km images over desert / struggle areas that contain the
palm-like confusers **and no real date palms** (choose areas away from farms;
any real palm in a negative tile teaches the model to miss palms). Then tile
them into 1024×1024 negatives + build the empty-annotation COCO in one step:

```bash
python configs/Custom/Finetune_HN/make_hard_negative_coco.py \
    --from-images /path/to/selected_1km_desert_tiffs \
    --out <coco_root>/HardNeg_GE \
    --tile 1024 --min-coverage 0.5 --stretch none
```

`--from-images` accepts a folder or a single .tif, tiles non-overlapping, drops
tiles that are mostly nodata, and passes uint8 through unchanged (matches
GE_train and GE inference). No shapefile and no labelme2coco: negatives have
no polygons.

Aim for ~2,000 negative tiles spanning the observed confuser types
(bare-desert shrubs, ghaf clusters, irrigation-edge scrub); see "Size the
negative set to the iteration budget" below for where that number comes from.
Diversity of confusers beats raw count.

### Preferred route: mine the tiles the model actually gets wrong
Where manually delineated palm-free AOIs (areas of interest) *and* existing
predictions are available, use `make_aoi_tiles.py` instead of tiling whole
images. Inside a palm-free AOI every predicted crown is a confirmed false
positive, so the predictions rank the tiles and the round trains only on real
confusers:

```bash
python configs/Custom/Finetune_HN/make_aoi_tiles.py \
    --images <ge_imagery>/struggle_areas \
    --aoi    <palm_free_aoi_polygons.shp> \
    --out    <coco_root>/HardNeg_GE_v2 \
    --tile 1024 --overlap 0 --min-aoi-frac 0.9 --aoi-id-field <name_field> \
    --detections <existing_predictions_dir> \
    --min-detections 1 --max-per-aoi 20 \
    --format jpg --jpeg-ref <a train_GE .jpg> \
    --emit-coco empty
```

`--min-detections 1` keeps only tiles containing a false positive;
`--max-per-aoi` spreads the budget over areas and keeps each area's worst
tiles. Run with `--dry-run` first to see how many tiles contain FPs at all.

The codec must match. `train_GE` is JPEG. Lossless GeoTIFF negatives mixed
with JPEG positives let the classifier separate the sources by compression
artefact alone ("clean image = not a palm"), so pass `--format jpg --jpeg-ref`
and the new tiles inherit the corpus quantisation tables exactly.

### Two other input modes
```bash
# Register tiles already cut with another tiler:
python .../make_hard_negative_coco.py --from-tiles /path/tiles --out .../HardNeg_GE

# Cut tiles at detections flagged is_fp=1 in QGIS on the merged master:
python .../make_hard_negative_coco.py --from-detections <merged_master.gpkg> \
    --imagery <ge_imagery> --fp-column is_fp --out .../HardNeg_GE
```

### Safer alternative (if any negatives might contain real palms)
Pure-empty negatives risk teaching the model to miss real palms if a palm slips
into a "negative" tile. If the selected areas are not guaranteed palm-free,
instead LABEL the real palms in those tiles (labelme → labelme2coco) and leave
the shrubs unlabeled: tiles with palms keep them, tiles with only shrubs become
negatives. Set the config's HN source `filter_empty_gt=False` either way. This
cannot induce false negatives and is the more defensible route.

## Step 2: Point the config at the data
Edit the top of
`configs/Custom/5_deployment_finetune/maskrcnn_spatialmamba_s_finetune_hn.py`:
- `GE_ROOT`: the real GE 15 cm COCO root (positives to replay)
- `HN_ROOT`: the `--out` dir from Step 1
- `load_from`: the `best_GE` checkpoint to adapt
- `HN_WEIGHT`: compute it, do not use 0.3 blindly (see below)
- `max_iters`: start `4000`

### HN_WEIGHT is a multiplier on natural size, not a share
`SensorBalancedSamplerN` allocates `quota_s = w_s·N_s / Σ(w_t·N_t) · Σ N_t`, so
the weight scales each source's *existing* size. With GE_train at 19,472
tiles, a small negative set at `HN_WEIGHT = 0.3` is almost invisible: 500
negatives get `0.3·500 / (19472 + 150) = 0.8 %` of batches, so the round would
do essentially nothing. For a target negative share `p`:

```
HN_WEIGHT = (p / (1 - p)) · (N_positives / N_negatives)
```

Example: `p = 0.25`, `N_pos = 19472`, `N_neg = 2500` → `0.333 × 7.79 ≈ 2.6`.
Sane band for `p` is 0.2–0.35. The original `0.3` suggestion only makes
sense when the negative set is itself in the thousands.

### Size the negative set to the iteration budget
`batch_size=1`, so `max_iters` iterations see `max_iters` tiles in total. At
`max_iters = 4000` and `p = 0.25` only 1,000 negative samples are drawn, so
a 9,000-tile negative set would be mostly unused, and each tile seen far
less than once. Match the two:

```
useful N_negatives  ≈  p · max_iters / (times each tile should be seen)
```

At `p = 0.25`, `max_iters = 8000`, 1× coverage → ~2,000 tiles. Prefer fewer,
harder negatives (rank them by the false positives they contain; see
`make_aoi_tiles.py --detections --min-detections`) over a large set the
schedule never reaches.

## Step 3: Fine-tune
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python tools/train.py \
    configs/Custom/5_deployment_finetune/maskrcnn_spatialmamba_s_finetune_hn.py
```
~1–2 GPU-hours. Watch the GE-val line every 1000 iters: **segm mAP@50 must not
drop materially** from the base model's value. The run saves `best_*.pth` on
GE-val segm mAP@50; that best checkpoint is the ship gate. If val mAP falls,
lower `HN_WEIGHT` (recompute with a smaller `p`) or cut `max_iters` to
2000–3000 and re-run.

## Step 4: Validate the fix actually worked (before deploying)
Re-run inference on a struggle folder with the adapted checkpoint and compare
desert FP counts against the original:

```bash
# adapted model, one struggle folder, into a scratch output
python configs/Custom/utils/palm_inference_pipeline.py infer --only <area_id> \
    --set INPUT_PATH='<ge_imagery>' \
    --set OUTPUT_DIR='<results_dir>_HNcheck' \
    --set CHECKPOINT_FILE='work_dirs/Finetune_HN/maskrcnn_spatialmamba_s_finetune_hn/best_coco_segm_mAP_50_iter_XXXX.pth'
```
Open both `<area_id>_palms.gpkg` outputs in QGIS. Success = desert detections
largely gone, farm detections unchanged. Only then point the country-scale run
at the adapted checkpoint.

## Two-round loop (recommended for "once and for all")
The first adaptation kills the obvious confusers; a second, smaller round on
whatever it *still* gets wrong closes the long tail. Re-run inference → flag the
remaining FPs → append to `HardNeg_GE` → re-fine-tune from the adapted (not the
original) checkpoint. Two rounds are normally enough to stabilise.

## Summary of the procedure
The deployed model is adapted to the operational domain by hard-negative
fine-tuning: tiles containing confirmed false positives (desert shrubs and
native trees absent from the training corpus) are added as empty-annotation
images and the model is fine-tuned briefly (a few thousand iterations,
learning rate 1e-5) with replay of the original imagery to preserve recall.
Validation-set mAP is monitored to prevent forgetting; the operating
threshold is re-calibrated after adaptation. Benchmark checkpoints and their
reported metrics are unaffected.

Report the before/after on the stratified validation sample (precision
per density class), not just the raw count; that is the defensible number.
