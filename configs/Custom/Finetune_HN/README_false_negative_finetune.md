# False-negative (missed-palm) fine-tuning — AOI tiling + hard-positive runbook

Goal: recover date palms the deployed **Stage C Spatial-Mamba-S (best_GE)**
model *misses* in specific UAE area types, using new labelled tiles cut from
the AOI extents you delineated — without losing GE performance or the
false-positive gains from the hard-negative round.

Companion to `README_hard_negative_finetune.md`. **They are not
interchangeable** — see the next section before doing anything.

---

## Why this needs a different pipeline than the hard-negative round

| | False positives (HN round) | False negatives (this round) |
|---|---|---|
| Symptom | shrubs/ghaf detected as palms | real palms not detected |
| Training signal | tiles with **zero** annotations | tiles with **complete** annotations |
| Digitising needed | none | yes (or correction of seeded labels) |
| What adapts | classifier heads (backbone freezable) | **features** — backbone must adapt |
| Tooling | `make_hard_negative_coco.py` | `make_aoi_tiles.py` |
| Config | `..._finetune_hn.py` | `..._finetune_fn.py` |

An empty tile cannot fix a missed palm. Feeding unlabelled palms as background
teaches the model to suppress them — it makes recall **worse**. Recall requires
labelled positives from the failing regime.

---

## Step 0 — Diagnose before you annotate (30 minutes, may save days)

Two non-training mechanisms produce "false negatives" in your deployment, and
neither is fixed by retraining. Rule them out first.

**0a. Score threshold.** `palm_inference_pipeline.py` ships `SCORE_THR = 0.35`,
while the model's own `test_cfg` keeps everything above `0.05`. Palms detected
at 0.10–0.34 are *found by the model and discarded by the pipeline*. Re-run one
struggle area at `SCORE_THR = 0.05` and compare:

```bash
python configs/Custom/utils/palm_inference_pipeline.py infer --only UAE_245 \
    --set SCORE_THR=0.05 \
    --set OUTPUT_DIR='/workspace/results/uae_palms_thr005'
```

If most of the missed palms reappear, the fix is **threshold recalibration**
(the pipeline's F1-optimal threshold mode), not fine-tuning. Recalibrate per
area type if the optimum differs between dense farms and sparse desert edges.

**0b. Detection cap.** `test_cfg.rcnn.max_per_img = 300` per 1024 px tile. At
15 cm a tile is ~154 m; at 8 m palm spacing that is ~370 palms. **Dense
plantations can hit the cap**, and the truncation looks exactly like recall
failure. Check whether missed palms cluster in tiles that returned ≈300
detections; if so raise the cap (`--cfg-options
model.test_cfg.rcnn.max_per_img=600`) — no retraining needed.

Only the residual after 0a and 0b is a genuine model failure. Annotate for
*that*.

---

## Step 1 — Extract tiles from your images + AOI shapefiles

`make_aoi_tiles.py` cuts 1024 px tiles restricted to your AOI polygons, on the
**raster's own pixel grid** (integer windows, no resampling) so GSD, radiometry
and crown scale are bit-identical to the imagery the model trained on.

```bash
python configs/Custom/Finetune_HN/make_aoi_tiles.py \
    --images /workspace/datasets/GE15cm/struggle_areas \
    --aoi    /workspace/datasets/GE15cm/struggle_areas/extents \
    --out    /workspace/datasets/COCO/HardPos_GE \
    --tile 1024 --overlap 0.25 --min-aoi-frac 0.5 \
    --aoi-id-field name \
    --seed-labels /workspace/results/uae_palms/UAE_palms_master.gpkg \
    --seed-query "score > 0.30" \
    --val-frac 0.2
```

**Match the codec.** `COCO/GE_15cm/train_GE/JPEGImages` is JPEG. Writing new
tiles as lossless GeoTIFF and mixing them with JPEG positives lets the
classifier separate the two sources by DCT blocking and chroma subsampling
alone — "clean image = not a palm" — instead of learning the content
distinction. Use `--format jpg --jpeg-ref <one training .jpg>`, which copies
that file's exact quantisation tables and subsampling. Georeferencing survives
as a `.jgw` + `.prj` beside each tile, plus `tile_footprints.gpkg`.

**Always `--dry-run` first.** It reports how many tiles each image and each AOI
polygon would produce and writes nothing — the cheapest way to catch a CRS
mismatch or extents that miss half your imagery:

```bash
python configs/Custom/Finetune_HN/make_aoi_tiles.py \
    --images ... --aoi ... --out ... --dry-run
```

**If your AOIs are PALM-FREE areas** (e.g. a `No_Datepalm` extent layer), you
are doing the *hard-negative* round, not this one. Add `--emit-coco empty` and
`--overlap 0`: the tool writes `annotations/hard_neg.json` (0 annotations)
directly, no LabelMe pass is needed, and the output plugs straight into
`HN_ROOT` in `maskrcnn_spatialmamba_s_finetune_hn.py`. See
`README_hard_negative_finetune.md` from there.

**Pairing.** `--aoi` takes either one shapefile for everything, or a folder of
per-image shapefiles matched by filename stem (`UAE_245.tif` ↔ `UAE_245.shp`).
AOI CRS is reprojected into each raster's CRS automatically.

**`--seed-labels` is the flag that matters.** It clips your existing predicted
crowns into per-tile LabelMe sidecars, so annotation becomes *correction* — add
the crowns the model missed, delete the ones it invented — instead of
digitising from scratch. Typically 5–10× faster, and it focuses your attention
precisely on the errors you are trying to fix.

**`--val-frac 0.2` holds out whole AOIs, not individual tiles.** With 25%
overlap, adjacent tiles share pixels; a tile-level split would leak and inflate
the recall gain you are about to report. Splitting by AOI is the honest version.

Key knobs:

| Flag | Default | Note |
|---|---|---|
| `--tile` | 1024 | must match the training tile size — do not change |
| `--overlap` | 0.25 | some overlap stops crowns being cut at every edge |
| `--min-aoi-frac` | 0.5 | lower to ~0.2 if your extents are small |
| `--min-coverage` | 0.5 | drops mostly-nodata tiles |
| `--stretch` | `none` | **keep `none` for GE** (matches GE_train + inference) |
| `--max-tiles` | 0 | deterministic cap when the AOIs give more than you can label |

Outputs:

```
HardPos_GE/
├── images_train/   *.tif + *.json (LabelMe sidecars)
├── images_val/     *.tif + *.json      <- whole held-out AOIs
├── tile_manifest.csv         per-tile source image, AOI, aoi_frac, valid_frac
├── tile_footprints.gpkg      LOAD THIS IN QGIS to vet coverage before labelling
└── tiling_provenance.json    every parameter used
```

**Vet before labelling.** Open `tile_footprints.gpkg` over your imagery in
QGIS. Confirm the tiles land where you intended and that AOIs you care about
actually produced tiles. Fixing coverage now costs minutes; after labelling it
costs days.

---

## Step 2 — Annotate (the step that decides whether this works)

Open `images_train/` and `images_val/` in LabelMe. The sidecars are pre-filled
if you used `--seed-labels`.

**The one hard rule: within every tile you keep, EVERY palm must be labelled.**
A partially annotated tile trains the model to suppress the palms you skipped —
the exact failure you are removing. If a tile is too dense or ambiguous to
finish, **delete the tile and its .json**; a smaller clean set beats a larger
dirty one.

Tiles that genuinely contain zero palms are fine and useful — the config keeps
them (`filter_empty_gt=False`) as in-domain negatives.

Target volume: **200–600 tiles** spanning every failure mode you observed
(young/small crowns, dense overlapping canopy, shadow, dust haze, mixed
orchards, atypical mosaic tone). As with the negatives, **diversity beats
count**.

Then run the project's usual conversion on each folder:

```bash
# labelme2coco -> the two json names the config expects
#   HardPos_GE/annotations/train_hardpos.json   (from images_train)
#   HardPos_GE/annotations/val_hardpos.json     (from images_val)
```

Sanity-check before training:

```bash
python -c "
import json
for s in ('train','val'):
    d=json.load(open(f'/workspace/datasets/COCO/HardPos_GE/annotations/{s}_hardpos.json'))
    n=len(d['images']); a=len(d['annotations'])
    print(f'{s}: {n} images, {a} anns, {a/max(n,1):.1f} palms/tile')
"
```

If `palms/tile` is far below what you see in the imagery, annotation is
incomplete — stop and finish it. Also run
`configs/Custom/utils/clean_coco_degenerate.py` on both files: sub-pixel boxes
are what produced the `NaN loss_bbox` in Stage D.

---

## Step 3 — Set the two knobs in the config

Edit the top of
`configs/Custom/maskrcnn_palm_finetune_hn/maskrcnn_spatialmamba_s_finetune_fn.py`:

- `GE_ROOT` — your real GE 15 cm COCO root (positives to replay)
- `HP_ROOT` — the `--out` dir from Step 1
- `load_from` — **the HN-adapted checkpoint if you already ran the FP round**
  (chaining preserves the false-positive gains); otherwise Stage C `best_GE`
- `HP_WEIGHT` — see below
- `max_iters` — start `6000`

**`HP_WEIGHT` is not a share, it is a multiplier.** `SensorBalancedSamplerN`
computes `quota_s = w_s·N_s / Σ(w_t·N_t) · Σ N_t`, so weights `[1, 1]` on a
250-tile HP set against 2500 GE tiles gives the new data only ~9% of batches —
far too dilute to move recall. For a target share `p`:

```
HP_WEIGHT = (p / (1 - p)) · (N_GE / N_HP)
```

Example: `p = 0.35`, `N_GE = 2500`, `N_HP = 250` → `0.538 × 10 ≈ 5.4`.

Read `N_GE` and `N_HP` off `len(images)` in the two COCO files. Stay in the
**0.3–0.4** band for `p`: above ~0.5 you start over-fitting a few hundred tiles.

---

## Step 4 — Fine-tune

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/train.py \
    configs/Custom/maskrcnn_palm_finetune_hn/maskrcnn_spatialmamba_s_finetune_fn.py
```

~2–3 GPU-hours at 6k iters. **Confirm the realised source mix in the first
log lines** (the sampler prints per-source quotas) before letting it run long.

Validation runs every 1000 iters on **two** sources and you watch both:

- `GE/coco/segm_mAP_50` — the forgetting monitor. Must not drop materially.
- `HPOS/coco/segm_mAP_50` — the held-out AOIs. This is the number that has to
  go **up**, or the round did not work.

Three checkpoints are saved so the trade-off is inspectable:

| Checkpoint | Selected on | Use |
|---|---|---|
| `best_mean_segm_mAP_50_*` | mean of GE + HPOS | **the ship gate** |
| `best_GE_*` | GE only | least forgetting |
| `best_HPOS_*` | new AOIs only | most recall recovered |

Deploy the **mean** one unless you have a reason not to: it cannot buy recall
by wrecking GE, and it cannot reject the improvement you were seeking.

If `HPOS` mAP barely moves: the tiles probably do not actually contain the
failure mode (re-check Step 0 — you may have been chasing a threshold problem),
or `HP_WEIGHT` is too low, or annotation is incomplete. If `GE` mAP drops:
lower `HP_WEIGHT`, or shorten to 3–4k iters.

Note this config deliberately **does not freeze the backbone**, unlike the HN
config's optional `frozen_stages=4`. Recall in a new appearance regime is a
feature problem, not a decision-boundary problem, so the backbone trains at
`lr_mult=0.1`.

---

## Step 5 — Validate operationally, then recalibrate the threshold

```bash
python configs/Custom/utils/palm_inference_pipeline.py infer --only UAE_245 \
    --set INPUT_PATH='/workspace/datasets/GE15cm' \
    --set OUTPUT_DIR='/workspace/results/uae_palms_FNcheck' \
    --set CHECKPOINT_FILE='work_dirs/Finetune_HN/maskrcnn_spatialmamba_s_finetune_fn/best_mean_segm_mAP_50_iter_XXXX.pth'
```

Open the before/after `.gpkg` pair in QGIS on a struggle area **that was not in
any AOI** — improvement on areas you trained from proves nothing.

Then **re-derive the F1-optimal `SCORE_THR`** (the pipeline's calibration
mode): the score distribution shifts after adaptation, so the old 0.35 is no
longer the operating point. Report precision/recall per density class on your
stratified validation sample, not raw counts.

---

## Ordering when you have both failure modes

Run them as two chained rounds, validating each:

1. **HN round** (`..._finetune_hn.py`) from Stage C `best_GE` → kills the
   desert false positives.
2. **FN round** (`..._finetune_fn.py`) with `load_from` = the HN best → recovers
   the missed palms while GE-val replay holds the FP gains in place.
3. Re-check desert FP counts after step 2. If some returned, append those tiles
   to `HardNeg_GE` and repeat the HN round from the FN checkpoint. Two passes
   normally stabilise.

Do **not** try to fix both in one run with a three-source config until each
round is separately validated — when something regresses you will not know
which source caused it.

---

## For the manuscript (one paragraph, Methods/Discussion)

> The deployed model was adapted to the operational domain in two brief
> hard-example rounds. First, N₁ tiles containing confirmed false positives
> (desert shrubs and native trees absent from the training corpus) were added
> as empty-annotation images. Second, N₂ exhaustively annotated tiles were cut
> from M delineated areas of interest in which the model under-detected, and
> used as hard positives. Both rounds fine-tuned briefly (6k iterations,
> learning rate 1e-5, backbone learning-rate multiplier 0.1) with replay of the
> original Google Earth training imagery to prevent forgetting. Checkpoints
> were selected on the mean of held-out Google Earth validation mAP@0.5 and a
> spatially disjoint held-out subset of the new areas of interest, and the
> operating threshold was re-calibrated after adaptation. Benchmark checkpoints
> and their reported metrics are unaffected.

Report the before/after on the **stratified validation sample** (recall per
density class), not the raw detection count — that is the defensible number.
