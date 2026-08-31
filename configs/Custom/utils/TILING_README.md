# Vector + mosaic → COCO, for every source

Two tools turn georeferenced mosaics and reference polygons into an
MMDetection-ready COCO dataset, at any GSD, with one configuration.

| | |
|---|---|
| `image_vector_to_labelme_pipeline.py` | mosaic + vectors → 512 px tiles + LabelMe JSON |
| `labelme2coco_palm.py` | tiles → COCO, straight into the project layout |

## Why there is no per-sensor table any more

The old script needed `OVERLAP`, `MIN_VISIBLE_AREA_M2` and `BANDS` set by hand
per resolution. Three rules replace it, each either GSD-invariant or derived
from the mosaic itself:

| Setting | Rule | UAV 5 cm | Aerial/GE 15 cm | WV-3 30 cm |
|---|---|---|---|---|
| `TILE_SIZE` | fixed 512 px — matches the crop the network trains on | 26 m | 77 m | 154 m |
| `OVERLAP_FRACTION` | 0.5 of the tile → 256 px at any GSD | 12.8 m | 38.4 m | 76.8 m |
| `MIN_VISIBLE_AREA_PX` | 4 px, converted to m² per mosaic | 0.01 m² | 0.09 m² | 0.36 m² |
| `BANDS = "auto"` | ≥4 bands → `[1,2,3]`; 3 bands → keep all | | | |

A fixed m² floor was never sensor-neutral: 0.5 m² is ~200 px at 5 cm but
~5.5 px at 30 cm, so one constant discarded nothing at one resolution and real
crown fragments at another. A pixel floor means the same thing everywhere.

Adding a source therefore means adding paths to a job file — nothing else.

## Three policies worth understanding before you run

**Empty tiles are training signal, train-split only.**
`KEEP_EMPTY_TILES = {"train": True, "val": False, "test": False}`. A detector
that never sees bare desert or sabkha has no negative evidence for them — the
failure the hard-negative work had to repair after the fact. They must NOT go
in val/test: an empty tile adds no ground truth to COCO mAP, only chances to
score false positives, which silently changes what the metric means relative to
Stages A–C. Pre-flight refuses that configuration.

**Background is capped.** `MAX_EMPTY_FRACTION = 0.30`. With 50% overlap over
mostly bare ground, empties can outnumber palm tiles several times over and
dominate the loss. Candidates are deferred during the sweep and a seeded subset
(`EMPTY_SAMPLE_SEED`) is written once the palm-tile count is known, so the ratio
is exact and reproducible rather than whatever the terrain happened to yield.

**Crowns cut by a tile edge become ignore regions.** `PARTIAL_POLICY = "flag"`.
Dropping them leaves palm pixels in the image with no label, which trains the
model that a visible crown is background. They are written with
`flags.partial`, the converter maps that to `iscrowd=1`, and the Stage D configs
set `ignore_iof_thr=0.5` so the assigner actually honours it.

⚠️ **The chain only works end to end.** Each link silently undoes the previous
one if it is missing:

| Link | Setting | If wrong |
|---|---|---|
| tiler | `PARTIAL_POLICY="flag"` | crowns dropped, taught as background |
| converter | maps flag → `iscrowd=1` | sliver taught as a whole palm → double counting |
| model config | `ignore_iof_thr=0.5` | iscrowd discarded → back to background |
| dataloader | `filter_empty_gt=False` | every background tile thrown away |

`filter_empty_gt` is the dangerous one: the only symptom is a lower image count
in the training log. **Always check it.**

## Run

```bash
# 1. resolved settings per job, nothing written
python configs/Custom/utils/image_vector_to_labelme_pipeline.py \
    --jobs my_jobs.json --out /path/to/tiles --dry-run

# 2. tile
python configs/Custom/utils/image_vector_to_labelme_pipeline.py \
    --jobs my_jobs.json --out /path/to/tiles

# 3. convert each split into the layout the dataset configs read
for S in train val test; do
  python configs/Custom/utils/labelme2coco_palm.py \
      /path/to/tiles/$S/images \
      --dataset-root /workspace/datasets/COCO/Sat_30cm \
      --split-name ${S}_sat \
      --labels configs/Custom/utils/labels.txt
done
```

Step 3 writes `<root>/<split-name>/JPEGImages/` and
`<root>/Annotations/<split-name>.json`, matching
`data_prefix=dict(img='train_sat/')` and
`ann_file='Annotations/train_sat.json'`. Nothing is moved by hand — the step
where a split most easily gets mixed up.

`--set` overrides any upper-case setting without editing the file, the same
idiom as `palm_inference_pipeline.py`:

```bash
--set TILE_SIZE=1024 --set OVERLAP_FRACTION=0.25 --set MAX_EMPTY_FRACTION=None
```

## Multispectral

The tiling and conversion side handles any band count today.

```bash
# tiler: omit "bands" (or set null) in the job file to keep every band
# converter: tif is the only format that can carry more than three
python configs/Custom/utils/labelme2coco_palm.py /path/to/tiles/train/images \
    --dataset-root /workspace/datasets/COCO/WV3_MS --split-name train_ms \
    --image-format tif --labels configs/Custom/utils/labels.txt
```

For a 3-band JPEG from a multispectral source you must state the composite:

```bash
--rgb-bands 5 3 2     # WorldView-3 true colour
--rgb-bands 7 5 3     # NIR-R-G false colour
```

The converter now refuses to guess. WorldView-3's 8-band order is Coastal,
Blue, Green, Yellow, Red, RedEdge, NIR1, NIR2 — so bands 1–3 are **not** RGB,
and the old `img[:, :, :3]` wrote a plausible-looking but wrong composite with
no warning.

### What multispectral costs downstream

The data is the easy half. MMDetection will not read an 8-band tile correctly
without three changes, and each has a price:

| Needed | Why | Cost |
|---|---|---|
| A loader keeping every band | `LoadImageFromFile` → `mmcv.imread` returns 3-channel BGR and silently drops the rest | a custom transform |
| `data_preprocessor` mean/std with N entries | normalisation is per channel | trivial |
| A backbone stem accepting N channels | ImageNet stems are 3-channel | **the real problem** |

That last row is why this is a research decision, not a configuration change.
Every backbone in Stages A–C starts from ImageNet. Widening the stem leaves the
extra channels untrained — the usual remedy is to replicate the RGB filters or
initialise the new channels to the mean, both of which are approximations — and
it makes the WV-3 numbers no longer comparable with the rest of the manuscript,
because a different initialisation is doing part of the work.

### And a sensor caveat specific to WorldView-3

The 8 VNIR bands are acquired at **1.24 m**, not 0.31 m; only the panchromatic
band is 0.31 m. A "30 cm multispectral" product is pan-sharpened, so the
spectral content at crown scale is interpolated from roughly 5 native pixels
across a 6 m crown. NIR would still help separate vegetation from sand — which
is precisely the desert false-positive mode the hard-negative work addressed —
but it will not sharpen crown delineation, and pan-sharpening artefacts are
themselves a confounder.

### Recommendation

Keep Stage D at 3-band pan-sharpened RGB. It answers the feasibility question,
preserves ImageNet initialisation, and stays comparable with Stages A–C.

Multispectral is a strong follow-up — "does NIR remove the desert false
positives that hard-negative mining had to remove by example?" is a good
question with an operational payoff — but it needs its own initialisation
study, and that is the companion paper, not a row in this table.

Generate the 8-band tiles now if the imagery is at hand; they cost one extra
converter run and nothing else, and they will be waiting.

## Numbers to record

`tiling_log.json` holds the full configuration, per-band stretch parameters,
per-split balance and library versions.

For the manuscript, quote **`unique_crowns`**, not the polygon count. 50%
overlap writes each crown up to four times; a reviewer who divides polygons by
tiles gets a dataset that does not exist.

State the empty-tile share and the seed, since both are choices, not
properties of the data.
