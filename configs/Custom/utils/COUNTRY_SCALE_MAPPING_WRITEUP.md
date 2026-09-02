# Country-Scale Date-Palm Mapping: Write-Up Notes

> The parameters recorded here describe the single-file pipeline
> `palm_inference_pipeline.py` in this folder. The command recorded in the
> project handover as the one run for the national inventory used the
> `palm_inference/` package with tile 1024, overlap 256 and threshold 0.30
> (see the repository README, "Inference with the deployed model").

Historical write-up notes for the operational UAE-wide mapping run, kept as
the record of the deployment model, the hard-negative adaptation, the
inference pipeline, and the limitations that must be disclosed alongside any
use of the map. Every number here is measured, not estimated.

Companion sources in this repo:
- `configs/Custom/utils/palm_inference_pipeline.py`: the pipeline itself; its
  `CONFIG` block is the authoritative record of the deployed parameters.
- `configs/Custom/Finetune_HN/`: mining, evaluation and threshold calibration.
- `configs/Custom/5_deployment_finetune/maskrcnn_spatialmamba_s_deploy.py`:
  the deploy config (separate from any evaluated benchmark config).

---

## 1. Deployment model, and why it is not a benchmark model

This distinction comes first: the mapping did not use one of the benchmark
models, and the metrics differ accordingly.

The benchmark checkpoints and their reported metrics are untouched. Country-scale
mapping uses a separate deployment model: the Stage C unified multi-source
Spatial-Mamba-S `best_GE` checkpoint, adapted by hard-negative fine-tuning, run
through a separate deploy config. The only architectural difference is
`test_cfg.rcnn.max_per_img` raised from 300 to 1000 (raised again to 2000
for the second national run; see the deploy config header), because a 1024 px GE
tile over a dense plantation can hold more than 300 crowns and the benchmark cap
would truncate the detections without raising an error. Keeping this in a
deploy config rather than editing the evaluated config is what preserves the
published numbers.

---

## 2. Hard-negative adaptation (the "fine-tuning approach")

### Problem

Stage C is trained on annotated farmland. The UAE is mostly desert,
and at inference the model fired on palm-like shrubs, native *Prosopis*
(ghaf, *Prosopis cineraria*) / *Ziziphus*, irrigation structures and
shadow-edge texture, error modes that no
validation split containing only labelled farmland can see. Measured on 2,098
palm-free desert tiles the model had never trained on:
4.158 false positives per tile at score 0.35, with 90.9% of tiles carrying at
least one.

### Method

Mine tiles from unlabelled desert AOIs, run the Stage C model, keep
tiles where it fires; these become training images with zero annotations
(`filter_empty_gt=False`, so they are not discarded by the dataset filter),
mixed with the labelled positives. Checkpoint selection is the critical detail:
the run's `save_best` selects on GE validation mAP, which is blind to this
failure mode entirely, so the deployed checkpoint (iteration 4000) was chosen
by directly measuring false positives on held-out desert tiles.

### Result

At score 0.35: 4.158 → 0.048 FP/tile (−98.9%); tiles with ≥1
false positive 90.9% → 1.6%; GE validation segm mAP@50 unchanged at
0.948. The suppression cost no recall on labelled palms; that unchanged mAP is
what makes the claim publishable rather than a precision/recall trade.

### The negative result

Report it. A second mining round on 5,451 negatives (round
1's 3,083 plus 2,368 more) was measurably worse at every threshold.
Explanation worth one sentence: with the negative share held fixed, enlarging the
pool draws the hardest tiles less often and the easier additions dilute the
signal. More hard negatives is not monotonically better, and this is a useful
finding for anyone repeating the procedure.

### Operating point

Re-derived for the deployed checkpoint on both axes, because
F1 on labelled farmland is blind to the desert error mode that dominates by area:

| score | precision | recall | F1 | FP per palm-free tile |
|-------|-----------|--------|--------|-----------------------|
| 0.25  | 0.9236    | 0.9241 | 0.9239 | 0.071 |
| **0.30** | **0.9334** | **0.9154** | **0.9243** | **0.055** |
| 0.35  | 0.9409    | 0.9062 | 0.9232 | 0.048 |

The 0.30 threshold was adopted: it is the F1 optimum, and gives +0.9 points
recall over the inherited 0.35 at negligible false-positive cost. 0.25 buys
another 0.9 points of recall but raises false positives by roughly half again,
so it would be a deliberate choice, not a default.
Re-derive with `configs/Custom/Finetune_HN/calibrate_threshold.py` whenever the
checkpoint changes.

---

## 3. Inference pipeline

Four aspects of the pipeline determine correctness at national scale.

### (a) Processing units and virtual mosaics
The national GE 15 cm archive is organised as ~250 folders of ~100 GeoTIFFs, each
folder covering a 20 km × 20 km block. Each folder is processed as one
virtual mosaic via windowed reads across its constituent rasters, with no VRT
and no full-raster load. This prevents crowns on internal tiff seams from being
split or double-counted. Grid validation is enforced, never assumed: mixed CRS,
mismatched pixel size and rotated transforms are rejected; sub-pixel offsets
(GEE exports carry arbitrary fractional offsets; 0.34 and 0.49 px observed)
are absorbed by snapping each tiff to the nearest integer pixel of the folder
grid, worst-case error half a pixel = 7.5 cm at 15 cm GSD, negligible against
3–8 m crowns.

### (b) Tiling and seam handling
Tiles are 1024 px, matching the training tile size so crowns appear at the
pixel scale the mask head learned. Overlap is derived per unit from the metre
GSD and the maximum expected crown (12.0 m × 1.15 safety), so the same code is
correct on 5 / 15 / 30 cm imagery: at the measured 14.9 cm/px this gives
93 px ≈ 13.9 m. Each tile owns an interior box; detections falling outside
the ownership box are dropped, so overlap costs nothing in duplicates.

### (c) Three-tier de-duplication
1. Tile ownership: interior-box assignment inside a unit.
2. Per-unit polygon-IoU NMS at 0.45 over the ownership survivors.
3. Cross-unit centroid NMS at merge time: suppression radius 3.0 m,
   restricted to detections whose centroid lies within 20 m of a unit
   boundary, and only pairs from *different* units may suppress each other, so
   genuine close neighbours inside a plantation are never eaten.

### (d) Output geometry and attributes
Each crown is written as an area-preserving circle at the mask centroid, with
equivalent diameter in metres. A circularity filter (≥ 0.60) is computed on
the *cleaned mask*, before simplification. Output is one GeoPackage plus one
stats JSON per unit in EPSG:32640 (UTM 40N for the whole country, to keep the
master seamless), written atomically (tmp file + rename, JSON sidecar last) so
`RESUME` trusts only completed units. The merge step produces a national master
with a neighbour-density attribute (count of other detected palms within
50 m), which lets the analysis separate dense cultivated plantations from
isolated desert detections where residual false positives concentrate.

### (e) Two-workstation partitioning
The workload was split across two workstations by exclusive ownership of
disjoint folder sets (each machine restricted to its own set via `--only`),
with deterministic `--shard K/N` available as the alternative.
Because the units are disjoint, the merge step's cross-unit NMS handles the
boundaries identically whether the two neighbours were computed on the same
machine or not.

Cross-machine parity was checked directly. On a
block with byte-identical inputs on both machines (UAE_374): 18,135 vs 18,136
detections, a difference of 0.006%, with identical diameter percentiles. On a
block whose Google Earth tiles had been downloaded at different dates
(UAE_61): 411,763 vs 411,954, or 0.046%. The first quantifies floating-point /
CUDA nondeterminism; the second quantifies imagery vintage. The two differing
by an order of magnitude is the evidence that the pipeline itself is
reproducible.

### (f) Other robustness properties worth a half-sentence
Geographic CRS handled (metre GSD derived from latitude, never "metres" from
degree pixels); >3-band, alpha and non-uint8 rasters handled with a single
per-unit contrast stretch computed from decimated overviews; nodata / uncovered
tiles skipped without GPU cost; AMP float16 inference with FP32 retry on error;
CUDA-OOM fallback retries a spiking batch tile-by-tile; a crash on one folder
does not abort the batch.

---

## 4. Limitations: disclose explicitly

1. The national total is a composite, not a single-epoch snapshot. Google
   Earth basemap tiles were acquired at different dates across the country;
   adjacent blocks can differ by years. The map is "date palms as visible in the
   most recent available imagery per block", and the count cannot be attributed
   to a single reference date.
2. Validation is not spatially stratified over the deployment domain. The
   operating point was derived on the GE validation split (farmland) and
   cross-checked on mined desert tiles. There is no probability-sample-based
   national accuracy estimate, so the reported P/R characterise the validation
   domain, not the country.
3. Benchmark val/test splits contain no background tiles (background tiles
   are train-only, capped at 30%). Reported precision on those splits is
   therefore optimistic relative to deployment, where most tiles contain no
   palms. This is exactly the gap the hard-negative adaptation addresses, and the
   reason the desert-tile FP rate is reported alongside mAP.
4. Residual false positives are spatially non-random, concentrating in open
   desert on isolated palm-like woody vegetation. The neighbour-density attribute
   is provided so users can filter, but no automatic gate is applied; isolated
   detections should be treated as lower-confidence than plantation detections.
5. Recall is bounded below on very small or heavily overlapping crowns, and
   the circle geometry deliberately discards crown shape, which is appropriate
   for counting and canopy-area estimation, not for morphology studies.
6. The operating point is checkpoint-specific. Any change of checkpoint
   (including the WorldView-3 models) requires re-deriving `SCORE_THR`; and no
   WorldView-3 desert false-positive gate exists yet, so 30 cm results must not
   be converted into area totals until one is built.

---

## Appendix A: §2.3 Crown Annotations and Reference Data (companion notes)

A scope caveat first: the tiling and COCO-conversion tools were rebuilt for
Stage D. Stages A–C used the earlier pipeline, which dropped edge-cut crowns,
kept no background tiles, and used the stock converter with `iscrowd=0`
hardcoded. §2.3 must therefore be scoped per dataset, or Stages A–C rebuilt.

Most important items to report (Stage D build):

> The Stage D corpus counts in items 2–3
> (37,066 / 63,953 / 149,903 unique-crown and polygon figures; 7,233 / 596
> iscrowd figures) predate the final ground-truth regeneration of
> 4 Aug 2026. The regenerated counts are carried by
> `configs/Custom/4_satellite_wv3_30cm/STAGE_D_README.md` and are the ones
> to quote.

1. Spatially disjoint splits by region polygon; tiles straddling a region
   boundary are discarded (`STRADDLE_TOLERANCE = 0.0`).
2. 50% training overlap; val/test non-overlapping. Therefore quote *unique
   crowns* (37,066 train; 63,953 total), not polygon instances (149,903).
3. Edge-cut crowns become `iscrowd=1` ignore regions (7,233 train / 596 val
   / 741 test, 4.8%), consumed by `MaxIoUAssigner.ignore_iof_thr = 0.5`.
4. Background tiles are train-only, 1,090 tiles at exactly the 30% cap, seed
   20260804; none in val/test.
5. GSD-invariant sliver floor of 4 px = 0.36 m² at 30 cm, 0.09 m² at 15 cm,
   0.01 m² at 5 cm.
6. Percentile (p2/p98) contrast stretch pooled per sensor group. The WV-3 MS
   float32 and uint16 mosaics pool correctly (p98 ratio 1.16×, same DN scale).

Second tier: EPSG:3857 → 32640 reprojection; multi-part crown merging (351
merges); `pycocotools.frPyObjects` rasterisation matching COCOeval; 50% tile
validity threshold; deterministic file ordering and fixed seeds.

Also note: Stage C's non-uniform `accumulative_counts` (2 for
three backbones, 4 for Spatial-Mamba-S); Spatial-Mamba's different anchor set;
and that val/test contain no background tiles.

Authoritative per-build values can be re-extracted from each build's
`tiling_log.json`.
