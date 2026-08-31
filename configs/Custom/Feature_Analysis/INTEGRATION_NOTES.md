# Feature-Analysis Suite v2.1 — Deep Multiscale Extension

**Date:** 2026-07-06
**Adds:** `deep` phase (CPU), `anatomy` phase (GPU), `multiscale.py` module
**Modifies:** `cli.py` (replaced), `analysis.py` (one-line patch, below)

---

## 1. Installation

Copy into `configs/Custom/Feature_Analysis/feature_analysis/`:

```
multiscale.py      # new module
cli.py             # replaces the existing cli.py
```

Then apply one edit to `analysis.py` inside `phase_extract` so that the
feature-grid shape is recorded in the location index (required to map COCO
ground truth onto cached feature locations). Replace:

```python
loc_meta[key] = {"hw": hw, "idx": idx.tolist(),
                 "channels": int(c)}
```

with:

```python
loc_meta[key] = {"hw": hw, "idx": idx.tolist(),
                 "channels": int(c), "shape": [int(h), int(w)]}
```

**Backward compatibility.** An existing cache written before this patch still
works: for square feature grids (the 1024×1024 tiles yield square grids at
every stride) the shape is reconstructed as sqrt(hw). Re-running `extract` is
therefore *not* required. Non-square grids without a recorded shape are
skipped with a warning.

No changes to `config_feature_analysis.json` are required. The new phases
reuse the existing fields: `tile_sets[*].coco_ann` (separability labels),
`vis_models` / `showcase_tiles` / `vis_levels` / `vis_source` (anatomy
figure), and `fpn_levels` / `n_tiles` / `bootstrap` (all quantitative
analyses).

## 2. New phases

```bash
# CPU-only; consumes the extract cache; safe to iterate freely
bash configs/Custom/Feature_Analysis/run_feature_analysis.sh deep

# GPU; one detector resident at a time; uses showcase_tiles + vis_models
bash configs/Custom/Feature_Analysis/run_feature_analysis.sh anatomy
```

`run` now executes: extract → analyze → deep → visualize → composite →
anatomy.

## 3. New outputs

```
<out_dir>/
├── tables/
│   ├── separability.csv          # model, resolution, level, d', CI, n_tiles
│   ├── separability_summary.csv  # per model x resolution: peak level and peak d'
│   ├── level_energy.csv          # model, resolution, level, share, CI
│   ├── spectral_decay.csv        # model, resolution, level, alpha
│   └── interlevel_bures.csv      # model, resolution, level_a, level_b, fidelity
└── figures/
    ├── separability_by_level.{pdf,png}
    ├── level_energy_profile.{pdf,png}
    ├── eigenspectra_<res>.{pdf,png}
    ├── interlevel_similarity.{pdf,png}
    └── anatomy/anatomy_<source>_<res>_<stem>.{pdf,png}
```

`deep_provenance.json` (alongside the existing `provenance.json`, which is not
overwritten) records the suite version, seed, bootstrap count, configuration
hash, torch/mmdet versions, and the separability thresholds (coverage 0.50 /
0.05, min 20 locations per class, 1% shrinkage) for the Methods
reproducibility statement.

## 4. What each result argues in the manuscript

| Output | Quantity | Manuscript role |
|---|---|---|
| `separability_by_level` | Shrinkage-regularised Mahalanobis d′ between crown and background feature locations, per FPN level and resolution | Localises *where in the pyramid* crown evidence is linearly accessible; the peak level is expected to track crown size in pixels, shifting from P2/P3 at UAV 5 cm toward P3/P4 as GSD coarsens. Family differences in how gracefully the peak shifts constitute a direct, GT-anchored mechanism for the cross-resolution ranking. |
| `level_energy_profile` | Per-location activation-energy share per level | The scale-allocation "fingerprint" of each backbone. A backbone whose profile re-allocates with GSD adapts its computation to object scale; a static profile under coarsening is the representational signature of zero-shot degradation. |
| `eigenspectra_*` + `spectral_decay.csv` | Eigenvalue spectra of the pooled channel covariance and the power-law exponent α (ranks 5–100) | Dimensionality/expressivity axis. Read against the Bures invariance ranking to demonstrate (or refute) an invariance–expressivity trade-off between families — a stronger claim than either metric alone. |
| `interlevel_similarity` | Bures fidelity between the covariances of FPN level pairs, per model | Distinguishes a *redundant* pyramid (high off-diagonal similarity; levels encode near-identical statistics) from a *scale-specialised* one. Bures is used because CKA requires location-matched samples, which do not exist across strides. |
| `anatomy/*` | Composite: input + GT crown outlines · P2–P5 PCA-to-RGB with activation-magnitude contours · activation-share bars | The section's hero figure. One panel simultaneously shows what each family encodes at each scale, where it concentrates activation relative to ground truth, and how it budgets the pyramid. |

## 5. Methodological notes for §Methods

- **Labelling.** GT instance masks are rasterised at native tile resolution,
  area-averaged onto each feature grid (PIL BOX resampling), and thresholded:
  coverage ≥ 0.50 → crown, ≤ 0.05 → background, intermediate coverage
  excluded to suppress boundary label noise. Labels are taken at exactly the
  cached subsampled locations, so separability is computed on the identical
  samples used for CKA and invariance.
- **Separability estimator.** d′ = √((μ₁−μ₀)ᵀ Σ_w⁻¹ (μ₁−μ₀)) with the pooled
  within-class covariance ridged by 1% of its mean variance
  (256-dimensional; guards against ill-conditioning at coarse levels where
  crown samples are few). Tiles with fewer than 20 locations in either class
  are excluded; `n_tiles` in the table records the effective sample.
- **Energy statistic.** Per-location mean squared L2 norm, normalised across
  levels per tile. The per-location (rather than total) statistic removes the
  4× spatial-count imbalance between adjacent strides, making shares
  comparable across the pyramid.
- **Spectra.** Per-tile channel covariances are trace-normalised before
  averaging so that no single high-energy tile dominates; α is the negative
  log–log slope over eigenvalue ranks 5–100 (Stringer et al., 2019, Nature).
- **All CIs** are 95% percentile bootstrap over tiles (default 1,000
  resamples), consistent with the existing analyses.

## 6. Verification checklist (first run)

```bash
# deep
head -5 <out_dir>/tables/separability.csv
# d' should be clearly > 0 at fine levels for UAV_5cm and decline toward P5;
# n_tiles should equal ~n_tiles for fine levels, possibly fewer at P5.

python - << 'EOF'
import csv
rows = list(csv.DictReader(open('<out_dir>/tables/level_energy.csv')))
# shares must sum to ~1 within each (model, resolution)
EOF

# anatomy
# GT outlines must align with crowns in column 1; if outlines are absent,
# the tile stem was not found in the tile set's coco_ann (check filenames).
```

## 7. Robustness behaviour (verified)

- **Legacy caches** written before the `shape` patch are handled by a
  square-grid fallback; re-running `extract` is unnecessary for 1024x1024
  tiles.
- A **tile set without `coco_ann`** is skipped for separability (warning
  logged); energy, spectra, and inter-level similarity are still computed.
- **Missing cached matrices** (interrupted extraction) reduce `n_tiles` for
  the affected cells without aborting the phase.
- **`vis_models` labels matching no configured model** degrade the
  inter-level figure to the first three models with a warning; all CSVs
  always cover the full model set.
- Running `deep` before `extract` fails immediately with an actionable
  message naming the missing cache artefact.
- Magnitude-contour geometry in the anatomy figure was verified against
  `imshow` coordinates (array-index contours align under the inverted image
  axis; no origin correction is required or applied).

## 8. Known limitations / options

- RLE-encoded segmentations require `pycocotools` (present in the MMDet
  environment); polygon segmentations need only PIL.
- The anatomy figure draws polygon outlines only; RLE instances appear in
  the separability labels but not as outlines.
- If a Stage C→D contrast at the representation level is wanted later, add
  the Stage D checkpoints to `models` and re-run `deep` on a WV-3-only
  config; the separability and energy tables then quantify what fine-tuning
  recovers, complementing the qualitative composite already planned.
