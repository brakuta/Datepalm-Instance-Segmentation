# Feature_Analysis

Representation-level analysis and manuscript figures for the date palm
multi-resolution benchmark (experiments 1–4, internally Stages A–D). The
suite quantifies and visualises how convolutional (CNN), Transformer, and
state-space (Mamba) backbones encode date palm crowns across the 5–30 cm
resolution continuum, and produces the evidence behind the
cross-resolution generalisation claim beyond aggregate mAP.

The detector is never modified: all features are captured through PyTorch
forward hooks on `model.backbone` and `model.neck`.

---

## 1. What it produces

Quantitative, over the entire sampled test set, with bootstrap confidence
intervals:

- Resolution invariance (the headline result). Per backbone, the channel
  covariance of an FPN level is estimated per resolution and compared
  across the continuum by trace-normalised Bures fidelity, a
  scale-invariant, registration-free descriptor. It can optionally be
  correlated (Spearman) with a supplied cross-resolution mAP gap to yield
  an explanatory result.
- Debiased linear CKA (centred kernel alignment) between backbones at
  matched locations (unbiased HSIC, the Hilbert–Schmidt independence
  criterion; optional RBF variant).
- Representational dimensionality: effective rank and participation ratio.
- Crown–background separability (`deep` phase): shrinkage-regularised
  Mahalanobis d′ between crown and background feature locations, per FPN
  level and resolution. This localises where in the pyramid crown
  evidence is linearly accessible.
- Level activation-energy share, eigenspectrum decay, inter-level
  similarity (`deep` phase): each backbone's scale-allocation
  fingerprint, its dimensionality under coarsening, and whether its
  pyramid is redundant or scale-specialised.

Qualitative, on selected representative tiles:

- Multiscale grids: rows = backbone families, columns = pyramid levels
  P2–P5 (or backbone stages C2–C5); PCA-to-RGB structure and activation-
  magnitude overlay.
- Composite figure recipes: rows = backbones (or backbone × stage),
  columns = an isolated axis (e.g. UAV/Aerial/WV-3 resolution, or a single
  tile degraded to coarser ground sampling distances).

All figures are written as vector PDF (for the manuscript) and 300-dpi
PNG, with a colourblind-safe Okabe–Ito family palette and editable
embedded fonts.

---

## 2. Layout

```
configs/Custom/Feature_Analysis/
├── README.md
├── run_feature_analysis.py          # launcher (sets sys.path, dispatches to cli)
├── make_stagec_config.py            # fill the config from a Stage C work tree
├── pick_showcase_tiles.py           # nominate median-density showcase tiles
├── make_qualitative_figures.py      # standalone mosaic / GSD-ladder figures
└── feature_analysis/                # package
    ├── __init__.py
    ├── cli.py            # argparse + dispatch
    ├── config.py         # schema, validation, example config
    ├── util.py           # logging, seeds, bootstrap, publication style
    ├── metrics.py        # CKA, Bures fidelity, effective rank, participation ratio
    ├── extractor.py      # hook-based feature capture (model intact)
    ├── sampling.py       # tile sampling/stratification + cache I/O
    ├── analysis.py       # extract/analyze/deep phases + tables/figures
    ├── multiscale.py     # deep + anatomy phases (separability, level energy)
    └── visualization.py  # visualize/composite phases
```

The config file (`config_feature_analysis.json`) is generated, not
shipped: step 1 below writes an annotated example, which you then edit.

## 3. Requirements

`torch`, `mmdet` (v3.x), `mmengine`, `numpy`, `scipy`, `matplotlib`, `Pillow`.
Most are already present in the training environment; if needed:

```bash
pip install scipy matplotlib pillow   # add --break-system-packages only outside a virtual environment
```

## 4. Usage

Run from the repository root (so `configs/...` paths resolve), with the
experiment 3 (Stage C) checkpoints available. They are not distributed
with this repository (see `WITHHELD.md`); every config needed to retrain
them is here.

```bash
# 1. generate the annotated example config, then edit it (models, tile_sets,
#    showcase/composite tiles, map_gap_csv). make_stagec_config.py can fill
#    the model entries from a Stage C work tree.
python configs/Custom/Feature_Analysis/run_feature_analysis.py make-config \
    --out config_feature_analysis.json

# 2. run all phases:
python configs/Custom/Feature_Analysis/run_feature_analysis.py run \
    --config config_feature_analysis.json

# or individual phases (same --config flag on each):
#   extract    GPU: build each model once, cache subsampled features
#   analyze    CPU: CKA, resolution invariance, dimensionality
#   deep       CPU: separability, level energy, spectra, inter-level similarity
#   visualize  GPU: multiscale feature grids
#   composite  GPU: composite figure recipes
#   anatomy    GPU: scale-anatomy summary figure
```

The GPU/CPU phase split lets the qualitative panels be re-rendered without
repeating extraction. `extract`, `analyze` and `deep` consume the whole
sampled test set; `visualize`, `composite` and `anatomy` use only the
nominated tiles. Running `deep` before `extract` fails immediately with a
message naming the missing cache artefact.

## 5. Key configuration fields

| Field | Meaning |
|---|---|
| `models` | per backbone: `label`, `family` (CNN/Transformer/Mamba), `config`, `checkpoint` |
| `tile_sets` | per resolution: `img_dir`, `pattern`, optional `coco_ann` (palm-count stratification) |
| `fpn_levels` | FPN indices for the quantitative analyses (0=P2 …) |
| `n_tiles` | sampled tiles per resolution group |
| `map_gap_csv` | CSV `label,map_gap`; enables the invariance↔accuracy correlation |
| `vis_models`, `vis_levels`, `vis_source` | multiscale grid: subset of backbones, levels, `neck`/`backbone` |
| `showcase_tiles` | resolution → tile paths for multiscale grids |
| `composite_figures` | list of figure recipes (see below) |

A composite recipe declares a row axis (`models`) and a column axis, either
explicit per-column tiles (`columns: [{label, tile}, …]`) or one `source_tile`
with `degrade_factors` (e.g. `[1,3,6]` for 5/15/30 cm) for the controlled-GSD
supplement. `level`, `source` (`neck`/`backbone`) and `view`
(`pcargb`/`magnitude`/`both`) are per-recipe.

## 6. Outputs

```
<out_dir>/
├── figures/
│   ├── resolution_invariance.{pdf,png}     # headline ranking
│   ├── invariance_vs_mapgap.{pdf,png}      # explanatory scatter
│   ├── cka_<res>__L<level>.{pdf,png}       # cross-architecture similarity
│   ├── separability_by_level.{pdf,png}     # deep: d' per level and resolution
│   ├── level_energy_profile.{pdf,png}      # deep: scale-allocation fingerprint
│   ├── eigenspectra_<res>.{pdf,png}        # deep: covariance spectra
│   ├── interlevel_similarity.{pdf,png}     # deep: pyramid redundancy
│   ├── multiscale/                         # families × pyramid levels
│   ├── composite/                          # isolated-axis recipes
│   └── anatomy/                            # summary figure: input + GT + P2-P5
├── tables/                                 # CSV for every metric
├── provenance.json                         # seed, config hash, package versions
└── deep_provenance.json                    # deep phase: version, thresholds
```

### Deep-phase method notes

- Labelling. Ground-truth instance masks are rasterised at native tile
  resolution, area-averaged onto each feature grid, and thresholded:
  coverage ≥ 0.50 → crown, ≤ 0.05 → background; intermediate coverage is
  excluded to suppress boundary label noise. Labels are taken at exactly
  the cached subsampled locations, so separability is computed on the same
  samples as CKA and invariance.
- Separability estimator. d′ = √((μ₁−μ₀)ᵀ Σ_w⁻¹ (μ₁−μ₀)) with the
  pooled within-class covariance ridged by 1% of its mean variance; tiles
  with fewer than 20 locations in either class are excluded.
- Energy statistic. Per-location mean squared L2 norm, normalised
  across levels per tile; the per-location statistic removes the 4×
  spatial-count imbalance between adjacent strides.
- Spectra. Per-tile channel covariances are trace-normalised before
  averaging; α is the negative log–log slope over eigenvalue ranks 5–100.
- All CIs are 95% percentile bootstrap over tiles (default 1,000
  resamples).
- A tile set without `coco_ann` is skipped for separability (with a
  warning); energy, spectra and inter-level similarity are still computed.
  RLE-encoded segmentations require `pycocotools`; polygon segmentations
  need only PIL.

## 7. Reproducibility

Seeds are derived deterministically, without a per-run salt, so tile sampling and location
subsampling are stable across runs; `provenance.json` records the seed, a hash
of the configuration, and the torch/mmdet versions used.

## 8. Verification (first run)

- after `extract`: `ls <cache_dir>/<model>/<res>/` is populated;
- after `analyze`: `tables/resolution_invariance.csv` Bures values lie in
  (0, 1) and are ordered sensibly;
- an all-NaN off-diagonal CKA matrix indicates the test pipelines yield
  different spatial sizes across configs (the log warns); make the test
  pipeline identical across backbone configs.
