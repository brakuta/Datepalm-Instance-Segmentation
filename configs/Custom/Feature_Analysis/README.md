# Feature_Analysis

Representation-level analysis and **publication-grade figures** for the date
palm multi-resolution benchmark (Stages A–D). The suite quantifies and
visualises how convolutional (CNN), Transformer, and state-space (Mamba)
backbones encode date palm crowns across the 5–30 cm resolution continuum, and
produces the evidence that substantiates the cross-resolution generalisation
claim beyond aggregate mAP.

The detector is **never modified**: all features are captured through PyTorch
forward hooks on `model.backbone` and `model.neck`.

---

## 1. What it produces

**Quantitative (entire sampled test set, with bootstrap confidence intervals)**

- **Resolution-invariance (headline).** Per backbone, the channel covariance of
  an FPN level is estimated per resolution and compared across the continuum by
  trace-normalised **Bures fidelity** — a scale-invariant, registration-free
  descriptor. Optionally correlated (Spearman) with a supplied cross-resolution
  mAP gap to yield an *explanatory* result.
- **Debiased linear CKA** between backbones at matched locations (unbiased
  HSIC; optional RBF variant).
- **Representational dimensionality** — effective rank and participation ratio.

**Qualitative (selected representative tiles)**

- **Multiscale grids** — rows = backbone families, columns = pyramid levels
  P2–P5 (or backbone stages C2–C5); PCA-to-RGB structure and activation-
  magnitude overlay.
- **Composite figure recipes** — rows = backbones (or backbone × stage),
  columns = an isolated axis (e.g. UAV/Aerial/WV-3 resolution, or a single tile
  degraded to coarser ground sampling distances).

All figures are written as vector **PDF** (for the manuscript) and 300-dpi PNG,
with a colourblind-safe Okabe–Ito family palette and editable embedded fonts.

---

## 2. Layout

```
configs/Custom/Feature_Analysis/
├── README.md
├── config_feature_analysis.json     # edit paths here
├── run_feature_analysis.py          # python launcher (sets sys.path)
├── run_feature_analysis.sh          # shell launcher (run_stage_*.sh style)
└── feature_analysis/                # package
    ├── __init__.py
    ├── cli.py            # argparse + dispatch
    ├── config.py         # schema, validation, example config
    ├── util.py           # logging, seeds, bootstrap, publication style
    ├── metrics.py        # CKA, Bures fidelity, effective rank, participation ratio
    ├── extractor.py      # hook-based feature capture (model intact)
    ├── sampling.py       # tile sampling/stratification + cache I/O
    ├── analysis.py       # extract/analyze phases + tables/figures
    └── visualization.py  # visualize/composite phases
```

## 3. Requirements

`torch`, `mmdet` (v3.x), `mmengine`, `numpy`, `scipy`, `matplotlib`, `Pillow`.
Most are already present in the training environment; if needed:

```bash
pip install scipy matplotlib pillow --break-system-packages
```

## 4. Usage

Run from the **mmdetection root** (so `configs/...` paths resolve) on the
machine holding the Stage C checkpoints.

```bash
# 1. (optional) regenerate the annotated example config
python configs/Custom/Feature_Analysis/run_feature_analysis.py make-config \
    --out configs/Custom/Feature_Analysis/config_feature_analysis.json

# 2. edit config_feature_analysis.json (models, tile_sets, showcase/composite
#    tiles, map_gap_csv), then run all phases:
bash configs/Custom/Feature_Analysis/run_feature_analysis.sh run

# or individual phases:
bash configs/Custom/Feature_Analysis/run_feature_analysis.sh extract
bash configs/Custom/Feature_Analysis/run_feature_analysis.sh analyze
bash configs/Custom/Feature_Analysis/run_feature_analysis.sh visualize
bash configs/Custom/Feature_Analysis/run_feature_analysis.sh composite
```

The two-phase split (`extract` on GPU, `analyze` on CPU) lets the qualitative
panels be re-rendered without repeating extraction. `extract`/`analyze`
consume the whole sampled test set; `visualize`/`composite` use only the
nominated tiles.

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
│   ├── multiscale/                         # families × pyramid levels
│   └── composite/                          # isolated-axis recipes
├── tables/                                 # CSV for every metric
└── provenance.json                         # seed, config hash, package versions
```

## 7. Reproducibility

Seeds are derived deterministically (salt-free) so tile sampling and location
subsampling are stable across runs; `provenance.json` records the seed, a hash
of the configuration, and the torch/mmdet versions used.

## 8. Verification (first run)

- after `extract`: `ls <cache_dir>/<model>/<res>/` is populated;
- after `analyze`: `tables/resolution_invariance.csv` Bures values lie in
  (0, 1) and are ordered sensibly;
- an all-NaN off-diagonal CKA matrix indicates the test pipelines yield
  different spatial sizes across configs (the log warns); make the test
  pipeline identical across backbone configs.
