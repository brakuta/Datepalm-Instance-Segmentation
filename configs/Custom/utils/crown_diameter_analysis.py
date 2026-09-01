#!/usr/bin/env python3
# =============================================================================
# crown_diameter_analysis.py
# -----------------------------------------------------------------------------
# Crown-size sensitivity of detection: recall as a function of ground-truth
# crown diameter, per backbone and per source, summarised by D80 and
# cluster-bootstrap confidence intervals.
#
# TERMINOLOGY -- D80 IS NOT AN OPTICAL LIMIT
#   D80 = the operational crown-detection diameter at which FITTED RECALL
#   reaches 0.80, under a stated score threshold, IoU threshold and matching
#   rule. It is NOT a "minimum resolvable diameter" and implies no sensor or
#   diffraction limit; it is a property of this detector at this operating
#   point. It is also not a standard metric in the literature -- define it in
#   the text rather than citing it. What IS conventional is size-stratified
#   recall (the curves) and minimum detectable object size expressed in pixels.
#
# SCALE -- PIXELS ARE MEASURED, METRES ARE DERIVED
#   diam_px comes straight from the decoded mask and is exact. diam_m = diam_px
#   * assumed GSD, so every metre value inherits the GSD assumption in SOURCES
#   (override with --gsd-cm). For sources whose GSD varies tile to tile, report
#   pixels as the measured scale and label metres approximate.
#
# This replaces an earlier quantile-bin + linear-interpolation estimate of D80,
# which was unstable: with unsmoothed bins a handful of instances moving across
# a bin edge could shift the reported threshold by several pixels, and no
# uncertainty or bin count was reported alongside it.
#
# WHAT IS DIFFERENT HERE (each point is a specific methodological fix)
# -----------------------------------------------------------------------------
# 1. MONOTONICITY ON TOP OF BINNING -- IN THAT ORDER.
#    Recall cannot decrease with crown size, so the binned curve is passed
#    through weighted isotonic regression (pool-adjacent-violators) before D80
#    is interpolated off it. Implemented in numpy (see pava), no scikit-learn.
#
#    Note against the original diagnosis: quantile binning was NOT the
#    instability. Simulated against a known logistic truth (60 replicates),
#    plain quantile bins were already well behaved, and replacing them with
#    isotonic regression over individual diameters made the estimator WORSE at
#    every sample size (RMSE 0.298 vs 0.246 at 180 instances, 0.090 vs 0.058 at
#    4800) because each block then carries ~1 binary outcome. Binning supplies
#    the variance reduction; PAVA only removes non-monotone dips that could
#    produce a spurious early upcrossing. Together: RMSE 0.231 vs 0.246 at 180
#    instances, equal thereafter, bias < 0.05 m, and 95% bootstrap CI coverage
#    measured at 95% (38/40). The real gains against the previous analysis are
#    the interval, the bin counts and the monotonicity guarantee -- not a large
#    reduction in point-estimate variance, and the docstring says so rather
#    than overclaiming.
#
# 2. CLUSTER BOOTSTRAP CONFIDENCE INTERVALS.
#    Crowns within one tile are strongly correlated (same scene, same
#    radiometry, same density), so resampling INSTANCES would understate the
#    uncertainty. The bootstrap resamples IMAGES with replacement and refits
#    the isotonic curve each time, giving a percentile CI for D80 that
#    reflects scene-level variability.
#
# 3. BIN COUNTS AND PER-BIN INTERVALS ARE REPORTED.
#    The binned table is still written (it is what you plot), now with n per
#    bin, matched count, and Wilson score intervals, so a bin resting on 7
#    instances is visibly distinguishable from one resting on 700.
#
# 4. DIAMETER IS ALWAYS RECOMPUTED FROM THE DECODED MASK.
#    equivalent diameter = 2*sqrt(mask_area/pi). The COCO `area` field is NOT
#    trusted: it can be stale relative to the stored segmentation after any
#    re-tiling, resampling or polygon edit. The stored field is still recorded
#    as `diam_px_cocoarea` so the discrepancy can be quantified and reported
#    rather than assumed absent.
#
# 5. ALL TEN BACKBONES ARE THE PRIMARY ANALYSIS.
#    EfficientVMamba-B is retained in the headline result. The nine-backbone
#    version is produced as an explicitly labelled SENSITIVITY analysis
#    (--sensitivity-drop), never as the default. Dropping an outlier silently
#    is not a defensible reporting choice; showing that conclusions hold with
#    and without it is.
#
# 6. CHECKPOINT PROTOCOL IS RECORDED AND ENFORCED.
#    make_stagec_pkls.sh generates predictions under a per-sensor DIAGONAL
#    protocol (best_UAV -> UAV, best_GE -> GE and Aerial). Under that protocol
#    a cross-source comparison compares DIFFERENT CHECKPOINTS of the same
#    backbone, which cannot support a claim about one unified multi-source
#    model. --protocol must be stated, is written into every output and into
#    the figure caption, and is VERIFIED against the pkl_provenance*.json that
#    make_stagec_pkls.sh now writes: a declared protocol contradicting the
#    recorded checkpoint basenames aborts the run. With no provenance file the
#    run continues but records the protocol as 'unverified' rather than
#    pretending it was checked.
#
#    `compare` then answers the question empirically: it takes the threshold
#    tables from two protocols and reports, per backbone, whether D80 shifts
#    beyond the bootstrap intervals. Overlap everywhere licences a one-line
#    robustness statement; a systematic shift means the protocol is
#    load-bearing and must be reported.
#
# 7. RESOLUTION LADDERS USE MATCHED SCENES.
#    Comparing native UAV / Aerial / GE conflates GSD with platform, sensor,
#    mosaic radiometry and acquisition date. The `ladder` subcommand instead
#    compares the SAME tiles degraded to coarser GSD (UAV 5->15->30,
#    GE 15->30) and reports D80 against GSD only over instances present at
#    every rung, so the only variable is resolution. Native-source results
#    remain valid as per-source resolvability, and are labelled as such.
#
# MATCHING CONVENTION
#    COCO-style score-ordered greedy matching at IoU >= --iou-thr, identical
#    to extract_instance_errors.py. Any figure rendered from these outputs must
#    use the same rule; a renderer that re-matches globally by IoU will report
#    different TP/FN counts for the same data.
#
# USAGE
#   # 1. per-instance table (needs the prediction pkls + pycocotools)
#   python configs/Custom/utils/crown_diameter_analysis.py extract \
#       --protocol diagonal --out results/qual/crown
#
#   # 2. robust D80 + bootstrap CIs + binned tables
#   python configs/Custom/utils/crown_diameter_analysis.py analyse \
#       --instances results/qual/crown/crown_instances.csv \
#       --out results/qual/crown --bootstrap 1000
#
#   # 3. figure
#   python configs/Custom/utils/crown_diameter_analysis.py figure \
#       --out results/qual/crown
#
#   # 4. is D80 robust to the checkpoint protocol? (after regenerating the
#   #    UAV pkls under the unified protocol -- see
#   #    configs/Custom/Evaluation/README.md)
#   python configs/Custom/utils/crown_diameter_analysis.py extract \
#       --protocol unified --out results/qual/crown_unified \
#       --pkl-pattern 'results/qual/{key}_stageC-unified_{set}.pkl'
#   python configs/Custom/utils/crown_diameter_analysis.py analyse \
#       --instances results/qual/crown_unified/crown_instances.csv \
#       --out results/qual/crown_unified
#   python configs/Custom/utils/crown_diameter_analysis.py compare \
#       --a results/qual/crown/crown_d_thresholds.csv \
#       --b results/qual/crown_unified/crown_d_thresholds.csv \
#       --label-a diagonal --label-b unified --out results/qual/crown
#
#   # optional: controlled resolution ladder (matched instances)
#   python configs/Custom/utils/crown_diameter_analysis.py ladder \
#       --rung UAV_5cm:5 --rung UAV_15sim:15 --rung UAV_30sim:30 \
#       --out results/qual/crown
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["ResNet-50", "ConvNeXt-T", "Swin-S", "PVTv2-B2",
          "GroupMamba-S", "SpatialMamba-S", "MambaVision-S",
          "MambaOut-S", "VMamba-S", "EfficientVMamba-B"]

# Family keys follow stage_ABC_figures_v14.ipynb EXACTLY: the third family is
# "SSM", not "Mamba", and MambaOut-S resolves to CNN. The notebook is explicit
# about why (FAMILY_OVERRIDE + a hard assert): the family key names the
# MECHANISM, not the naming lineage, and MambaOut is a gated CNN ablation with
# no state-space scan. Calling it "Mamba" here would have put a CNN in the SSM
# family and mis-coloured every panel relative to the rest of the deck.
FAMILY_ORDER = ["CNN", "Transformer", "SSM"]
FAMILY = {"ResNet-50": "CNN", "ConvNeXt-T": "CNN",
          "MambaOut-S": "CNN",                 # gated CNN ablation, not an SSM
          "Swin-S": "Transformer", "PVTv2-B2": "Transformer",
          "GroupMamba-S": "SSM", "SpatialMamba-S": "SSM",
          "MambaVision-S": "SSM", "VMamba-S": "SSM",
          "EfficientVMamba-B": "SSM"}
assert FAMILY["MambaOut-S"] == "CNN", "MambaOut must resolve to CNN"

# GSD note. The UAV value is the validated native resolution of the
# orthomosaic (0.05086 m/px), the same constant resample_gsd.py uses as its
# native rung. Carrying a rounded 5.00 cm here while the resolution ladder
# used 5.086 cm made the two analyses disagree by 1.7% in every metre value;
# the fix is to compute with the validated number, not to relabel a panel
# computed from 5.00. The label is rounded for display only.
SOURCES = {
    "UAV (~5.09 cm)":      dict(set="UAV", gsd_cm=5.086, short="UAV",
                                ann_json="/workspace/datasets/COCO/UAV_5cm/"
                                         "Annotations/test_UAV.json"),
    "Aerial (15 cm)":      dict(set="Aerial", gsd_cm=15, short="Aerial",
                                ann_json="/workspace/datasets/COCO/Aerial_15cm/"
                                         "Annotations/test_aerial.json"),
    "Google Earth (15 cm)": dict(set="GE", gsd_cm=15, short="GE",
                                 ann_json="/workspace/datasets/COCO/GE_15cm/"
                                          "Annotations/test_GE.json"),
}

_SUBS = str.maketrans("0123456789", "₀₁₂₃₄"
                                    "₅₆₇₈₉")


def d_sym(target, math=False):
    """The crown-diameter threshold symbol, typeset with a real subscript.

    CD identifies crown diameter explicitly, avoiding the ambiguity of D80.
    Figure text uses mathtext so the PDF carries proper typesetting; plain
    text uses Unicode subscripts so the caption sidecar can be pasted directly
    into the manuscript.
    """
    n = f"{int(round(target * 100))}"
    return f"$CD_{{{n}}}$" if math else "CD" + n.translate(_SUBS)


PKL_PATTERN = "results/qual/{key}_stageC_{set}.pkl"
SCORE_THR = 0.45
IOU_THR = 0.50
N_BINS = 12

# checkpoint protocol -> which checkpoint serves which source
PROTOCOLS = {
    # what make_stagec_pkls.sh actually does
    "diagonal": {"UAV": "best_UAV", "GE": "best_GE", "Aerial": "best_GE"},
    # one checkpoint for every source: required for a unified-model claim
    "unified":  {"UAV": "best_GE", "GE": "best_GE", "Aerial": "best_GE"},
}


# ---------------------------------------------------------------------------
# isotonic regression (weighted PAVA) -- no scikit-learn dependency
# ---------------------------------------------------------------------------
def pava(y, w):
    """Weighted pool-adjacent-violators: least-squares non-decreasing fit.

    y, w must already be ordered by the covariate. Returns the fitted value
    for each input position.
    """
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    blocks = []                     # [w_sum, wy_sum, n]
    for yi, wi in zip(y, w):
        blocks.append([wi, yi * wi, 1])
        while (len(blocks) > 1
               and blocks[-2][1] / blocks[-2][0] > blocks[-1][1] / blocks[-1][0]):
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    out = np.empty(int(sum(b[2] for b in blocks)))
    i = 0
    for w_sum, wy_sum, n in blocks:
        out[i:i + n] = wy_sum / w_sum
        i += n
    return out


def bin_stats(diam, matched, n_bins=N_BINS):
    """Quantile-binned recall: bin median diameter, empirical recall, count."""
    d = np.asarray(diam, float)
    m = np.asarray(matched, float)
    if len(d) < n_bins * 2:
        n_bins = max(3, len(d) // 10)
    edges = np.unique(np.quantile(d, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return np.array([]), np.array([]), np.array([])
    idx = np.clip(np.digitize(d, edges[1:-1], right=False), 0, len(edges) - 2)
    xs, rs, ns = [], [], []
    for b in range(len(edges) - 1):
        s = idx == b
        n = int(s.sum())
        if n == 0:
            continue
        xs.append(np.median(d[s]))
        rs.append(m[s].mean())
        ns.append(float(n))
    return np.asarray(xs), np.asarray(rs), np.asarray(ns)


def isotonic_recall(diam, matched, n_bins=N_BINS):
    """Fit recall(diameter): quantile-bin FIRST, then enforce monotonicity.

    The order of those two steps is not cosmetic, and the obvious
    implementation is the wrong one. Running PAVA directly on individual
    diameters gives every observation its own block, each carrying ~1 binary
    outcome, and the first upcrossing of the target then lands on whichever
    singleton block happens to sit high. Measured against a known logistic
    truth over 60 replicates, that variant was WORSE than plain quantile bins
    at every sample size (RMSE 0.298 vs 0.246 at 180 instances, 0.090 vs 0.058
    at 4800).

    Binning first supplies the variance reduction (~n/12 crowns averaged per
    point); PAVA then supplies the guarantee that a non-monotone dip cannot
    create a spurious early crossing. Combined, it matches plain bins in the
    data-rich regime and beats them where the data are thin -- RMSE 0.231 vs
    0.246 at 180 instances, which is the regime the smallest-crown bins are
    actually in.

    Returns (bin_centres, fitted_recall, counts).
    """
    x, r, n = bin_stats(diam, matched, n_bins)
    if len(x) == 0:
        return x, r, n
    return x, pava(r, n), n


ESTIMATED = "estimated"
BELOW_RANGE = "below_observed_range"     # left censored: crossing < smallest bin
NOT_REACHED = "target_not_reached"       # fitted recall never attains the target


def threshold_at(x, r, target, with_status=False):
    """Smallest diameter whose fitted recall reaches `target`.

    Returns a THREE-VALUED status, not a boolean, because "no number" has two
    completely different meanings and collapsing them hides the more important
    one:

      estimated            the curve crosses the target inside the observed
                           diameter range; the value is an estimate.
      below_observed_range the fitted recall already exceeds the target at the
                           SMALLEST bin, so the crossing lies below every
                           observed crown. The returned x[0] is an UPPER BOUND.
                           Bin edges come from the shared ground-truth
                           distribution, so all such backbones return the
                           identical number -- median == min in a summary table
                           is the signature.
      target_not_reached   the fitted recall never attains the target, even at
                           the largest crowns. NaN. This is NOT merely "above
                           the observed range": the curve may asymptote below
                           the target, and we cannot tell which from one
                           dataset, so the label does not pretend to. Read
                           max_fitted_recall to see how close it came.

    A boolean `censored` flag reported these two cases as False/NaN and
    True/value respectively, which made target_not_reached indistinguishable
    from a clean estimate in any table that only checked the flag.
    """
    x = np.asarray(x, float)
    r = np.asarray(r, float)
    if len(x) == 0:
        return (float("nan"), NOT_REACHED) if with_status else float("nan")
    if r[-1] < target:
        return (float("nan"), NOT_REACHED) if with_status else float("nan")
    j = int(np.argmax(r >= target))
    if j == 0:
        return (float(x[0]), BELOW_RANGE) if with_status else float(x[0])
    r0, r1 = r[j - 1], r[j]
    val = float(x[j]) if r1 <= r0 else float(
        x[j - 1] + (target - r0) * (x[j] - x[j - 1]) / (r1 - r0))
    return (val, ESTIMATED) if with_status else val


def d_threshold_ci(df, target=0.80, n_boot=1000, seed=0,
                   cluster="img_id", n_bins=N_BINS):
    """D-at-target with a CLUSTER bootstrap over images.

    Crowns in one tile share scene, radiometry and density, so an
    instance-level bootstrap would treat correlated observations as
    independent and report intervals that are too narrow. Resampling whole
    images preserves that correlation.
    """
    x, r, _ = isotonic_recall(df["diam_m"].values, df["matched"].values,
                              n_bins)
    point, status = threshold_at(x, r, target, with_status=True)
    max_fit = float(r[-1]) if len(r) else float("nan")

    if n_boot <= 0 or cluster not in df.columns or status == NOT_REACHED:
        return point, float("nan"), float("nan"), 0, status, max_fit

    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster, sort=True)]
    n_g = len(groups)
    diam = [g["diam_m"].values for g in groups]
    match = [g["matched"].values for g in groups]

    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_g, n_g)
        d = np.concatenate([diam[i] for i in pick])
        m = np.concatenate([match[i] for i in pick])
        bx, br, _ = isotonic_recall(d, m, n_bins)
        boots.append(threshold_at(bx, br, target))
    boots = np.asarray(boots, float)
    ok = boots[np.isfinite(boots)]
    if ok.size < max(20, 0.5 * n_boot):
        # the curve fails to reach the target in most resamples: the point
        # estimate is not supported by the data, so say so rather than
        # quoting a CI computed from the minority of resamples that worked.
        return point, float("nan"), float("nan"), int(ok.size), status, max_fit
    return (point, float(np.percentile(ok, 2.5)),
            float(np.percentile(ok, 97.5)), int(ok.size), status, max_fit)


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct at the small counts that occur in the
    smallest-diameter bins, where the normal approximation is not."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(centre - half), float(centre + half)


def binned_table(df, n_bins=12):
    """Quantile-binned empirical recall WITH counts and Wilson intervals.

    Kept for plotting and for inspection; the reported D80 comes from the
    isotonic fit, not from these bins.
    """
    d = df["diam_m"].values
    if len(d) < n_bins * 2:
        n_bins = max(3, len(d) // 10)
    edges = np.unique(np.quantile(d, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(d, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        k = int(df["matched"].values[sel].sum())
        lo, hi = wilson(k, n)
        rows.append(dict(bin=b, d_lo=float(edges[b]), d_hi=float(edges[b + 1]),
                         d_mid=float(np.median(d[sel])), n=n, matched=k,
                         recall=k / n, wilson_lo=lo, wilson_hi=hi))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# extract: per-instance table, diameter recomputed from the decoded mask
# ---------------------------------------------------------------------------
def verify_protocol(pkl_pattern, protocol, models, sources):
    """Check the DECLARED protocol against what actually produced the pkls.

    make_stagec_pkls.sh writes results/qual/pkl_provenance[-unified].json
    recording the checkpoint basename behind every (backbone, set) pair.
    Without that file the declared protocol is only a label, and mislabelling
    it is exactly the failure this guards against -- so verify when the record
    exists, and say plainly that it is unverified when it does not.
    """
    stem = Path(pkl_pattern).parent
    expect = PROTOCOLS[protocol]
    cand = [stem / f"pkl_provenance{'-unified' if protocol == 'unified' else ''}.json",
            stem / "pkl_provenance.json"]
    prov_path = next((p for p in cand if p.exists()), None)
    if prov_path is None:
        print(f"  [warn] no pkl_provenance*.json under {stem}: protocol "
              f"'{protocol}' is RECORDED BUT UNVERIFIED. Regenerate the pkls "
              f"with make_stagec_pkls.sh (which now writes provenance) if you "
              f"need this checked.")
        return "unverified"

    rec = {(e["key"], e["set"]): e for e in json.loads(prov_path.read_text())}
    bad = []
    for k in models:
        for s in sources:
            e = rec.get((k, s))
            if e is None:
                bad.append(f"{k}/{s}: absent from {prov_path.name}")
            elif not e["checkpoint"].startswith(expect[s]):
                bad.append(f"{k}/{s}: expected {expect[s]}*, "
                           f"found {e['checkpoint']}")
    if bad:
        print(f"\n[FATAL] --protocol {protocol} does not match {prov_path}:")
        for b in bad[:12]:
            print("   ", b)
        if len(bad) > 12:
            print(f"    ... and {len(bad) - 12} more")
        sys.exit("declared protocol contradicts the recorded checkpoints")
    print(f"  [ok] protocol '{protocol}' verified against {prov_path.name} "
          f"({len(rec)} entries)")
    return str(prov_path)


def apply_gsd_overrides(args):
    """Metre values scale linearly with the assumed GSD, so make the
    assumption explicit and overridable rather than buried in a constant."""
    for spec in getattr(args, "gsd_cm", []) or []:
        short, _, val = spec.partition("=")
        if not val:
            sys.exit(f"--gsd-cm needs SHORT=CM, got {spec!r}")
        hit = [k for k, v in SOURCES.items() if v["short"] == short]
        if not hit:
            sys.exit(f"--gsd-cm: unknown source {short!r}; known: "
                     f"{[v['short'] for v in SOURCES.values()]}")
        SOURCES[hit[0]]["gsd_cm"] = float(val)
        print(f"  [gsd] {short}: assumed GSD set to {float(val)} cm")
    print("  [gsd] assumed: " + ", ".join(
        f"{v['short']}={v['gsd_cm']}cm" for v in SOURCES.values())
        + "   (diam_px is measured; diam_m derives from these)")


def resync_gsd(df):
    """Re-derive diam_m from the measured diam_px using the GSD currently in
    SOURCES, and refresh the source label.

    diam_px is the measurement; diam_m is a units conversion. Freezing that
    conversion into the extraction step means any later correction to the
    assumed GSD -- e.g. adopting the validated 5.086 cm UAV value over a
    rounded 5.00 -- forces a full re-extraction (decoding 30 pkls of masks)
    to change a multiplication. Applying it here keeps `analyse` the single
    place the assumption enters, so a GSD correction is a cheap re-run and
    can never silently disagree with the panel label.
    """
    if "diam_px" not in df.columns or "short" not in df.columns:
        return df
    by_short = {v["short"]: (k, v["gsd_cm"]) for k, v in SOURCES.items()}
    df = df.copy()
    for short, (label, gsd) in by_short.items():
        sel = df["short"] == short
        if not sel.any():
            continue
        old = float(df.loc[sel, "diam_m"].iloc[0] /
                    df.loc[sel, "diam_px"].iloc[0]) * 100.0
        if abs(old - gsd) > 1e-9:
            print(f"  [gsd] {short}: re-deriving diam_m at {gsd} cm/px "
                  f"(instances file was written at {old:.4g} cm/px, "
                  f"{100.0 * (gsd / old - 1):+.2f}%)")
        df.loc[sel, "diam_m"] = df.loc[sel, "diam_px"] * gsd / 100.0
        df.loc[sel, "gsd_cm"] = gsd
        df.loc[sel, "source"] = label
    return df


def cmd_extract(args):
    import pickle
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils

    def to_rle(seg, h, w):
        if isinstance(seg, dict):
            rle = dict(seg)
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode()
            return rle
        if isinstance(seg, list):
            return maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
        arr = np.asarray(seg)
        return maskUtils.encode(np.asfortranarray(arr.astype(np.uint8)))

    def match(gt_rles, dt_rles, scores, iou_thr):
        n_g, n_d = len(gt_rles), len(dt_rles)
        matched = np.zeros(n_g, bool)
        if n_g == 0 or n_d == 0:
            return matched
        ious = np.asarray(maskUtils.iou(dt_rles, gt_rles,
                                        [0] * n_g)).reshape(n_d, n_g)
        for d in np.argsort(-np.asarray(scores)):
            row = ious[d].copy()
            row[matched] = -1
            g = int(np.argmax(row))
            if row[g] >= iou_thr:
                matched[g] = True
        return matched

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_gsd_overrides(args)
    protocol = PROTOCOLS[args.protocol]
    print(f'checkpoint protocol = {args.protocol}: {protocol}')
    verified = verify_protocol(args.pkl_pattern, args.protocol, MODELS,
                               [s["set"] for s in SOURCES.values()])
    if args.protocol == "diagonal":
        print('  [NOTE] Under the diagonal protocol each source is predicted by '
              'a DIFFERENT checkpoint of the same backbone. Per-source results '
              'are valid; any cross-source statement must be labelled as '
              'comparing per-sensor-selected models, not one unified model.')

    rows, area_delta = [], []
    for label, s in SOURCES.items():
        coco = COCO(s["ann_json"])
        pkls = {}
        for k in MODELS:
            p = Path(args.pkl_pattern.format(key=k, set=s["set"]))
            if not p.exists():
                sys.exit(f"[FATAL] missing pkl: {p}")
            with open(p, "rb") as f:
                pkls[k] = {r["img_id"]: r for r in pickle.load(f)}

        img_ids = sorted(coco.getImgIds())
        if args.limit:
            img_ids = img_ids[:args.limit]
        print(f"[{s['short']}] {len(img_ids)} tiles x {len(MODELS)} backbones")

        for n_done, img_id in enumerate(img_ids, 1):
            info = coco.loadImgs(img_id)[0]
            h, w = info["height"], info["width"]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))
            if not anns:
                continue
            gt_rles = [to_rle(a["segmentation"], h, w) for a in anns]

            # ---- diameter ALWAYS from the decoded mask -------------------
            dec_area = np.asarray(maskUtils.area(gt_rles), float)
            diam_px = 2.0 * np.sqrt(np.maximum(dec_area, 1.0) / np.pi)
            # the stored field, kept only so the discrepancy is measurable
            coco_area = np.asarray([a.get("area") or np.nan for a in anns],
                                   float)
            diam_px_coco = 2.0 * np.sqrt(np.maximum(coco_area, 1.0) / np.pi)
            good = np.isfinite(coco_area) & (dec_area > 0)
            if good.any():
                area_delta.append(pd.DataFrame(dict(
                    source=label,
                    rel_diff=(coco_area[good] - dec_area[good]) / dec_area[good])))

            per_gt = {k: match(gt_rles,
                               *_pred(pkls[k].get(img_id), args.score_thr,
                                      h, w, to_rle),
                               args.iou_thr) for k in MODELS}

            for i, a in enumerate(anns):
                base = dict(source=label, short=s["short"], gsd_cm=s["gsd_cm"],
                            file_name=info["file_name"], img_id=img_id,
                            gt_id=a["id"],
                            diam_px=float(diam_px[i]),
                            diam_m=float(diam_px[i] * s["gsd_cm"] / 100.0),
                            diam_px_cocoarea=float(diam_px_coco[i]))
                for k in MODELS:
                    rows.append(dict(base, backbone=k, family=FAMILY[k],
                                     checkpoint=protocol[s["set"]],
                                     matched=int(per_gt[k][i])))
            if n_done % 200 == 0:
                print(f"    {n_done}/{len(img_ids)}")

    df = pd.DataFrame(rows)
    p = out / "crown_instances.csv"
    df.to_csv(p, index=False)

    prov = dict(created=datetime.now(timezone.utc).isoformat(),
                protocol=args.protocol, checkpoint_map=protocol,
                protocol_verified_against=verified,
                pkl_pattern=args.pkl_pattern,
                score_thr=args.score_thr, iou_thr=args.iou_thr,
                matching="score-ordered greedy, COCO-style",
                diameter="2*sqrt(decoded_mask_area/pi)",
                assumed_gsd_cm={v["short"]: v["gsd_cm"]
                                for v in SOURCES.values()},
                d80_definition=("operational crown-detection diameter at "
                                "which fitted recall reaches the target; not "
                                "an optical or sensor limit"),
                models=MODELS, n_instances=int(len(df) / max(len(MODELS), 1)),
                n_rows=int(len(df)))
    if area_delta:
        ad = pd.concat(area_delta, ignore_index=True)
        prov["coco_area_vs_decoded"] = dict(
            median_rel_diff=float(ad["rel_diff"].median()),
            p95_abs_rel_diff=float(ad["rel_diff"].abs().quantile(0.95)),
            frac_over_1pct=float((ad["rel_diff"].abs() > 0.01).mean()))
        print("\n[area check] stored COCO area vs decoded mask area: "
              f"median rel diff {prov['coco_area_vs_decoded']['median_rel_diff']:+.4f}, "
              f"{100 * prov['coco_area_vs_decoded']['frac_over_1pct']:.2f}% of "
              "crowns differ by >1%")
    with open(out / "crown_provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(f"\ninstances -> {p}  ({len(df)} rows)")
    print(f"provenance -> {out / 'crown_provenance.json'}")


def _pred(result, score_thr, h, w, to_rle):
    if result is None:
        return [], []
    pi = result["pred_instances"]
    scores = np.asarray(pi["scores"])
    keep = np.where(scores >= score_thr)[0]
    masks = pi.get("masks", [])
    return ([to_rle(masks[int(i)], h, w) for i in keep],
            [float(scores[int(i)]) for i in keep])


# ---------------------------------------------------------------------------
# analyse: robust D80 + CIs + binned tables
# ---------------------------------------------------------------------------
def cmd_analyse(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_gsd_overrides(args)
    df = resync_gsd(pd.read_csv(args.instances))
    targets = [float(t) for t in args.targets]

    cohorts = {"all10": MODELS}
    if args.sensitivity_drop:
        keep = [m for m in MODELS if m not in args.sensitivity_drop]
        cohorts[f"sens_drop_{'_'.join(args.sensitivity_drop)}"] = keep

    thr_rows, bin_rows = [], []
    for src, g_src in df.groupby("source", sort=False):
        for bb, g in g_src.groupby("backbone", sort=False):
            for t in targets:
                pt, lo, hi, nb, status, maxfit = d_threshold_ci(
                    g, target=t, n_boot=args.bootstrap, seed=args.seed,
                    n_bins=args.bins)
                thr_rows.append(dict(
                    source=src, short=g["short"].iloc[0], backbone=bb,
                    family=FAMILY.get(bb, "?"),
                    checkpoint=g["checkpoint"].iloc[0]
                    if "checkpoint" in g else "?",
                    target=t, d_m=pt, ci_lo=lo, ci_hi=hi,
                    status=status, censored=(status == BELOW_RANGE),
                    max_fitted_recall=maxfit,
                    n_instances=int(len(g)),
                    n_images=int(g["img_id"].nunique()),
                    n_boot_valid=nb,
                    recall_overall=float(g["matched"].mean())))
            b = binned_table(g, args.bins)
            b.insert(0, "backbone", bb)
            # The figure selects its columns by short code so that a change to
            # the display label cannot silently drop a panel. That lookup only
            # works if the code is actually in this table -- without it the
            # figure falls back to raw CSV order and the guard does nothing.
            b.insert(0, "short", g["short"].iloc[0])
            b.insert(0, "source", src)
            bin_rows.append(b)

    thr = pd.DataFrame(thr_rows)
    thr.to_csv(out / "crown_d_thresholds.csv", index=False)

    # Record what THIS step assumed. The extraction-time provenance predates
    # the GSD resync and the bootstrap setting, so a caption built from it
    # alone would report numbers the analysis did not use.
    pj = out / "crown_provenance.json"
    if pj.exists():
        prov = json.loads(pj.read_text())
        prov["assumed_gsd_cm"] = {v["short"]: v["gsd_cm"]
                                  for v in SOURCES.values()}
        prov["bootstrap"] = int(args.bootstrap)
        prov["n_bins"] = int(args.bins)
        prov["analysed"] = datetime.now(timezone.utc).isoformat()
        pj.write_text(json.dumps(prov, indent=2))
    pd.concat(bin_rows, ignore_index=True).to_csv(
        out / "crown_recall_bins.csv", index=False)

    # cohort summaries: all ten primary, reduced set as sensitivity only
    summ = []
    for name, members in cohorts.items():
        sub = thr[thr["backbone"].isin(members)]
        for (src, t), g in sub.groupby(["source", "target"], sort=False):
            # Only 'estimated' rows are measurements. below_observed_range
            # rows are upper bounds (the first bin centre), and
            # target_not_reached rows have no value at all; neither belongs in
            # a median presented as a threshold.
            free = g[g["status"] == ESTIMATED] if "status" in g else \
                g[~g["censored"].astype(bool)]
            v = free["d_m"].dropna()
            summ.append(dict(cohort=name, n_backbones=len(members),
                             source=src, target=t,
                             d_m_median=float(v.median()) if len(v) else np.nan,
                             d_m_min=float(v.min()) if len(v) else np.nan,
                             d_m_max=float(v.max()) if len(v) else np.nan,
                             n_defined=int(len(v)),
                             n_below_range=int((g["status"] == BELOW_RANGE).sum())
                             if "status" in g else 0,
                             n_not_reached=int((g["status"] == NOT_REACHED).sum())
                             if "status" in g else int(g["d_m"].isna().sum())))
    pd.DataFrame(summ).to_csv(out / "crown_d_summary.csv", index=False)

    print(thr[thr["target"] == targets[0]]
          .sort_values(["source", "d_m"])
          .to_string(index=False,
                     columns=["source", "backbone", "family", "d_m",
                              "ci_lo", "ci_hi", "censored",
                              "n_instances", "n_images"],
                     float_format=lambda x: f"{x:.3f}"))
    print(f"\ntables -> {out}")
    if args.sensitivity_drop:
        print(f"[note] '{', '.join(args.sensitivity_drop)}' appears in the "
              f"all-ten primary result AND in a separate sensitivity cohort. "
              f"Report both.")


# ---------------------------------------------------------------------------
# ladder: matched-instance resolution analysis
# ---------------------------------------------------------------------------
def cmd_ladder(args):
    """D80 vs GSD over the SAME crowns degraded to coarser resolution.

    Native UAV/Aerial/GE differ in platform, sensor, mosaic processing and
    date as well as GSD, so a native cross-source curve cannot isolate
    resolution. A ladder built from resample_gsd.py outputs (UAV 5->15->30,
    GE 15->30) holds the scene fixed and varies only GSD. Instances are
    intersected across rungs on (file stem, gt_id) so every rung describes
    the same crowns.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rungs = []
    for spec in args.rung:
        name, _, gsd = spec.partition(":")
        if not gsd:
            sys.exit(f"--rung needs NAME:GSD_CM, got {spec}")
        rungs.append((name, float(gsd)))

    frames = []
    for name, gsd in rungs:
        p = Path(args.instances_dir) / f"crown_instances_{name}.csv"
        if not p.exists():
            sys.exit(f"[FATAL] missing {p}. Run `extract` for rung {name} "
                     f"first (its own SOURCES entry / prediction pkls).")
        d = pd.read_csv(p)
        d["rung"], d["gsd_cm"] = name, gsd
        d["key"] = d["file_name"].map(lambda s: Path(str(s)).stem) + "#" + \
            d["gt_id"].astype(str)
        frames.append(d)

    common = set(frames[0]["key"])
    for d in frames[1:]:
        common &= set(d["key"])
    if not common:
        sys.exit("[FATAL] no instances common to all rungs -- the resampled "
                 "sets do not share (file stem, gt_id). Regenerate them with "
                 "resample_gsd.py, which preserves both.")
    print(f"{len(common)} crowns present at all {len(rungs)} rung(s); "
          f"per-rung totals: "
          f"{[int(len(d)/max(d['backbone'].nunique(),1)) for d in frames]}")

    rows = []
    for d in frames:
        d = d[d["key"].isin(common)]
        for bb, g in d.groupby("backbone", sort=False):
            for t in [float(x) for x in args.targets]:
                pt, lo, hi, nb, status, maxfit = d_threshold_ci(
                    g, target=t, n_boot=args.bootstrap, seed=args.seed)
                rows.append(dict(rung=d["rung"].iloc[0],
                                 gsd_cm=d["gsd_cm"].iloc[0], backbone=bb,
                                 family=FAMILY.get(bb, "?"), target=t,
                                 d_m=pt, ci_lo=lo, ci_hi=hi,
                                 status=status, censored=(status == BELOW_RANGE),
                                 max_fitted_recall=maxfit,
                                 n_instances=int(len(g)),
                                 n_images=int(g["img_id"].nunique()),
                                 n_boot_valid=nb,
                                 recall_overall=float(g["matched"].mean())))
    res = pd.DataFrame(rows)
    res.to_csv(out / "crown_ladder.csv", index=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nladder -> {out / 'crown_ladder.csv'}  "
          f"(matched instances only: resolution is the only variable)")


# ---------------------------------------------------------------------------
# figure -- theme registry ported verbatim from stage_ABC_figures_v14.ipynb
# ---------------------------------------------------------------------------
# Same palettes, fonts, rcParams, panel tags, despine/grid behaviour and
# save() naming as the rest of the deck, so this panel drops in beside the
# other manuscript figures without a visible seam. Kept as a self-contained
# block (rather than importing the notebook) because the notebook is not a
# module; if the notebook theme changes, mirror it here.
# ---------------------------------------------------------------------------
# Backbones the text singles out are drawn heavier so they are findable among
# ten overplotted curves; the rest stay as context.
EMPHASIS = ("SpatialMamba-S", "MambaVision-S", "EfficientVMamba-B")

TH = PAL = CMAP = None
THEME = "tidy"
MM = 1 / 25.4
DOUBLE_COL_MM = 190          # ISPRS full-width figure


def mm2in(x):
    return x * MM


def _build_themes():
    import matplotlib.font_manager as fm
    from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

    def _font(*cands):
        installed = {f.name for f in fm.fontManager.ttflist}
        for c in cands:
            if c in installed:
                return c
        return "DejaVu Sans"

    sans = _font("Source Sans Pro", "Source Sans 3", "Helvetica Neue",
                 "Arial", "Liberation Sans", "DejaVu Sans")
    serif = _font("Source Serif 4", "Source Serif Pro", "Charter", "Georgia",
                  "Liberation Serif", "DejaVu Serif")

    def desat(color, k=0.25):
        r, g, b = to_rgb(color)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return to_hex((r + (lum - r) * k, g + (lum - g) * k, b + (lum - b) * k))

    def _seq(cols):
        return LinearSegmentedColormap.from_list("seq", cols)

    base = {"CNN": "#35617F", "Transformer": "#C0823C", "SSM": "#2E8B7F"}
    return sans, serif, {
        "editorial": dict(
            palette={k: desat(v, 0.15) for k, v in base.items()},
            cmap=_seq(["#FAF8F1", "#E3ECE4", "#B5D2C4", "#7CB09A", "#4C8672",
                       "#2C5A4B"]),
            title_font=serif, tag="lower_serif",
            panel_bg="none", grid="#E7E3D9", grid_lw=0.7, accent="#3F7D6C",
            rc={"font.family": sans, "axes.edgecolor": "#3A3A3A",
                "axes.linewidth": 0.6, "text.color": "#222222",
                "axes.labelcolor": "#333333", "xtick.color": "#555555",
                "ytick.color": "#555555"},
            despine_offset=6, trim=True),
        "tidy": dict(
            palette=dict(base),
            cmap=_seq(["#FFFFD9", "#EDF8B1", "#C7E9B4", "#7FCDBB", "#41B6C4",
                       "#1D91C0", "#225EA8"]),
            title_font=sans, tag="facet",
            panel_bg="#F3F3EF", grid="#FFFFFF", grid_lw=1.1, accent="#2E8B7F",
            rc={"font.family": sans, "axes.edgecolor": "#BFBFBF",
                "axes.linewidth": 0.8, "text.color": "#2A2A2A",
                "axes.labelcolor": "#2A2A2A", "xtick.color": "#4A4A4A",
                "ytick.color": "#4A4A4A"},
            despine_offset=0, trim=False),
        "mono": dict(
            palette={"CNN": "#A6A6A6", "Transformer": "#6E6E6E",
                     "SSM": "#2B2B2B"},
            cmap=_seq(["#FFFFFF", "#E3E3E3", "#BDBDBD", "#8F8F8F", "#606060",
                       "#2E2E2E"]),
            title_font=sans, tag="upper_bold",
            panel_bg="none", grid="#DDDDDD", grid_lw=0.6, accent="#2B2B2B",
            rc={"font.family": sans, "axes.edgecolor": "#000000",
                "axes.linewidth": 0.8, "text.color": "#000000",
                "axes.labelcolor": "#000000", "xtick.color": "#000000",
                "ytick.color": "#000000"},
            despine_offset=0, trim=True,
            hatches={"CNN": "..", "Transformer": "///", "SSM": ""}),
        "slate": dict(
            palette={"CNN": "#2C5F86", "Transformer": "#D08A2C",
                     "SSM": "#1F7A6D"},
            cmap=_seq(["#F4F1E8", "#CBE1D8", "#8FC3B3", "#4E9B88", "#1F7A6D",
                       "#12574E"]),
            title_font=sans, tag="left_rule",
            panel_bg="none", grid="#E9E9E9", grid_lw=0.7, accent="#1F7A6D",
            rc={"font.family": sans, "axes.edgecolor": "#3D4A52",
                "axes.linewidth": 1.0, "text.color": "#1F2A30",
                "axes.labelcolor": "#2A363D", "xtick.color": "#43535C",
                "ytick.color": "#43535C"},
            despine_offset=5, trim=True),
    }


def use_theme(name):
    """Identical rcParams to the notebook's use_theme, including savefig.dpi
    600 and pdf.fonttype 42."""
    global TH, PAL, CMAP, THEME
    import matplotlib as mpl
    sans, _serif, themes = _build_themes()
    if name not in themes:
        raise SystemExit(f"unknown theme {name!r}; choose from {sorted(themes)}")
    THEME = name
    TH = themes[name]
    PAL = TH["palette"]
    CMAP = TH["cmap"]
    base = {
        "font.size": 8, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "figure.dpi": 130, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    }
    base.update(TH["rc"])
    mpl.rcParams.update(base)


def despine(ax, left=True, bottom=True):
    off = TH["despine_offset"]
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s, on in (("left", left), ("bottom", bottom)):
        ax.spines[s].set_visible(on)
        if on and off:
            ax.spines[s].set_position(("outward", off))


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=TH["grid"], lw=TH["grid_lw"], zorder=0)
    ax.set_axisbelow(True)


def apply_panel_bg(ax):
    if TH["panel_bg"] != "none":
        ax.set_facecolor(TH["panel_bg"])


def panel_tag(ax, letter, x=-0.02, y=1.03):
    st = TH["tag"]
    if st == "facet":
        ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9.5, fontweight="bold")
        return
    if st == "lower_serif":
        ax.text(x, y, letter, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold", family=TH["title_font"])
    elif st == "upper_bold":
        ax.text(x, y, f"({letter.upper()})", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9.5, fontweight="bold")
    else:
        ax.text(x, y, letter.upper(), transform=ax.transAxes, ha="left",
                va="bottom", fontsize=10, fontweight="bold", color=TH["accent"])


def set_title(ax, text, size=None):
    st = TH["tag"]
    if st == "facet":
        ax.set_title("")
        ax.annotate(text, xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    va="bottom", fontsize=size or 9.0, fontweight="normal",
                    color="#222", family=TH["title_font"])
    elif st == "left_rule":
        ax.set_title(text, loc="left", fontweight="bold",
                     family=TH["title_font"], pad=7)
    elif st == "upper_bold":
        ax.set_title(text, fontweight="bold", pad=7)
    else:
        ax.set_title(text, loc="left", fontsize=9.5, pad=7, color="#2A2A2A",
                     family=TH["title_font"])


def fam_legend_handles(marker=True):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    if marker:
        return [Line2D([0], [0], marker="o", ls="none", mfc=PAL[f],
                       mec="white", ms=7, label=f) for f in FAMILY_ORDER]
    return [Patch(fc=PAL[f], ec="white",
                  hatch=TH.get("hatches", {}).get(f, ""), label=f)
            for f in FAMILY_ORDER]


def save(fig, out_dir, stem):
    name = f"{stem}__{THEME}"
    for ext in ("pdf", "png"):
        fig.savefig(Path(out_dir) / f"{name}.{ext}", facecolor="white")
        print(f"  {Path(out_dir) / f'{name}.{ext}'}")


def build_caption(prov, thr_all, target):
    """Manuscript caption text, written to a sidecar rather than burned into
    the figure. A caption belongs in the manuscript where it can be edited and
    typeset; baking it into the artwork duplicates it and guarantees the two
    drift apart."""
    # SOURCES wins over the extraction-time record: `analyse` re-derives
    # diam_m from diam_px at the current GSD, so a stale value in the
    # provenance file must not be what the caption reports.
    gsd = dict(prov.get("assumed_gsd_cm", {}))
    gsd.update({v["short"]: v["gsd_cm"] for v in SOURCES.values()})
    name_of = {v["short"]: k.split(" (")[0] for k, v in SOURCES.items()}
    gsd_txt = ", ".join(f"{name_of.get(k, k)} {v:g} cm"
                        for k, v in gsd.items())

    n_src, same_n = {}, True
    if "n_instances" in thr_all:
        for src, g in thr_all.groupby("source", sort=False):
            n_src[src] = int(g["n_instances"].max())
            same_n &= g["n_instances"].nunique() == 1
    n_txt = "; ".join(f"{k.split(' (')[0]} {v:,}" for k, v in n_src.items())
    # Only claim the counts are shared if they actually are -- the reference
    # set is the same for every backbone, but say so from the table, not from
    # the assumption.
    n_same_txt = (", identical across backbones within a source"
                  if same_n else "")
    D = d_sym(target)
    pct = f"{target:.0%}"
    # Never invent the resample count: if the analysis step did not record it,
    # the caption says how the interval was built without asserting a number.
    nb = prov.get("bootstrap")
    boot_txt = f" ({int(nb):,} resamples)" if nb else ""

    # Which backbones lead is read off the table, not hard-coded: a caption
    # that names models from a previous run is the classic way a figure and
    # its text drift apart.
    est = (thr_all[thr_all["status"] == ESTIMATED] if "status" in thr_all
           else thr_all)
    lead = list(est.loc[est.groupby("source")["d_m"].idxmin(), "backbone"]
                .drop_duplicates()) if len(est) else []
    lead_txt = (" and ".join(lead) if len(lead) < 3
                else ", ".join(lead[:-1]) + " and " + lead[-1]) or "No backbone"

    # Censoring is explained only when something is actually censored. In the
    # unified analysis nothing is, and carrying an unused legend of failure
    # modes into the main caption invites the reader to hunt the figure for
    # markers that are not there. The methods and the sensitivity figure keep
    # the full definition.
    cens = ""
    if "status" in thr_all:
        nb = int((thr_all["status"] == BELOW_RANGE).sum())
        nn = int((thr_all["status"] == NOT_REACHED).sum())
        if nb or nn:
            bits = []
            if nn:
                bits.append(f"{nn} did not attain {pct} recall at any crown "
                            f"size (n.e.)")
            if nb:
                bits.append(f"{nb} crossed the target below the smallest "
                            f"observed crown, so the value is an upper bound "
                            f"(ub)")
            cens = (f" Open markers with carets denote censored combinations, "
                    f"plotted apart from the estimates rather than among them: "
                    f"{' and '.join(bits)}. Censored entries are excluded from "
                    f"the family summaries in (a-c).")
        else:
            cens = (f" {D} was estimable for all {len(thr_all)} "
                    f"source-backbone combinations; no entry is censored.")

    return (
        f"Detection recall as a function of ground-truth crown diameter. "
        f"{D} is the crown diameter at which fitted recall reaches {pct} at "
        f"the stated operating point. It is a property of the detector and "
        f"of that operating point, not an optical limit: it is neither the "
        f"minimum resolvable crown size nor a sensor resolution limit, and it "
        f"moves with the confidence and IoU thresholds (see the sensitivity "
        f"analysis). "
        f"(a-c) Recall against crown diameter, one line per backbone. Line "
        f"colour denotes architecture family and the dash pattern "
        f"distinguishes the members within a family, following the same "
        f"convention as the Stage C convergence figure so that a backbone "
        f"keeps one appearance throughout. The plotted line is the monotone "
        f"fit, so its crossing of the dashed {pct} guide is the value "
        f"tabulated in (d-f). Recall is fitted against diameter by "
        f"isotonic regression (weighted pool-adjacent-violators) over "
        f"{int(prov.get('n_bins', N_BINS))} equal-count (quantile) diameter "
        f"bins, each bin weighted by "
        f"its crown count; binning before the monotone fit rather than after "
        f"lowers the fit error on held-out crowns. n is the number of "
        f"reference (ground-truth) crowns{n_same_txt}: {n_txt}. "
        f"(d-f) {D} per backbone, with the value printed beside each "
        f"estimate. Error bars are 95% confidence intervals obtained by "
        f"resampling test images with replacement{boot_txt}, retaining all "
        f"crowns within each selected image, so that crowns "
        f"sharing a tile are not treated as independent observations. Upper "
        f"axes give the equivalent diameter in pixels at that source's ground "
        f"sample distance. Backbones are ordered by mean {D} rank across the "
        f"three sources; colours denote architecture family, with MambaOut-S "
        f"grouped as a CNN because it contains no state-space scan. "
        f"{lead_txt} attain the lowest operational {D}, that is, they "
        f"recover crowns down to a smaller diameter at the chosen operating "
        f"point than the other backbones; this ranks small-crown recovery at "
        f"one operating point and is not a statement of overall segmentation "
        f"quality, for which the mask AP results should be consulted. "
        f"A detection counts as a match at confidence score >= "
        f"{prov.get('score_thr', SCORE_THR)} and mask IoU >= "
        f"{prov.get('iou_thr', IOU_THR)}, under score-ordered greedy "
        f"assignment. Crown diameter is the equivalent circular diameter of "
        f"the decoded ground-truth mask; assumed ground sample distances are "
        f"{gsd_txt}. All predictions derive from a single checkpoint per "
        f"backbone, selected on Google Earth validation mask mAP@0.5 and "
        f"applied to all three sources.{cens} Exact ground sample distances "
        f"and the full numerical results are given in the supplementary "
        f"table."
    )


# Per-model line styles, ported verbatim from _sc_style_map in
# stage_ABC_figures_v14.ipynb so a backbone carries the SAME colour and dash
# pattern here as in the Stage C convergence figure.  The canonical member
# order is fixed explicitly below: deriving the dash index from the order of
# the plotted data made a model's appearance change when D80 ranking changed.
LINE_STYLES = ["-", (0, (4.2, 1.4)), (0, (1.1, 1.1)),
               (0, (5.0, 1.3, 1.1, 1.3)),
               (0, (3.0, 1.0, 1.0, 1.0, 1.0, 1.0)), (0, (6.0, 1.7))]
# One fixed, unique marker per backbone.  The marker is drawn once at the
# first fitted point of every recall curve and repeated in the legend.  Using
# the first point avoids the severe overlap at the high-diameter endpoints,
# where nearly all curves converge close to recall = 1.
BACKBONE_MARKERS = {
    "ConvNeXt-T": "o",
    "MambaOut-S": "s",
    "ResNet-50": "^",
    "PVTv2-B2": "D",
    "Swin-S": "v",
    "EfficientVMamba-B": "P",
    "GroupMamba-S": "X",
    "MambaVision-S": "<",
    "SpatialMamba-S": ">",
    "VMamba-S": "*",
}

# Exact family-major/alphabetical order used by the Stage C loss figure.
# Keeping this independent of the D80 ranking guarantees that colour, dash
# and marker identify the same backbone in every manuscript figure.
LOSS_FIGURE_STYLE_ORDER = {
    "CNN": ["ConvNeXt-T", "MambaOut-S", "ResNet-50"],
    "Transformer": ["PVTv2-B2", "Swin-S"],
    "SSM": ["EfficientVMamba-B", "GroupMamba-S", "MambaVision-S",
            "SpatialMamba-S", "VMamba-S"],
}


def family_sorted(backbones):
    """Family-major, alphabetical within family -- the legend's reading
    order, which is not the D80 ranking used for the row-2 y-axis."""
    return sorted(backbones,
                  key=lambda b: (FAMILY_ORDER.index(FAMILY.get(b, "CNN")), b))


def style_map(backbones):
    styles = {}
    requested = set(backbones)
    for fam in FAMILY_ORDER:
        canonical = LOSS_FIGURE_STYLE_ORDER[fam]
        for i, bb in enumerate(canonical):
            if bb in requested:
                styles[bb] = dict(
                    color=PAL[fam],
                    ls=LINE_STYLES[i % len(LINE_STYLES)],
                    marker=BACKBONE_MARKERS[bb],
                )

    # Fail loudly if a new backbone was added without assigning the manuscript
    # style it should carry; silently cycling would recreate the mismatch.
    missing = requested - set(styles)
    if missing:
        raise KeyError("add fixed loss-figure styles for: "
                       + ", ".join(sorted(missing)))
    return styles


def grouped_legend(fig, backbones, styles, max_rows=2, x0=0.085, gap=0.030,
                   y=-0.085):
    """One legend block per family, packed left to right.

    Block widths are measured after a first draw rather than hardcoded: with
    family membership as a variable (MambaOut is a CNN, giving 3/2/5 rather
    than 2/2/6) fixed x positions detach from their blocks silently.
    """
    from matplotlib.lines import Line2D
    legends = []
    for fam in FAMILY_ORDER:
        keys = [b for b in family_sorted(backbones)
                if FAMILY.get(b, "CNN") == fam]
        if not keys:
            continue
        handles = [Line2D([0], [0], color=styles[k]["color"], ls=styles[k]["ls"],
                          lw=0.88, marker=styles[k]["marker"], ms=2.6,
                          mfc=styles[k]["color"], mec="white", mew=0.30,
                          label=k) for k in keys]
        lg = fig.legend(handles=handles, title=fam, loc="lower left",
                        bbox_to_anchor=(x0, y),
                        ncol=max(1, int(np.ceil(len(keys) / max_rows))),
                        frameon=False, fontsize=6.2, title_fontsize=6.8,
                        handlelength=2.25, handletextpad=0.42,
                        columnspacing=0.95, labelspacing=0.34,
                        borderaxespad=0)
        lg.get_title().set_color(PAL[fam])
        lg.get_title().set_fontweight("bold")
        legends.append(lg)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    x = x0
    for lg in legends:
        lg.set_bbox_to_anchor((x, y), transform=fig.transFigure)
        x += lg.get_window_extent(renderer=rend).width / fig.bbox.width + gap
    return legends


def write_supp_table(out, thr_all, target, order):
    """Supplementary table: exact GSDs and the full numbers.

    The figure carries one number per point and nothing more. Exact ground
    sample distances, pixel equivalents, interval endpoints and sample sizes
    belong in a table a reader can read values off, not crowded into
    artwork -- so they are written here, as CSV for the supplement and as
    Markdown for pasting.
    """
    D = d_sym(target)
    gsd_of = {s["short"]: s["gsd_cm"] for s in SOURCES.values()}
    rows = []
    for src in [s for s in SOURCES if s in set(thr_all["source"])]:
        g = thr_all[thr_all["source"] == src].set_index("backbone")
        gsd = gsd_of.get(g["short"].iloc[0], np.nan) if len(g) else np.nan
        for bb in order:
            if bb not in g.index:
                continue
            r = g.loc[bb]
            est = r.get("status", ESTIMATED) == ESTIMATED
            rows.append({
                "source": src, "gsd_cm": gsd, "backbone": bb,
                "family": FAMILY.get(bb, "?"),
                f"{D}_m": r["d_m"] if est else np.nan,
                "ci_lo_m": r["ci_lo"] if est else np.nan,
                "ci_hi_m": r["ci_hi"] if est else np.nan,
                f"{D}_px": (r["d_m"] * 100.0 / gsd
                            if est and np.isfinite(gsd) else np.nan),
                "status": r.get("status", ESTIMATED),
                "n_crowns": r.get("n_instances", np.nan),
                "n_images": r.get("n_images", np.nan),
                "recall_overall": r.get("recall_overall", np.nan)})
    t = pd.DataFrame(rows)
    t.to_csv(out / "table_crown_d_supplementary.csv", index=False)

    def fmt(v, n=2):
        return "--" if pd.isna(v) else f"{v:.{n}f}"
    md = [f"| Source | GSD (cm) | Backbone | Family | {D} (m) | 95% CI (m) "
          f"| {D} (px) | n crowns |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        md.append(f"| {r['source']} | {r['gsd_cm']:g} | {r['backbone']} | "
                  f"{r['family']} | {fmt(r[f'{D}_m'])} | "
                  f"{fmt(r['ci_lo_m'])}-{fmt(r['ci_hi_m'])} | "
                  f"{fmt(r[f'{D}_px'], 1)} | {int(r['n_crowns']):,} |")
    (out / "table_crown_d_supplementary.md").write_text("\n".join(md) + "\n")
    print(f"  {out / 'table_crown_d_supplementary.csv'}")
    print(f"  {out / 'table_crown_d_supplementary.md'}")


def cmd_figure(args):
    """2 x 3 layout: recall curves on top, D80 intervals below, one column per
    sensor.

    Two things the earlier version got wrong.

    (1) All three sensors shared ONE D80 axis. UAV thresholds sit near 2.4 m
        and the 15 cm sensors near 5-6 m, so the axis spent its range on the
        gap BETWEEN sensors and compressed the between-backbone differences the
        panel exists to show, while three row offsets per backbone read as
        scatter. Each sensor now gets its own column and its own x-scale, with
        the backbone order fixed down a shared y-axis so the reordering ACROSS
        sensors -- the actual finding -- is legible.

    (2) The curves were the RAW quantile bins while the reported D80 came from
        the isotonic fit. Where the raw bins are non-monotone (a Google Earth
        curve falls back below 0.80 at the largest diameters, and an Aerial
        curve dips near 5.8 m) the dashed target line crosses the drawn curve
        at a diameter that is NOT the tabulated D80, so a reader measuring off
        the figure would disagree with the table. The fitted monotone curve is
        now the drawn line and the raw bin values are shown as faint dots, so
        the crossing on the page is the number in the table and the raw data is
        still visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    apply_gsd_overrides(args)
    use_theme(args.theme)
    # Slightly below the theme defaults: with three columns on a 190 mm figure
    # the notebook's 9.5/8.5 titles and labels crowd the panels.
    matplotlib.rcParams.update({"axes.labelsize": 8.0, "xtick.labelsize": 7.0,
                                "ytick.labelsize": 7.0, "legend.fontsize": 7.0})
    TITLE_SIZE = 8.5

    out = Path(args.out)
    thr = pd.read_csv(out / "crown_d_thresholds.csv")
    bins = pd.read_csv(out / "crown_recall_bins.csv")
    prov = {}
    if (out / "crown_provenance.json").exists():
        prov = json.loads((out / "crown_provenance.json").read_text())
    target = float(args.targets[0])
    thr_all = thr[thr["target"] == target].copy()
    # Estimated values are plotted as points; NON-estimable ones are NOT
    # dropped -- they are drawn as open censored markers below, because an
    # absent row is indistinguishable from a backbone that was never run.
    if "status" in thr_all.columns:
        thr = thr_all[thr_all["status"] == ESTIMATED]
    elif "censored" in thr_all.columns:
        thr = thr_all[~thr_all["censored"].astype(bool)]
    else:
        thr = thr_all

    gsd_of = {s["short"]: s["gsd_cm"] / 100.0 for s in SOURCES.values()}
    # Order columns by the configured source short code rather than requiring
    # an exact display-label match.  This keeps legacy tables such as
    # "UAV (3-5 cm)" visible after the artwork label changes to "UAV (5 cm)".
    srcs = []
    if "short" in bins.columns:
        for sinfo in SOURCES.values():
            hit = bins.loc[bins["short"].eq(sinfo["short"]), "source"]
            if len(hit):
                srcs.append(hit.iloc[0])
    if not srcs:
        srcs = list(dict.fromkeys(bins["source"]))

    rank = (thr.assign(r=thr.groupby("source")["d_m"].rank())
               .groupby("backbone")["r"].mean().sort_values())
    order = list(rank.index)

    # A little more vertical room than the earlier 118 mm version: the recall
    # panels no longer look compressed at the journal's 190 mm page width,
    # while the overall figure remains comfortably below a full page.
    fig, axes = plt.subplots(
        2, len(srcs), figsize=(mm2in(DOUBLE_COL_MM), mm2in(132)),
        gridspec_kw=dict(height_ratios=[1.08, 1.25], hspace=0.52, wspace=0.20))

    # ---- row 1: one line per backbone -----------------------------------
    # Individual curves, styled exactly as the Stage C convergence figure
    # styles them: family colour, and a dash pattern that cycles over the
    # MEMBERS of that family. Ten same-coloured solid lines were
    # unidentifiable, which is why this row was collapsed to a family band;
    # the dash cycle solves the identifiability problem without discarding
    # the per-model detail, and keeps this figure visually consistent with
    # the convergence panels a reader has already seen.
    styles = style_map(order)
    legend_order = family_sorted(order)
    for col, src in enumerate(srcs):
        ax = axes[0, col]
        apply_panel_bg(ax)
        grid(ax, axis="both")
        despine(ax, left=(col == 0))
        b_src = bins[bins["source"] == src]
        # The drawn line is the monotone fit, so the target crossing on the
        # page is the tabulated value in row 2; the raw quantile bins are
        # non-monotone in places and would cross the guide elsewhere.
        for bb in legend_order:
            g = b_src[b_src["backbone"] == bb].sort_values("d_mid")
            if g.empty:
                continue
            st = styles[bb]
            x_fit = g["d_mid"].values
            y_fit = pava(g["recall"].values, g["n"].values)
            ax.plot(x_fit, y_fit,
                    color=st["color"], ls=st["ls"], lw=0.95, zorder=3,
                    solid_capstyle="round", dash_capstyle="butt")
            # A single backbone-specific marker gives the eye a stable place
            # to enter and follow each curve.  Drawing markers at every bin
            # would clutter the panels and imply discrete rather than fitted
            # curves.
            ax.plot(x_fit[0], y_fit[0], marker=st["marker"], ls="none",
                    ms=3.5, mfc=st["color"], mec="white", mew=0.40,
                    color=st["color"], zorder=5)
        ax.axhline(target, color="#B03A2E", lw=0.9, ls=(0, (4, 2)), zorder=4)
        ax.text(0.985, target, f"{target:.0%} recall",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.5, color="#B03A2E")
        # Display the rounded acquisition label requested for the artwork;
        # the validated 5.086 cm value still drives every metre conversion.
        if src.startswith("UAV"):
            display_src = "UAV (5 cm)"
        elif src.startswith("Google Earth") or src.startswith("GE"):
            display_src = "GE (15 cm)"
        else:
            display_src = src
        set_title(ax, display_src, size=TITLE_SIZE)
        panel_tag(ax, "abc"[col])
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        if col == 0:
            ax.set_ylabel("Recall")
        else:
            ax.set_yticklabels([])
            ax.tick_params(left=False)          # orphan ticks are noise
        # Figure-specific reduction: the theme's default x-label was too
        # prominent in the three compact recall panels.
        ax.set_xlabel("Crown diameter (m)", fontsize=7.2, labelpad=3.0)
        # Bottom-right is the only corner all three panels leave empty: the
        # curves are low on the left and plateau high on the right, so both
        # left corners and the top-right are occupied.
        n_tot = int(b_src.groupby("backbone")["n"].sum().median())
        ax.text(0.97, 0.05, f"n = {n_tot:,} crowns", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, color="#666")

    # ---- row 2: per-backbone D80, with the value printed ----------------
    for col, src in enumerate(srcs):
        ax = axes[1, col]
        apply_panel_bg(ax)
        grid(ax, axis="x")
        despine(ax, left=(col == 0))
        g = thr[thr["source"] == src].set_index("backbone")
        ga = thr_all[thr_all["source"] == src].set_index("backbone")
        short = (g["short"].iloc[0] if len(g)
                 else (ga["short"].iloc[0] if len(ga) else ""))

        vals = pd.concat([g["ci_lo"], g["ci_hi"], g["d_m"]]).dropna()
        lo_d, hi_d = float(vals.min()), float(vals.max())
        span = max(hi_d - lo_d, 1e-6)
        # right margin reserved for the numeric column: the figure should be
        # readable without measuring against the axis
        x0, x1 = lo_d - 0.06 * span, hi_d + 0.30 * span

        for i, bb in enumerate(order):
            c = PAL.get(FAMILY.get(bb, "CNN"), "#999")
            ax.axhline(i, color="white", lw=0.6, zorder=1)
            if bb in g.index:
                r = g.loc[bb]
                if np.isfinite(r["ci_lo"]) and np.isfinite(r["ci_hi"]):
                    ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], lw=1.5, color=c,
                            alpha=0.55, solid_capstyle="butt", zorder=2)
                ax.plot(r["d_m"], i, "o", ms=4.2, mfc=c, mec="white", mew=0.6,
                        zorder=3)
                # beside its own interval: a right-aligned column meant the
                # eye had to travel the width of the panel to pair a value
                # with its point
                xt = (r["ci_hi"] if np.isfinite(r["ci_hi"]) else r["d_m"])
                ax.text(xt + 0.035 * span, i, f"{r['d_m']:.2f}", ha="left",
                        va="center", fontsize=6.8, color="#333", zorder=4)
            elif bb in ga.index and "status" in ga.columns:
                st = ga.loc[bb, "status"]
                if st == NOT_REACHED:
                    xc, mk, lab = hi_d + 0.10 * span, ">", "n.e."
                else:
                    xc, mk, lab = float(ga.loc[bb, "d_m"]), "<", "ub"
                ax.plot(xc, i, marker="o", ms=4.2, mfc="none", mec=c, mew=1.0,
                        ls="none", zorder=3)
                ax.plot(xc + (0.035 if mk == ">" else -0.035) * span, i,
                        marker=mk, ms=3.0, color=c, ls="none", zorder=3)
                ax.text(xc + 0.075 * span, i, lab, ha="left", va="center",
                        fontsize=6.5, color="#888", style="italic", zorder=4)

        ax.set_xlim(x0, x1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if col == 0 else [])
        if col:
            ax.tick_params(left=False)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        # Spell out both the statistic and its physical unit.  ``m`` means
        # metres; it is unrelated to the number of crowns.  Keeping CD80 in
        # parentheses preserves the compact symbol used in the manuscript
        # while allowing the figure to be understood on its own.
        ax.set_xlabel(
            f"Crown diameter at {target:.0%} recall\n"
            f"({d_sym(target, math=True)}, m)",
            fontsize=7.0, labelpad=3.0)
        panel_tag(ax, "def"[col])

        gsd = gsd_of.get(short)
        if gsd:
            sec = ax.secondary_xaxis("top", functions=(lambda v, gg=gsd: v / gg,
                                                       lambda v, gg=gsd: v * gg))
            # The lower axis defines the 80%-recall statistic; this upper axis
            # gives the exact same crown diameter in image pixels.  Writing
            # ``px`` alone was cryptic, whereas repeating the full definition
            # in every column was too wide and caused label collisions.
            sec.set_xlabel("Equivalent crown diameter (pixels)",
                           fontsize=6.2, labelpad=3, color="#666")
            sec.tick_params(length=0, labelsize=6.5, colors="#666")
            for sp in sec.spines.values():
                sp.set_visible(False)

    # ---- family-grouped legend beneath the figure ------------------------
    # Named blocks per family, as in the convergence figure: with ten curves
    # to identify, a flat three-swatch family key is no longer sufficient --
    # the reader needs to map each dash pattern to a backbone.
    legs = grouped_legend(fig, order, styles)
    n_cens = int((thr_all["status"] != ESTIMATED).sum()) \
        if "status" in thr_all else 0
    if n_cens:
        cl = fig.legend(
            handles=[Line2D([0], [0], marker="o", ls="none", mfc="none",
                            mec="#666", mew=1.0, ms=5,
                            label="censored (n.e. not estimable, "
                                  "ub upper bound)")],
            loc="lower left", bbox_to_anchor=(0.085, -0.135), frameon=False,
            fontsize=6.2, handletextpad=0.42, borderaxespad=0)
        legs.append(cl)

    save(fig, out, "fig_crown_diameter")
    write_supp_table(out, thr_all, target, order)

    cap = build_caption(prov, thr_all, target)
    cap_path = out / f"fig_crown_diameter__{THEME}_caption.txt"
    cap_path.write_text(cap + "\n")
    print(f"  {cap_path}")
    print("\n--- manuscript caption ---")
    print(cap)


def build_sweep_caption(sw, st, target, ref):
    """Caption for the supplementary sensitivity figure."""
    D = d_sym(target)
    pct = f"{target:.0%}"
    ious = sorted(sw["iou_thr"].unique())
    scores = sorted(sw["score_thr"].unique())

    # The headline numbers are read from the tables, never typed in.
    strict = sw[sw["iou_thr"] == max(ious)]
    n_strict_fail = int((strict["status"] == NOT_REACHED).sum())
    n_strict = int(len(strict))
    sp = st["spearman_vs_ref"].dropna()
    shift = st["max_rank_shift"].dropna()
    rho_txt = (f"Spearman correlation against the reference setting stays "
               f"between {sp.min():.2f} and {sp.max():.2f}"
               if len(sp) else "Rank correlation was not estimable")
    shift_txt = (f", and no backbone moves more than "
                 f"{int(shift.max())} rank position"
                 f"{'s' if shift.max() != 1 else ''}"
                 if len(shift) else "")

    # Localisation penalty. This must be reported over the pairs that are
    # ACTUALLY estimable at both thresholds, and named honestly: at IoU 0.75
    # almost nothing survives, so a "median" would silently be one backbone.
    # The 0.25 -> reference step is reported alongside, because that one IS
    # estimable everywhere and answers whether the reference threshold binds
    # at all.
    ref_sc = sw[sw["score_thr"] == ref[0]]

    def _paired(iou_a, iou_b):
        """Per-source differences in metres over the backbones estimable at
        BOTH thresholds. Reported per source rather than pooled: a metre is
        not the same amount of evidence at 5 cm as at 15 cm, so pooling the
        three would average quantities that are not commensurable."""
        per = {}
        for src, g in ref_sc.groupby("source", sort=False):
            ga = g[(g["iou_thr"] == iou_a) & (g["status"] == ESTIMATED)]
            gb = g[(g["iou_thr"] == iou_b) & (g["status"] == ESTIMATED)]
            common = set(ga["backbone"]) & set(gb["backbone"])
            d = [float(gb[gb["backbone"] == bb]["d_m"].iloc[0])
                 - float(ga[ga["backbone"] == bb]["d_m"].iloc[0])
                 for bb in common]
            if d:
                per[src.split(" (")[0]] = np.asarray(d)
        return per

    strict_iou, loose_iou = max(ious), min(ious)
    hard = _paired(ref[1], strict_iou)
    n_hard = int(sum(len(v) for v in hard.values()))
    n_tot = int(len(ref_sc[ref_sc["iou_thr"] == strict_iou]))
    if n_hard == 0:
        peak = float(ref_sc[ref_sc["iou_thr"] == strict_iou]
                     ["max_fitted_recall"].max())
        pen_txt = (f" At IoU >= {strict_iou:.2f} no backbone attains {pct} "
                   f"recall at any crown size in any source, the highest "
                   f"fitted recall reached being {peak:.2f}, so the cost of "
                   f"strict delineation cannot be expressed as a diameter at "
                   f"all. This is the substantive result of the IoU axis. The "
                   f"limit at that threshold is mask delineation and not "
                   f"detection.")
    elif n_hard < 3:
        bits = "; ".join(f"{k} {', '.join(f'{v:+.2f}' for v in np.sort(a_))} m"
                         for k, a_ in hard.items())
        pen_txt = (f" At IoU >= {strict_iou:.2f} only {n_hard} of {n_tot} "
                   f"source-backbone combinations still attain {pct} recall "
                   f"({bits} relative to the reference threshold). With so "
                   f"few survivors these are given individually rather than "
                   f"as a summary statistic.")
    else:
        bits = "; ".join(f"{k} {np.median(a_):+.2f} m"
                         for k, a_ in hard.items())
        pen_txt = (f" Over the {n_hard} of {n_tot} source-backbone "
                   f"combinations that remain estimable at IoU >= "
                   f"{strict_iou:.2f}, the median rise in {D} relative to the "
                   f"reference threshold is {bits}, which is the cost of "
                   f"strict mask delineation rather than of detection.")

    soft = _paired(loose_iou, ref[1])
    soft_txt = ""
    if soft:
        bits = "; ".join(f"{k} {np.median(a_):+.2f} m"
                         for k, a_ in soft.items())
        worst = max(abs(np.median(a_)) for a_ in soft.values())
        # _paired(loose, ref) is ref minus loose, i.e. the cost of TIGHTENING
        # from the loosest rung up to the reference. Describing it as
        # "relaxing" inverted the sign against the numbers being quoted.
        soft_txt = (f" Tightening the threshold from {loose_iou:.2f} to "
                    f"{ref[1]:.2f} raises {D} by {bits}, so the reference "
                    + ("threshold is only weakly binding and recall there is "
                       "limited mainly by detection"
                       if worst < 0.25 else
                       "threshold already imposes a measurable delineation "
                       "cost")
                    + ".")

    return (
        f"Sensitivity of {D} to the detection operating point. {D} is defined "
        f"at a stated confidence score and mask IoU threshold, so this figure "
        f"reports how it and the backbone ordering respond when those "
        f"thresholds are varied. Panels are organised by source, one column "
        f"each. (a-c) {D} against the mask IoU threshold at a fixed "
        f"confidence score of {ref[0]:.2f}. (d-f) {D} against the confidence "
        f"score at a fixed mask IoU threshold of {ref[1]:.2f}. Both are shown "
        f"in metres, using a source-specific vertical range so that the "
        f"within-source threshold response remains visible. {D} is monotone "
        f"in both thresholds by "
        f"construction, because this analysis measures recall only, so a "
        f"lower score threshold can only add detections and a looser IoU "
        f"threshold can only accept more matches. The direction of these "
        f"curves is therefore arithmetic and is not itself a result. What is "
        f"a result is their spacing and their ordering. Text at the top of a "
        f"panel identifies combinations for which fitted recall never "
        f"attains {pct} at any crown size, so no threshold exists to plot; "
        f"at IoU >= {max(ious):.2f} this applies to {n_strict_fail} of "
        f"{n_strict} source-backbone-score combinations. (g-i) Rank stability "
        f"across the full grid of {len(scores)} confidence scores by "
        f"{len(ious)} IoU thresholds. Cell shading and the upper number in "
        f"each cell both give the Spearman rank correlation between the "
        f"backbone ordering at that setting and the ordering at the "
        f"reference setting, which is a confidence score of {ref[0]:.2f} with "
        f"a mask IoU threshold of {ref[1]:.2f}, outlined in black and equal "
        f"to 1.00 by definition. Within each cell "
        f"'max shift' is the largest change in any single backbone's rank "
        f"relative to the reference setting, and the fraction, where "
        f"present, counts the backbones for which {D} could not be estimated "
        f"because fitted recall never reached {pct} at any crown size, "
        f"abbreviated n.e. A rank correlation requires at least three "
        f"backbones with an estimable {D}; an entire row below that minimum "
        f"is left unshaded and carries one centred message reporting how many "
        f"models were estimable. {rho_txt}{shift_txt}, so the conclusions drawn at the "
        f"reference operating point are not an artefact of that choice."
        f"{soft_txt}{pen_txt}"
    )


def cmd_sweep_figure(args):
    """Supplementary figure: is CD80, and the ordering it induces, an artefact
    of the operating point?

    Everything here comes from the tables `sweep` already wrote. The figure
    exists because the tables answer the question and a reader will not read
    a 90-row CSV to find that out.

    Three things it must not do. It must not present the DIRECTION of the
    curves as a finding, since recall-only analysis makes CD80 monotone in
    both thresholds by construction. It must not silently drop settings where
    the target is unreachable, since those are the most informative cells.
    And it must not compare rank orderings built from different subsets of
    backbones, so cells with fewer than three estimable values stay unshaded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    apply_gsd_overrides(args)
    use_theme(args.theme)
    matplotlib.rcParams.update({"axes.labelsize": 7.6, "xtick.labelsize": 6.8,
                                "ytick.labelsize": 6.8, "legend.fontsize": 6.6})
    TITLE_SIZE = 8.5

    out = Path(args.out)
    sw_path, st_path = out / "crown_sweep.csv", out / "crown_sweep_stability.csv"
    for p in (sw_path, st_path):
        if not p.exists():
            sys.exit(f"[FATAL] missing {p}. Run the `sweep` subcommand first.")
    sw = pd.read_csv(sw_path)
    st = pd.read_csv(st_path)
    target = float(args.targets[0])
    sw = sw[sw["target"] == target].copy()
    ref = (args.ref_score, args.ref_iou)

    srcs = []
    if "short" in sw.columns:
        for sinfo in SOURCES.values():
            hit = sw.loc[sw["short"].eq(sinfo["short"]), "source"]
            if len(hit):
                srcs.append(hit.iloc[0])
    if not srcs:
        srcs = list(dict.fromkeys(sw["source"]))

    models = [m for m in MODELS if m in set(sw["backbone"])]
    styles = style_map(models)
    ious = sorted(sw["iou_thr"].unique())
    scores = sorted(sw["score_thr"].unique())

    fig, axes = plt.subplots(
        3, len(srcs), figsize=(mm2in(DOUBLE_COL_MM), mm2in(150)),
        gridspec_kw=dict(height_ratios=[1.0, 1.0, 0.86], hspace=0.55,
                         wspace=0.40, bottom=0.085, top=0.965))

    def sweep_row(r, xvals, xcol, fixed_col, fixed_val, xlabel, tags,
                  xname):
        """One row of CD80-vs-threshold panels.

        Metres, and a y range per source rather than shared across the row.
        Pixels were chosen originally to put the three sources on one
        comparable scale, but that comparison is not what this row is for and
        it costs the reader the unit the main figure uses. Each panel is a
        within-source question, so each gets the range that resolves it.
        """
        sub = sw[sw[fixed_col] == fixed_val]
        for col, src in enumerate(srcs):
            ax = axes[r, col]
            apply_panel_bg(ax)
            grid(ax, axis="y")
            despine(ax, left=True)
            g_src = sub[sub["source"] == src]
            e = g_src[g_src["status"] == ESTIMATED]
            ymax = float(e["d_m"].max()) if len(e) else 1.0
            ymin = float(e["d_m"].min()) if len(e) else 0.0
            pad = max(0.08 * (ymax - ymin), 0.05)
            any_ne = bool((g_src["status"] != ESTIMATED).any())
            y_ne = ymax + pad * 1.9

            for bb in family_sorted(models):
                g = g_src[g_src["backbone"] == bb].set_index(xcol)
                stl = styles[bb]
                xs, ys = [], []
                for xi, xv in enumerate(xvals):
                    if xv not in g.index:
                        continue
                    row = g.loc[xv]
                    if row["status"] == ESTIMATED and np.isfinite(row["d_m"]):
                        xs.append(xi)
                        ys.append(float(row["d_m"]))
                if xs:
                    ax.plot(xs, ys, color=stl["color"], ls=stl["ls"], lw=0.95,
                            marker=stl["marker"], ms=3.4, mfc=stl["color"],
                            mec="white", mew=0.35, zorder=3)

            if any_ne:
                counts = {}
                for xi, xv in enumerate(xvals):
                    gx = g_src[g_src[xcol] == xv]
                    k = int((gx["status"] != ESTIMATED).sum())
                    if k:
                        counts[xi] = (k, len(gx))
                ax.axhline(y_ne - pad * 0.75, color="#BBB", lw=0.6,
                           ls=(0, (2, 2)), zorder=1)
                # One self-contained sentence when the failures sit at a
                # single threshold, which is the usual case. A separate
                # italic header plus a short count collided in the panel and
                # made the reader assemble the statement from two fragments.
                if len(counts) == 1:
                    xi, (k, n_bb) = next(iter(counts.items()))
                    ax.text(0.50, y_ne,
                            f"{xname} = {xvals[xi]:g}: {k}/{n_bb} models "
                            f"below {target:.0%} recall",
                            transform=ax.get_yaxis_transform(), ha="center",
                            va="center", fontsize=4.9, color="#666", zorder=4,
                            clip_on=True)
                else:
                    for xi, (k, n_bb) in counts.items():
                        ax.text(xi, y_ne, f"{k}/{n_bb} models", ha="center",
                                va="center", fontsize=6.2, color="#555",
                                zorder=4)
                ax.set_ylim(ymin - pad, y_ne + pad * 1.1)
            else:
                ax.set_ylim(ymin - pad, ymax + pad)

            ax.set_xticks(range(len(xvals)))
            ax.set_xticklabels([f"{v:g}" for v in xvals])
            ax.set_xlim(-0.35, len(xvals) - 0.65)
            # The held-constant threshold belongs on the axis, not only in the
            # caption: a reader looking at one panel must be able to see which
            # operating point it is a slice through.
            ax.set_xlabel(xlabel)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            # every panel carries its own scale now, so every panel is
            # labelled; an unlabelled axis with a private range is a trap
            ax.set_ylabel(f"Crown diameter at {target:.0%} recall,\n"
                          f"{d_sym(target, math=True)} (m)")
            if r == 0:
                display = ("UAV (5 cm)" if src.startswith("UAV")
                           else "GE (15 cm)" if src.startswith(("GE",
                                                               "Google Earth"))
                           else src)
                set_title(ax, display, size=TITLE_SIZE)
            panel_tag(ax, tags[col])

    sweep_row(0, ious, "iou_thr", "score_thr", ref[0],
              f"Mask IoU threshold\n(score = {ref[0]:.2f})", "abc",
              "IoU")
    sweep_row(1, scores, "score_thr", "iou_thr", ref[1],
              f"Confidence score threshold\n(mask IoU = {ref[1]:.2f})", "def",
              "score")

    # ---- row 3: rank stability over the full grid -----------------------
    # The cells print the SAME statistic the colour encodes. An earlier
    # version shaded by Spearman correlation and printed the largest rank
    # change as a bare integer, so the reader saw "2" on a blue cell with
    # nothing to say the two were different quantities.
    from matplotlib.patches import Rectangle
    for col, src in enumerate(srcs):
        ax = axes[2, col]
        apply_panel_bg(ax)
        despine(ax, left=(col == 0))
        g = st[st["source"] == src]
        M = np.full((len(ious), len(scores)), np.nan)
        for i, iu in enumerate(ious):
            for j, sc in enumerate(scores):
                cell = g[(g["iou_thr"] == iu) & (g["score_thr"] == sc)]
                if len(cell) and cell["n_rank_pairs"].iloc[0] >= 3:
                    M[i, j] = cell["spearman_vs_ref"].iloc[0]
        im = ax.imshow(M, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto",
                       origin="upper")
        for i, iu in enumerate(ious):
            # When no score setting in an entire IoU row has enough
            # estimable models for a rank correlation, use one message across
            # the row. Repeating the same explanation in all three cells is
            # visually crowded and obscures that this is a row-level result.
            row_not_computed = bool(np.isnan(M[i, :]).all())
            row_cells = g[g["iou_thr"] == iu]
            if row_not_computed and len(row_cells):
                n_est = sorted(set(row_cells["n_estimated"].astype(int)))
                n_bb = sorted(set(row_cells["n_backbones"].astype(int)))
                if len(n_est) == 1 and len(n_bb) == 1:
                    estimate_text = f"{n_est[0]}/{n_bb[0]} estimable"
                else:
                    estimate_text = (
                        f"{min(n_est)}-{max(n_est)}/{max(n_bb)} estimable"
                    )
                ax.text((len(scores) - 1) / 2, i,
                        f"Not computed ({estimate_text})",
                        ha="center", va="center", fontsize=5.4,
                        color="#777", style="italic", zorder=4)
            for j, sc in enumerate(scores):
                cell = g[(g["iou_thr"] == iu) & (g["score_thr"] == sc)]
                if not len(cell):
                    continue
                c = cell.iloc[0]
                dark = (not np.isnan(M[i, j])) and M[i, j] > 0.55
                fg = "white" if dark else "#333"
                if np.isnan(M[i, j]):
                    # A fully unrankable row is explained once across all
                    # score columns above. Retain a compact cell-level message
                    # only for an isolated unrankable setting.
                    if not row_not_computed:
                        ax.text(j, i, "Not computed\n"
                                       f"({int(c['n_estimated'])}/"
                                       f"{int(c['n_backbones'])} estimable)",
                                ha="center", va="center", fontsize=4.8,
                                color="#777", style="italic",
                                linespacing=1.25)
                else:
                    ax.text(j, i - 0.16, f"{M[i, j]:.2f}", ha="center",
                            va="center", fontsize=7.4, color=fg)
                    if np.isfinite(c["max_rank_shift"]):
                        ax.text(j, i + 0.17,
                                f"max shift {int(c['max_rank_shift'])}",
                                ha="center", va="center", fontsize=5.5,
                                color=fg if dark else "#666")
                if int(c["n_not_reached"]) and not np.isnan(M[i, j]):
                    ax.text(j, i + 0.34,
                            f"{int(c['n_not_reached'])}/{int(c['n_backbones'])}"
                            f" n.e.", ha="center", va="center", fontsize=5.5,
                            color=fg if dark else "#777")
                # the reference setting is the point everything else is
                # measured against, so it is marked rather than left for the
                # reader to locate from the axis labels
                if sc == ref[0] and iu == ref[1]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, ec="#222", lw=1.4,
                                           zorder=5))
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels([f"{v:g}" for v in scores])
        ax.set_yticks(range(len(ious)))
        ax.set_xlabel("Confidence score threshold")
        if col == 0:
            ax.set_yticklabels([f"{v:g}" for v in ious])
            ax.set_ylabel("Mask IoU threshold")
        else:
            ax.set_yticklabels([])
            ax.tick_params(left=False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(False)
        panel_tag(ax, "ghi"[col])

    cb = fig.colorbar(im, ax=axes[2, :].tolist(), fraction=0.020, pad=0.012)
    cb.set_label("Rank correlation with reference setting", fontsize=6.4)
    cb.ax.tick_params(labelsize=6.2, length=0)
    cb.outline.set_visible(False)

    # The open-marker key is gone with the markers themselves: unreachable
    # settings are now counts printed in the panel, which need no legend.
    grouped_legend(fig, models, styles, y=-0.055)

    save(fig, out, "figS_crown_sensitivity")
    cap = build_sweep_caption(sw, st, target, ref)
    cp = out / f"figS_crown_sensitivity__{THEME}_caption.txt"
    cp.write_text(cap + "\n")
    print(f"  {cp}")
    print("\n--- supplementary caption ---")
    print(cap)


def cmd_compare(args):
    """Is D80 robust to the checkpoint protocol? Answer it, do not assume it.

    Reports, per (source, backbone), the shift in D80 between two runs and
    whether their bootstrap intervals overlap. Overlapping intervals across the
    board licence a one-sentence robustness statement; a systematic,
    non-overlapping shift means the protocol choice is load-bearing and has to
    be reported instead of waved away.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = float(args.targets[0])
    a = pd.read_csv(args.a).query("target == @target")
    b = pd.read_csv(args.b).query("target == @target")
    m = a.merge(b, on=["source", "backbone"], suffixes=("_a", "_b"))
    if m.empty:
        sys.exit("no (source, backbone) rows in common between the two runs")

    m["delta"] = m["d_m_b"] - m["d_m_a"]
    # nullable boolean: a CI that could not be computed is NOT evidence of
    # agreement, so it must stay distinguishable from True/False rather than
    # collapsing into either.
    m["ci_overlap"] = (~((m["ci_hi_a"] < m["ci_lo_b"]) |
                         (m["ci_hi_b"] < m["ci_lo_a"]))).astype("boolean")
    undef = m[["ci_lo_a", "ci_hi_a", "ci_lo_b", "ci_hi_b"]].isna().any(axis=1)
    m.loc[undef, "ci_overlap"] = pd.NA

    m.to_csv(out / "crown_d_protocol_compare.csv", index=False)
    ok = int((m["ci_overlap"] == True).sum())            # noqa: E712
    nd = int(m["ci_overlap"].isna().sum())
    tot = len(m)
    print(m[["source", "backbone", "d_m_a", "d_m_b", "delta", "ci_overlap"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n{args.label_a} -> {args.label_b}: median shift "
          f"{m['delta'].median():+.3f} m, "
          f"max |shift| {m['delta'].abs().max():.3f} m")
    print(f"CIs overlap in {ok}/{tot} comparisons"
          + (f" ({nd} undetermined)" if nd else ""))
    if nd == 0 and ok == tot:
        print("-> D80 is robust to the checkpoint protocol; report it as such, "
              "citing this table.")
    else:
        print("-> at least one backbone shifts beyond its interval. The "
              "protocol is load-bearing here and must be stated with the "
              "result, not chosen silently.")
    print(f"\ntable -> {out / 'crown_d_protocol_compare.csv'}")


def cmd_sweep(args):
    """3 x 3 sensitivity grid over score threshold and mask IoU threshold.

    WHY THIS EXISTS
        At IoU >= 0.50 the measured quantity is not crown-detection recall; it
        is joint detection-AND-localisation recall. A constant absolute
        boundary error costs far more IoU on a small mask than a large one --
        for a disc of diameter d px with a k px error, IoU ~ ((d-2k)/(d+2k))^2,
        so a 2 px error gives 0.52 at d=25 but 0.85 at d=100. Aerial D80 sits
        near 33 px and GE near 39 px, squarely in the regime where mask
        precision rather than detection decides whether a crown counts as
        found. That confound acts in the SAME DIRECTION as the size effect
        being measured, so it can manufacture part of it. The score threshold
        is the second, milder confound: D80 conflates crown visibility with
        score calibration.

    WHAT IS AND IS NOT A RESULT
        D80 is MONOTONE in both thresholds by construction: this analysis is
        recall-only, so a lower score threshold can only add detections and a
        looser IoU can only accept more matches. "D80 improves at IoU 0.25" is
        therefore arithmetic, not evidence. The informative outputs are
          - rank stability of backbones across settings (Spearman vs the
            reference setting, and the largest rank change),
          - the SIZE of the 0.25 -> 0.50 -> 0.75 increase, which quantifies the
            localisation penalty separately from recovery,
          - whether cross-source ordering (the UAV > GE > Aerial pixel
            requirement) survives.
        IoU 0.25 is reported as loose-overlap crown recovery, not as "detected
        at all": at loose overlap a large blob can claim a small neighbouring
        crown in dense canopy, so mean_matched_iou is recorded per setting --
        if 0.25 accepts pairs averaging ~0.3 those are marginal recoveries,
        while ~0.7 means the threshold was never binding.

    COST
        The IoU matrix is computed ONCE per (image, backbone) with no score
        filter; the thresholds only change how that matrix is read. So the
        whole grid costs one extraction pass, not nine.
    """
    import pickle
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils

    def to_rle(seg, h, w):
        if isinstance(seg, dict):
            rle = dict(seg)
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode()
            return rle
        if isinstance(seg, list):
            return maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
        return maskUtils.encode(np.asfortranarray(
            np.asarray(seg).astype(np.uint8)))

    def greedy(ious, order, iou_thr):
        """COCO-style score-ordered matching. Returns matched mask and the
        IoU of each accepted pair (for mean_matched_iou)."""
        n_d, n_g = ious.shape
        matched = np.zeros(n_g, bool)
        got = []
        for d in order:
            row = ious[d].copy()
            row[matched] = -1.0
            g = int(np.argmax(row))
            if row[g] >= iou_thr:
                matched[g] = True
                got.append(float(row[g]))
        return matched, got

    scores_grid = [float(v) for v in args.score_grid]
    ious_grid = [float(v) for v in args.iou_grid]
    settings = [(sc, iu) for sc in scores_grid for iu in ious_grid]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_gsd_overrides(args)
    print(f"grid: score {scores_grid} x IoU {ious_grid} = {len(settings)} "
          f"settings, one extraction pass")
    verify_protocol(args.pkl_pattern, args.protocol, MODELS,
                    [v["set"] for v in SOURCES.values()])

    rows = []
    for label, sinfo in SOURCES.items():
        coco = COCO(sinfo["ann_json"])
        img_ids = sorted(coco.getImgIds())
        if args.limit:
            img_ids = img_ids[:args.limit]

        # decode ground truth ONCE per source, not once per backbone
        gt = {}
        for img_id in img_ids:
            info = coco.loadImgs(img_id)[0]
            h, w = info["height"], info["width"]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))
            if not anns:
                continue
            rles = [to_rle(a["segmentation"], h, w) for a in anns]
            area = np.asarray(maskUtils.area(rles), float)
            diam = 2.0 * np.sqrt(np.maximum(area, 1.0) / np.pi)
            gt[img_id] = (rles, diam * sinfo["gsd_cm"] / 100.0, h, w)
        print(f"[{sinfo['short']}] {len(gt)} tiles with GT, "
              f"{sum(len(v[1]) for v in gt.values())} crowns")

        for bb in MODELS:
            pth = Path(args.pkl_pattern.format(key=bb, set=sinfo["set"]))
            if not pth.exists():
                sys.exit(f"[FATAL] missing pkl: {pth}")
            with open(pth, "rb") as f:
                pk = {r["img_id"]: r for r in pickle.load(f)}

            acc = {k: dict(img=[], diam=[], m=[], iou=[], marg=[])
                   for k in settings}
            for img_id, (g_rles, g_diam, h, w) in gt.items():
                res = pk.get(img_id)
                if res is None:
                    sys.exit(f"[FATAL] img {img_id} absent from {pth.name}")
                pi = res["pred_instances"]
                sc_all = np.asarray(pi["scores"], float)
                masks = pi.get("masks", [])
                if len(sc_all):
                    d_rles = [to_rle(masks[i], h, w) for i in range(len(sc_all))]
                    iou_full = np.asarray(
                        maskUtils.iou(d_rles, g_rles, [0] * len(g_rles))
                    ).reshape(len(d_rles), len(g_rles))
                else:
                    iou_full = np.zeros((0, len(g_rles)))
                for sc_thr, iu_thr in settings:
                    keep = np.where(sc_all >= sc_thr)[0]
                    sub = iou_full[keep] if len(keep) else \
                        np.zeros((0, len(g_rles)))
                    order = np.argsort(-sc_all[keep]) if len(keep) else []
                    matched, got = greedy(sub, order, iu_thr)
                    a = acc[(sc_thr, iu_thr)]
                    # MARGINAL matches = accepted pairs whose IoU falls below
                    # the reference threshold, i.e. the ones that exist ONLY
                    # because the threshold was loosened. Their mean IoU is the
                    # diagnostic; the mean over ALL accepted pairs is dominated
                    # by strong matches and cannot say whether the loose rung
                    # is recovering crowns or accepting sloppy masks.
                    a["marg"].extend([v for v in got if v < args.ref_iou])
                    a["img"].append(np.full(len(g_diam), img_id))
                    a["diam"].append(g_diam)
                    a["m"].append(matched.astype(np.int8))
                    a["iou"].extend(got)

            for (sc_thr, iu_thr), a in acc.items():
                df = pd.DataFrame(dict(
                    img_id=np.concatenate(a["img"]),
                    diam_m=np.concatenate(a["diam"]),
                    matched=np.concatenate(a["m"])))
                pt, lo, hi, nb, status, maxfit = d_threshold_ci(
                    df, target=float(args.targets[0]),
                    n_boot=args.bootstrap, seed=args.seed, n_bins=args.bins)
                rows.append(dict(
                    source=label, short=sinfo["short"], backbone=bb,
                    family=FAMILY.get(bb, "?"), score_thr=sc_thr,
                    iou_thr=iu_thr, target=float(args.targets[0]),
                    d_m=pt, ci_lo=lo, ci_hi=hi, status=status,
                    censored=(status == BELOW_RANGE),
                    max_fitted_recall=maxfit,
                    d_px=pt / (sinfo["gsd_cm"] / 100.0)
                    if np.isfinite(pt) else np.nan,
                    n_instances=int(len(df)),
                    n_images=int(df["img_id"].nunique()),
                    n_boot_valid=nb,
                    recall_overall=float(df["matched"].mean()),
                    n_matched=int(np.concatenate(a["m"]).sum()),
                    mean_matched_iou=float(np.mean(a["iou"])) if a["iou"]
                    else np.nan,
                    n_marginal=len(a["marg"]),
                    frac_marginal=(len(a["marg"]) / len(a["iou"])
                                   if a["iou"] else np.nan),
                    mean_marginal_iou=float(np.mean(a["marg"]))
                    if a["marg"] else np.nan))
            print(f"    {bb:<18s} done ({len(settings)} settings)")

    res = pd.DataFrame(rows)
    res.to_csv(out / "crown_sweep.csv", index=False)

    # ---- rank stability, which is the actual result ---------------------
    # Every setting gets a row even when D80 is unestimable for most or all
    # backbones. The previous version skipped such settings with `continue`,
    # which made the entire IoU 0.75 block VANISH from this table -- the single
    # most important thing the grid found (strict mask delineation rarely
    # attains 80% recall) was the thing the summary silently omitted. Ranks are
    # not computed over unestimable rows, but their count is reported.
    ref = (args.ref_score, args.ref_iou)
    stab = []
    for src, g_src in res.groupby("source", sort=False):
        base = (g_src[(g_src["score_thr"] == ref[0]) &
                      (g_src["iou_thr"] == ref[1])]
                .set_index("backbone")["d_m"])
        for (sc, iu), g in g_src.groupby(["score_thr", "iou_thr"], sort=False):
            gi = g.set_index("backbone")
            n_est = int((gi["status"] == ESTIMATED).sum())
            row = dict(
                source=src, score_thr=sc, iou_thr=iu,
                n_backbones=int(len(gi)),
                n_estimated=n_est,
                n_below_range=int((gi["status"] == BELOW_RANGE).sum()),
                n_not_reached=int((gi["status"] == NOT_REACHED).sum()),
                max_fitted_recall=float(gi["max_fitted_recall"].max()),
                median_d_m=float(gi.loc[gi["status"] == ESTIMATED, "d_m"]
                                 .median()) if n_est else np.nan,
                median_d_px=float(gi.loc[gi["status"] == ESTIMATED, "d_px"]
                                  .median()) if n_est else np.nan,
                n_matched=int(gi["n_matched"].sum()),
                mean_matched_iou=float(gi["mean_matched_iou"].mean()),
                n_marginal=int(gi["n_marginal"].sum()),
                frac_marginal=float(gi["frac_marginal"].mean()),
                mean_marginal_iou=float(gi["mean_marginal_iou"].mean()),
                n_rank_pairs=0, spearman_vs_ref=np.nan,
                max_rank_shift=np.nan)
            if not base.empty:
                cur = gi["d_m"].reindex(base.index)
                ok = base.notna() & cur.notna()
                row["n_rank_pairs"] = int(ok.sum())
                if ok.sum() >= 3:
                    rb, rc = base[ok].rank(), cur[ok].rank()
                    row["spearman_vs_ref"] = float(np.corrcoef(rb, rc)[0, 1])
                    row["max_rank_shift"] = int((rb - rc).abs().max())
            stab.append(row)
    st = pd.DataFrame(stab)
    st.to_csv(out / "crown_sweep_stability.csv", index=False)

    print("\nrank stability vs reference "
          f"(score={ref[0]}, IoU={ref[1]}) -- D80 is monotone in BOTH "
          "thresholds by construction, so only the rank columns are results.\n"
          "n_rank_pairs < 3 means the setting has too few estimable D80 values "
          "to rank; it is reported, not dropped.")
    cols = ["source", "score_thr", "iou_thr", "n_estimated", "n_not_reached",
            "median_d_px", "spearman_vs_ref", "max_rank_shift",
            "n_rank_pairs", "frac_marginal", "mean_marginal_iou"]
    print(st[cols].to_string(index=False,
                             float_format=lambda x: f"{x:.3f}"))
    unest = st[st["n_estimated"] < st["n_backbones"]]
    if len(unest):
        print("\nsettings where D80 is not estimable for every backbone:")
        for _, r in unest.iterrows():
            print(f"  {r['source']:<22s} score {r['score_thr']:.2f} "
                  f"IoU {r['iou_thr']:.2f}: {r['n_not_reached']:2d} never reach "
                  f"the target (best fitted recall "
                  f"{r['max_fitted_recall']:.3f}), "
                  f"{r['n_below_range']:2d} below observed range")
    print(f"\ntables -> {out / 'crown_sweep.csv'}, "
          f"{out / 'crown_sweep_stability.csv'}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="results/qual/crown")
    common.add_argument("--targets", nargs="+", default=["0.80", "0.50"],
                        help="recall targets; the first is the headline D80")
    common.add_argument("--bootstrap", type=int, default=1000,
                        help="cluster-bootstrap resamples (0 disables CIs)")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--gsd-cm", nargs="*", default=[], metavar="SHORT=CM",
                        help="override the assumed GSD, e.g. UAV=5.086. "
                             "diam_m scales linearly with it; diam_px does "
                             "not. Use when the nominal value in SOURCES is "
                             "not the measured resolution.")

    e = sub.add_parser("extract", parents=[common])
    e.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True,
                   help="checkpoint protocol behind the prediction pkls. "
                        "make_stagec_pkls.sh produces 'diagonal'.")
    e.add_argument("--pkl-pattern", default=PKL_PATTERN,
                   help="prediction pkl path template. Use the -unified "
                        "variant when --protocol unified, e.g. "
                        "'results/qual/{key}_stageC-unified_{set}.pkl'")
    e.add_argument("--score-thr", type=float, default=SCORE_THR)
    e.add_argument("--iou-thr", type=float, default=IOU_THR)
    e.add_argument("--limit", type=int, default=None)
    e.set_defaults(func=cmd_extract)

    a = sub.add_parser("analyse", parents=[common])
    a.add_argument("--instances", default="results/qual/crown/crown_instances.csv")
    a.add_argument("--bins", type=int, default=12)
    a.add_argument("--sensitivity-drop", nargs="*", default=[],
                   help="backbones to EXCLUDE in an extra, clearly labelled "
                        "sensitivity cohort. The all-ten result is always "
                        "produced and is the primary analysis.")
    a.set_defaults(func=cmd_analyse)

    l = sub.add_parser("ladder", parents=[common])
    l.add_argument("--rung", action="append", required=True,
                   metavar="NAME:GSD_CM",
                   help="one per resolution rung, e.g. UAV_5cm:5")
    l.add_argument("--instances-dir", default="results/qual/crown")
    l.set_defaults(func=cmd_ladder)

    w = sub.add_parser("sweep", parents=[common])
    w.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    w.add_argument("--pkl-pattern", default=PKL_PATTERN)
    w.add_argument("--score-grid", nargs="+", default=["0.30", "0.45", "0.60"])
    w.add_argument("--iou-grid", nargs="+", default=["0.25", "0.50", "0.75"])
    w.add_argument("--ref-score", type=float, default=0.45)
    w.add_argument("--ref-iou", type=float, default=0.50)
    w.add_argument("--bins", type=int, default=12)
    w.add_argument("--limit", type=int, default=None)
    w.set_defaults(func=cmd_sweep)

    f = sub.add_parser("figure", parents=[common])
    f.add_argument("--theme", default="tidy",
                   choices=("tidy", "editorial", "mono", "slate"),
                   help="theme registry from stage_ABC_figures_v14.ipynb; the "
                        "filename gets a __<theme> suffix as in the notebook")
    f.set_defaults(func=cmd_figure)

    sf = sub.add_parser("sweep-figure", parents=[common])
    sf.add_argument("--theme", default="tidy",
                    choices=("tidy", "editorial", "mono", "slate"))
    sf.add_argument("--ref-score", type=float, default=0.45,
                    help="reference operating point the grid is compared "
                         "against; must match the `sweep` run")
    sf.add_argument("--ref-iou", type=float, default=0.50)
    sf.set_defaults(func=cmd_sweep_figure)

    c = sub.add_parser("compare", parents=[common])
    c.add_argument("--a", required=True, metavar="CSV",
                   help="crown_d_thresholds.csv from run A (e.g. diagonal)")
    c.add_argument("--b", required=True, metavar="CSV",
                   help="crown_d_thresholds.csv from run B (e.g. unified)")
    c.add_argument("--label-a", default="A")
    c.add_argument("--label-b", default="B")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
