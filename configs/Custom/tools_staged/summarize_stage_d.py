#!/usr/bin/env python3
# =============================================================================
# summarize_stage_d.py
# -----------------------------------------------------------------------------
# Tabulate every Stage D run from its own logs: best score, the iteration it
# peaked at, where it stopped, and -- the part that decides what the paper may
# claim -- the size of the late-training noise band.
#
# WHY THE NOISE BAND IS REPORTED
#   `save_best` keeps the maximum over every validation, and a run here is
#   validated ~39 times. The maximum of 39 noisy draws is biased upward, so two
#   arms differing by less than the oscillation of a single run are not
#   distinguishable, however clean the two numbers look side by side. Measured
#   on ConvNeXt-T b0: after convergence the curve moves between 0.768 and 0.802,
#   with 0.028 between adjacent evaluations. Any Stage D claim resting on a gap
#   smaller than that band is a claim about seed noise.
#
#   The table therefore reports, alongside the peak: the mean and standard
#   deviation of the post-convergence plateau, and the plateau range. Compare
#   arms on those, not on the peak alone.
#
# THE ITERATION NUMBER IS DERIVED, NOT READ
#   MMEngine writes validation rows into vis_data/scalars.json with "step": 0
#   for every row -- the counter is never advanced -- so the file records the
#   scores in order but not the iteration they belong to. The iteration is
#   therefore (index + 1) * val_interval.
#
#   That derivation is not taken on trust. val_interval comes from the config
#   dumped into the work_dir, and the result is cross-checked against the
#   iteration in the best_*.pth filename, which MMEngine wrote independently.
#   A mismatch is printed loudly rather than silently tolerated: it would mean
#   the interval changed mid-run (a resume with a different config), and every
#   iteration in that row would be wrong.
#
# RESUMED RUNS
#   A run resumed after a crash writes a second timestamp directory. Their
#   curves are concatenated in timestamp order, which is right for a clean
#   resume and wrong if the resume re-validated iterations the first attempt had
#   already covered. Runs built from more than one timestamp directory are
#   flagged so the overlap can be checked before the numbers are used.
#
# USAGE
#   python configs/Custom/tools_staged/summarize_stage_d.py
#   python configs/Custom/tools_staged/summarize_stage_d.py --curve
#   python configs/Custom/tools_staged/summarize_stage_d.py --csv out.csv
#   python configs/Custom/tools_staged/summarize_stage_d.py --root work_dirs/Stage_C
# =============================================================================

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

METRIC_SUFFIX = "segm_mAP_50"
VAL_INTERVAL_RE = re.compile(r"val_interval\s*=\s*(\d+)")
CKPT_ITER_RE = re.compile(r"_iter_(\d+)\.pth$")


def _metric_key(row: dict):
    """The validation metric in this row, or None if it is a training row."""
    keys = [k for k in row if k.endswith(METRIC_SUFFIX)]
    if not keys:
        return None
    # Prefer the plain COCO metric when a run also logs per-sensor variants.
    for k in keys:
        if k.startswith("coco/"):
            return k
    return sorted(keys)[0]


def read_curve(run_dir: Path):
    """Ordered validation scores for one run, merged across timestamp dirs."""
    ts_dirs = sorted(d for d in run_dir.iterdir()
                     if d.is_dir() and (d / "vis_data" / "scalars.json").exists())
    scores: list[float] = []
    key = None
    bad = 0
    for ts in ts_dirs:
        for line in (ts / "vis_data" / "scalars.json").read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            k = _metric_key(row)
            if k is None:
                continue
            key = key or k
            scores.append(float(row[k]))
    return key, scores, len(ts_dirs), bad


def val_interval_of(run_dir: Path):
    """val_interval from the config MMEngine dumped into the work_dir."""
    cfgs = list(run_dir.glob("*.py"))
    if not cfgs:
        return None
    m = VAL_INTERVAL_RE.search(cfgs[0].read_text())
    return int(m.group(1)) if m else None


def best_ckpt_iter(run_dir: Path):
    """The iteration save_best kept, parsed from the checkpoint filename."""
    hits = sorted(run_dir.glob(f"best_*{METRIC_SUFFIX}_iter_*.pth"))
    if not hits:
        return None
    m = CKPT_ITER_RE.search(hits[-1].name)
    return int(m.group(1)) if m else None


def plateau_stats(scores, warmup_frac=0.25):
    """Mean, sd and range over the converged tail of the curve.

    The first quarter of the run is discarded as the rise; what remains is the
    band the model oscillates in, which is the scale any between-arm difference
    has to beat to mean anything.
    """
    if len(scores) < 8:
        return None
    tail = scores[max(1, int(len(scores) * warmup_frac)):]
    return dict(mean=statistics.fmean(tail),
                sd=statistics.pstdev(tail),
                lo=min(tail), hi=max(tail), n=len(tail))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="work_dirs/Stage_D")
    ap.add_argument("--curve", action="store_true",
                    help="print every validation point")
    ap.add_argument("--csv")
    ap.add_argument("--warmup-frac", type=float, default=0.25,
                    help="fraction of the curve treated as the rise and excluded "
                         "from the plateau statistics (default 0.25)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")

    rows, notes = [], []
    for run in sorted(d for d in root.iterdir() if d.is_dir()):
        key, scores, n_ts, bad = read_curve(run)
        if not scores:
            continue

        interval = val_interval_of(run)
        ckpt = best_ckpt_iter(run)
        peak_idx = max(range(len(scores)), key=lambda i: scores[i])

        # Cross-check the derived iteration against the checkpoint filename.
        if interval is None and ckpt is not None:
            if ckpt % (peak_idx + 1) == 0:
                interval = ckpt // (peak_idx + 1)
                notes.append(f"{run.name}: val_interval not in config; inferred "
                             f"{interval} from the checkpoint filename")
        derived = interval * (peak_idx + 1) if interval else None
        if derived is not None and ckpt is not None and derived != ckpt:
            notes.append(f"{run.name}: (!) derived peak iter {derived} != "
                         f"checkpoint {ckpt}. val_interval may have changed "
                         f"mid-run; treat every iteration for this run as "
                         f"unverified.")
        if n_ts > 1:
            notes.append(f"{run.name}: {n_ts} timestamp dirs concatenated "
                         f"(resumed run) -- check for re-validated overlap")
        if bad:
            notes.append(f"{run.name}: {bad} unparseable line(s) in scalars.json")

        pl = plateau_stats(scores, args.warmup_frac)
        rows.append(dict(
            run=run.name, metric=key, best=scores[peak_idx],
            best_iter=derived if derived is not None else "?",
            ckpt_iter=ckpt if ckpt is not None else "-",
            stopped=interval * len(scores) if interval else "?",
            n_evals=len(scores),
            plateau_mean=round(pl["mean"], 4) if pl else "",
            plateau_sd=round(pl["sd"], 4) if pl else "",
            plateau_lo=pl["lo"] if pl else "", plateau_hi=pl["hi"] if pl else ""))

        if args.curve:
            print(f"\n=== {run.name}  ({key})")
            for i, v in enumerate(scores):
                it = interval * (i + 1) if interval else i + 1
                print(f"    {it:>7}  {v:.4f}{'   <- best' if i == peak_idx else ''}")

    if not rows:
        raise SystemExit(f"no runs with validation scalars found under {root}")

    rows.sort(key=lambda r: -r["best"])
    w = max(len(r["run"]) for r in rows)
    print(f"\n{'run'.ljust(w)}   best   @iter    ckpt  stopped  n   "
          f"plateau mean±sd      range")
    print("-" * (w + 62))
    for r in rows:
        band = (f"{r['plateau_lo']:.3f}-{r['plateau_hi']:.3f}"
                if r["plateau_mean"] != "" else "")
        ms = (f"{r['plateau_mean']:.4f}±{r['plateau_sd']:.4f}"
              if r["plateau_mean"] != "" else "")
        print(f"{r['run'].ljust(w)}  {r['best']:.4f}  {str(r['best_iter']):>6}  "
              f"{str(r['ckpt_iter']):>6}  {str(r['stopped']):>7}  {r['n_evals']:>3}  "
              f"{ms:>17}  {band}")

    sds = [r["plateau_sd"] for r in rows if r["plateau_sd"] != ""]
    if sds:
        print(f"\nTypical late-training oscillation: sd {min(sds):.4f}-{max(sds):.4f}.")
        print("Two arms differing by less than roughly 2x the larger sd are not")
        print("separated by these runs; report such pairs as comparable rather")
        print("than ranked.")

    if notes:
        print("\nNotes:")
        for n in dict.fromkeys(notes):
            print(f"  {n}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
