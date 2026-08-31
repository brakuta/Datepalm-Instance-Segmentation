#!/usr/bin/env python3
# ==========================================================================
# compile_validation.py
# --------------------------------------------------------------------------
# Assemble the per-(model, validation-set) CSVs written by
# evaluate_validation.py into:
#   * validation_summary.csv        — one row per (stage, model, val_set)
#   * validation_by_stage_<S>.csv   — wide per-stage table (model x metric)
#   * validation_tables.tex         — publication LaTeX tables per stage
#
# Primary reported metrics: segm mAP@0.50 and segm F1 (test-swept optimal).
# bbox columns are retained for completeness.
# ==========================================================================

import argparse
import glob
import os
import os.path as osp

import pandas as pd

FAM_ORDER = {'CNN': 0, 'Transformer': 1, 'Mamba': 2, '': 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results/validation')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    rd = args.results_dir
    out = args.out_dir or osp.join(rd, 'compiled')
    os.makedirs(out, exist_ok=True)

    files = [f for f in glob.glob(osp.join(rd, '*.csv'))
             if osp.basename(f) not in ('validation_summary.csv',)]
    if not files:
        raise SystemExit(f'[ERROR] no per-cell CSVs in {rd}')

    rows = []
    for f in files:
        try:
            rows.append(pd.read_csv(f))
        except Exception as e:
            print(f'[warn] skip {f}: {e}')
    df = pd.concat(rows, ignore_index=True)

    # tidy + order
    df['fam_rank'] = df['family'].map(lambda x: FAM_ORDER.get(str(x), 9))
    df = df.sort_values(['stage', 'fam_rank', 'model', 'val_set']).drop(columns='fam_rank')

    summ = osp.join(out, 'validation_summary.csv')
    keep = ['stage', 'model', 'family', 'val_set',
            'segm_mAP50', 'segm_F1_opt', 'segm_best_thr',
            'bbox_mAP50', 'bbox_F1_opt', 'checkpoint']
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(summ, index=False)
    print(f'  summary -> {summ}  ({len(df)} rows)')

    # per-stage wide tables + LaTeX
    tex_blocks = []
    for stage, g in df.groupby('stage'):
        wide = (g.pivot_table(index=['model', 'family'], columns='val_set',
                              values='segm_mAP50')
                  .reset_index())
        wpath = osp.join(out, f'validation_by_stage_{stage}.csv')
        wide.to_csv(wpath, index=False)
        print(f'  stage {stage} -> {wpath}')

        # LaTeX: model | family | val_set | segm mAP@50 | segm F1
        gt = g[['model', 'family', 'val_set', 'segm_mAP50', 'segm_F1_opt']].copy()
        lines = [r'\begin{table}[t]\centering',
                 rf'\caption{{Stage {stage}: validation-set accuracy '
                 r'(mask mAP@0.50 and F1 at the test-swept optimal threshold).}',
                 rf'\label{{tab:val_stage_{stage}}}',
                 r'\begin{tabular}{lllrr}', r'\toprule',
                 r'Model & Family & Val.\ set & mAP@0.50 & F1 \\', r'\midrule']
        NL = r" \\"            # LaTeX row terminator (two backslashes)
        for _, r in gt.iterrows():
            vs = str(r['val_set']).replace('_', r'\_')
            m = r['segm_mAP50']; fscore = r['segm_F1_opt']
            lines.append(f"{r['model']} & {r['family']} & {vs} & "
                         f"{m:.3f} & {fscore:.3f}" + NL)
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
        tex_blocks.append('\n'.join(lines))

    tex_path = osp.join(out, 'validation_tables.tex')
    open(tex_path, 'w').write('\n'.join(tex_blocks))
    print(f'  latex   -> {tex_path}')

    # console preview
    print('\n=== validation summary (segm) ===')
    prev = df[['stage', 'model', 'family', 'val_set', 'segm_mAP50', 'segm_F1_opt']]
    with pd.option_context('display.max_rows', None, 'display.width', 140):
        print(prev.to_string(index=False))


if __name__ == '__main__':
    main()
