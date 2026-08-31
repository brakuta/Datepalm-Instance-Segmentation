#!/usr/bin/env python3
# ==========================================================================
# compile_cross_transfer.py
# --------------------------------------------------------------------------
# Analyzer for the cross-resolution / cross-sensor transfer experiment
# (paper §5.2). Consumes the per-cell CSVs written by run_cross_transfer.py
# (and, identically, by evaluate_model.py for the in-domain §5.1 cells), all
# in the metrics_engine.write_csv long format:
#
#   columns: model, metric, bbox, segm, score_thr, iou_thr, test_set
#
# Produces, under <results_dir>/compiled/:
#   matrix_segm_mAP_50.csv      models x sensors (PRIMARY accuracy)
#   matrix_segm_f1_opt.csv      models x sensors (operating-point F1)
#   matrix_bbox_mAP_50.csv      models x sensors (supplementary)
#   transfer_gaps.csv           per-model gaps + matched-GSD decomposition
#   map_gap.csv                 (label, map_gap) for the feature-analysis suite
#   fig_degradation_uav.{pdf,png}      segm mAP@50 vs GSD, UAV-trained models
#   fig_decomposition_15cm.{pdf,png}   GSD vs sensor-residual at matched 15 cm
#
# GAP DEFINITIONS (made explicit for the manuscript)
# --------------------------------------------------
# For a UAV-trained model (in-domain = UAV 5 cm):
#   cross-resolution gap   = mAP(UAV) - mAP(UAV_<g>sim)       [pure GSD]
#   cross-sensor gap       = mAP(UAV) - mAP(<native 15 cm>)
#   matched-GSD (15 cm) decomposition of the native-sensor gap:
#       GSD component      = mAP(UAV) - mAP(UAV_15sim)
#       sensor residual    = mAP(UAV_15sim) - mAP(<native 15 cm>)
#   map_gap (for §4.9)     = mAP(UAV) - mean(mAP(UAV_15sim), mAP(UAV_30sim))
#
# For a 15 cm-trained model (in-domain = mean of GE, Aerial):
#   the only matched-GSD cross-sensor cell is UAV_15sim, so
#       sensor shift (15 cm) = mean(GE,Aerial) - mAP(UAV_15sim)
#   gaps to UAV 5/30 cm mix sensor and resolution and are reported as such.
# A controlled within-sensor resolution sweep does not exist for 15 cm-trained
# models in this matrix (only UAV was resampled); the §4.9 correlation against
# representational resolution-invariance is therefore cleanest when restricted
# to UAV-trained models, whose cross-resolution gap is confound-free.
# ==========================================================================

import argparse
import csv
import glob
import json
import math
import os
import os.path as osp
import sys
from typing import Dict, Optional

# Per-sensor nominal GSD (m). Used only for the degradation-versus-GSD figure.
GSD_M = {'UAV': 0.05, 'UAV_15sim': 0.15, 'UAV_30sim': 0.30,
         'GE': 0.15, 'Aerial': 0.15, 'Sat': 0.30}

FAMILY_COLOR = {'CNN': '#0072B2', 'Transformer': '#D55E00', 'Mamba': '#009E73',
                'SSM': '#009E73', 'SSM/Mamba': '#009E73'}

# write_csv metric-row labels -> short keys we extract.
METRIC_ROWS = {
    'mAP@50 [PRIMARY]': 'mAP_50',
    'F1@0.5 (optimal threshold) [PRIMARY]': 'f1_opt',
    'Best score threshold': 'best_thr',
    'mAP@[.5:.95] [suppl.]': 'mAP',
}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def read_cell_csv(path: str) -> Optional[dict]:
    """Parses one write_csv long-format CSV into
    {'model':..., 'test_set':..., 'segm':{key:val}, 'bbox':{key:val}}."""
    segm, bbox = {}, {}
    model = test_set = None
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                model = row.get('model', model)
                test_set = row.get('test_set', test_set)
                key = METRIC_ROWS.get(row.get('metric', ''))
                if key is None:
                    continue
                for fld, dst in (('segm', segm), ('bbox', bbox)):
                    raw = (row.get(fld) or '').strip()
                    if raw != '':
                        try:
                            dst[key] = float(raw)
                        except ValueError:
                            pass
    except Exception as exc:                              # noqa: BLE001
        print(f'[warn] could not parse {path}: {exc}')
        return None
    if model is None or test_set is None:
        return None
    return {'model': model, 'test_set': test_set, 'segm': segm, 'bbox': bbox}


def label_to_sensor(test_set_label: str, registry_map: Dict[str, str]) -> Optional[str]:
    """Maps a CSV test_set tag (e.g. 'test_UAV_15cm') to a sensor key."""
    return registry_map.get(test_set_label)


def build_registry_map() -> Dict[str, str]:
    """label (e.g. 'test_UAV_15cm') -> sensor key (e.g. 'UAV_15sim')."""
    try:
        sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
        from sensor_registry import SENSORS
        return {v['label']: k for k, v in SENSORS.items()}
    except Exception:
        # Fallback to the known labels if the registry is not importable.
        return {'test_UAV': 'UAV', 'test_UAV_15cm': 'UAV_15sim',
                'test_UAV_30cm': 'UAV_30sim', 'test_GE': 'GE',
                'test_aerial': 'Aerial', 'test_sat': 'Sat'}


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def _g(d, model, sensor, metric):
    """Safe getter into the nested {model:{sensor:{metric:val}}} store."""
    return d.get(model, {}).get(sensor, {}).get(metric)


def assemble(results_dir: str, manifest: dict):
    """Returns (store, sensors_seen). store[config_stem][sensor]['segm'/'bbox'] = {key:val}."""
    reg = build_registry_map()
    store: Dict[str, Dict[str, dict]] = {}
    sensors_seen = set()
    for path in sorted(glob.glob(osp.join(results_dir, '*__*.csv'))):
        if osp.basename(path).startswith(('matrix_', 'transfer_', 'map_gap')):
            continue
        cell = read_cell_csv(path)
        if not cell:
            continue
        sensor = label_to_sensor(cell['test_set'], reg)
        if sensor is None:
            print(f'[warn] {osp.basename(path)}: test_set "{cell["test_set"]}" '
                  f'not in registry; skipping.')
            continue
        store.setdefault(cell['model'], {})[sensor] = {
            'segm': cell['segm'], 'bbox': cell['bbox']}
        sensors_seen.add(sensor)
    return store, sensors_seen


def model_meta(manifest: dict) -> Dict[str, dict]:
    """config_stem -> {label, family, train_source}."""
    meta = {}
    for mdl in manifest.get('models', []):
        stem = osp.splitext(osp.basename(mdl['config']))[0]
        meta[stem] = {'label': mdl.get('label', stem),
                      'family': mdl.get('family', ''),
                      'train_source': mdl.get('train_source', '')}
    return meta


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------
def write_matrix(store, meta, sensors_order, iou_field, metric_key, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'family', 'train_source'] + sensors_order)
        for stem in sorted(store, key=lambda s: (meta.get(s, {}).get('train_source', ''),
                                                  meta.get(s, {}).get('family', ''),
                                                  meta.get(s, {}).get('label', s))):
            mm = meta.get(stem, {'label': stem, 'family': '', 'train_source': ''})
            row = [mm['label'], mm['family'], mm['train_source']]
            for s in sensors_order:
                v = store.get(stem, {}).get(s, {}).get(iou_field, {}).get(metric_key)
                row.append(f'{v:.4f}' if isinstance(v, float) else '')
            w.writerow(row)
    print(f'  matrix -> {out_path}')


def _sub(a, b):
    return (a - b) if (isinstance(a, float) and isinstance(b, float)) else None


def _mean(vals):
    xs = [v for v in vals if isinstance(v, float)]
    return sum(xs) / len(xs) if xs else None


def write_gaps(store, meta, out_path):
    cols = ['label', 'family', 'train_source',
            'mAP_UAV', 'mAP_UAV_15sim', 'mAP_UAV_30sim', 'mAP_GE', 'mAP_Aerial',
            'gap_res_15', 'gap_res_30', 'gap_sensor_GE', 'gap_sensor_Aerial',
            'gsd_component_15', 'sensor_residual_GE', 'sensor_residual_Aerial',
            'map_gap_crossres', 'sensor_shift_15_for_15cm_models']
    rows = []
    for stem, byS in store.items():
        mm = meta.get(stem, {'label': stem, 'family': '', 'train_source': ''})
        def m(sensor):
            return byS.get(sensor, {}).get('segm', {}).get('mAP_50')
        uav, u15, u30 = m('UAV'), m('UAV_15sim'), m('UAV_30sim')
        ge, aer = m('GE'), m('Aerial')
        r = {c: '' for c in cols}
        r.update({'label': mm['label'], 'family': mm['family'],
                  'train_source': mm['train_source'],
                  'mAP_UAV': uav, 'mAP_UAV_15sim': u15, 'mAP_UAV_30sim': u30,
                  'mAP_GE': ge, 'mAP_Aerial': aer})
        if mm['train_source'] == 'UAV':
            r['gap_res_15'] = _sub(uav, u15)
            r['gap_res_30'] = _sub(uav, u30)
            r['gap_sensor_GE'] = _sub(uav, ge)
            r['gap_sensor_Aerial'] = _sub(uav, aer)
            r['gsd_component_15'] = _sub(uav, u15)
            r['sensor_residual_GE'] = _sub(u15, ge)
            r['sensor_residual_Aerial'] = _sub(u15, aer)
            r['map_gap_crossres'] = _sub(uav, _mean([u15, u30]))
        elif mm['train_source'] == 'GE_Aerial':
            indomain = _mean([ge, aer])
            r['sensor_shift_15_for_15cm_models'] = _sub(indomain, u15)
        rows.append(r)

    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([f'{r[c]:.4f}' if isinstance(r[c], float) else
                        ('' if r[c] is None else r[c]) for c in cols])
    print(f'  gaps   -> {out_path}')
    return rows


def write_map_gap(rows, out_path):
    """(label, map_gap) for the feature-analysis suite. Uses the confound-free
    controlled cross-resolution gap; only UAV-trained models qualify."""
    n = 0
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'map_gap'])
        for r in rows:
            if r['train_source'] == 'UAV' and isinstance(r['map_gap_crossres'], float):
                w.writerow([r['label'], f'{r["map_gap_crossres"]:.4f}'])
                n += 1
    print(f'  map_gap-> {out_path}  ({n} UAV-trained models)')


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(store, meta, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:                              # noqa: BLE001
        print(f'[warn] matplotlib unavailable ({exc}); figures skipped.')
        return
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 9, 'axes.linewidth': 0.8,
        'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'legend.frameon': False,
    })

    # --- Fig 1: degradation vs GSD for UAV-trained models -----------------
    uav_stems = [s for s in store if meta.get(s, {}).get('train_source') == 'UAV']
    if uav_stems:
        order = ['UAV', 'UAV_15sim', 'UAV_30sim']
        xs = [GSD_M[s] * 100 for s in order]
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        for stem in sorted(uav_stems, key=lambda s: meta[s]['label']):
            ys = [store[stem].get(s, {}).get('segm', {}).get('mAP_50') for s in order]
            if not any(isinstance(y, float) for y in ys):
                continue
            fam = meta[stem].get('family', '')
            ax.plot([x for x, y in zip(xs, ys) if isinstance(y, float)],
                    [y for y in ys if isinstance(y, float)],
                    marker='o', ms=3, lw=1.0,
                    color=FAMILY_COLOR.get(fam, '#444444'),
                    label=meta[stem]['label'])
        ax.set_xlabel('Ground sampling distance (cm)')
        ax.set_ylabel(r'Mask AP$_{50}$')
        ax.set_xticks([5, 15, 30])
        ax.set_title('Cross-resolution degradation (UAV-trained)')
        ax.legend(fontsize=6, ncol=2, loc='upper right')
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(osp.join(out_dir, f'fig_degradation_uav.{ext}'))
        plt.close(fig)
        print(f'  figure -> {osp.join(out_dir, "fig_degradation_uav.{pdf,png}")}')

    # --- Fig 2: matched-GSD decomposition at 15 cm ------------------------
    rows = []
    for stem in uav_stems:
        seg = store[stem]
        uav = seg.get('UAV', {}).get('segm', {}).get('mAP_50')
        u15 = seg.get('UAV_15sim', {}).get('segm', {}).get('mAP_50')
        nat = _mean([seg.get('GE', {}).get('segm', {}).get('mAP_50'),
                     seg.get('Aerial', {}).get('segm', {}).get('mAP_50')])
        if all(isinstance(v, float) for v in (uav, u15, nat)):
            rows.append((meta[stem]['label'], meta[stem].get('family', ''),
                         uav - u15, max(u15 - nat, 0.0)))
    if rows:
        labels = [r[0] for r in rows]
        gsd = [r[2] for r in rows]
        sen = [r[3] for r in rows]
        import numpy as np
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(3.4, 0.5 * len(labels) + 1.5), 2.7))
        ax.bar(x, gsd, 0.6, label='GSD component (UAV 5\u219215 cm)', color='#56B4E9')
        ax.bar(x, sen, 0.6, bottom=gsd, label='Sensor residual (15 cm, native)',
               color='#E69F00')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        ax.set_ylabel(r'$\Delta$ Mask AP$_{50}$')
        ax.set_title('Decomposition of the 15 cm transfer gap')
        ax.legend(fontsize=6, loc='upper left')
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(osp.join(out_dir, f'fig_decomposition_15cm.{ext}'))
        plt.close(fig)
        print(f'  figure -> {osp.join(out_dir, "fig_decomposition_15cm.{pdf,png}")}')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description='Compile the cross-resolution/cross-sensor transfer results (§5.2).')
    ap.add_argument('--manifest', required=True,
                    help='the same manifest passed to run_cross_transfer.py')
    ap.add_argument('--results-dir', default=None,
                    help='override manifest results_dir')
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    results_dir = args.results_dir or manifest.get('results_dir', 'results/cross_transfer')
    if not osp.isdir(results_dir):
        sys.exit(f'[ERROR] results dir not found: {results_dir}')

    store, sensors_seen = assemble(results_dir, manifest)
    if not store:
        sys.exit(f'[ERROR] no parseable per-cell CSVs in {results_dir}.')
    meta = model_meta(manifest)

    out_dir = osp.join(results_dir, 'compiled')
    os.makedirs(out_dir, exist_ok=True)

    sensors_order = [s for s in ['UAV', 'UAV_15sim', 'UAV_30sim', 'GE', 'GE_30sim', 'Aerial', 'Sat']
                     if s in sensors_seen]

    print(f'\n[compile] {len(store)} models, sensors {sensors_order}')
    write_matrix(store, meta, sensors_order, 'segm', 'mAP_50',
                 osp.join(out_dir, 'matrix_segm_mAP_50.csv'))
    write_matrix(store, meta, sensors_order, 'segm', 'f1_opt',
                 osp.join(out_dir, 'matrix_segm_f1_opt.csv'))
    write_matrix(store, meta, sensors_order, 'bbox', 'mAP_50',
                 osp.join(out_dir, 'matrix_bbox_mAP_50.csv'))
    gap_rows = write_gaps(store, meta, osp.join(out_dir, 'transfer_gaps.csv'))
    write_map_gap(gap_rows, osp.join(out_dir, 'map_gap.csv'))
    if not args.no_figures:
        make_figures(store, meta, out_dir)

    print(f'\n[done] compiled outputs in {out_dir}')


if __name__ == '__main__':
    main()
