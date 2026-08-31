#!/usr/bin/env python3
# ==========================================================================
# run_cross_transfer.py
# --------------------------------------------------------------------------
# Cross-resolution and cross-sensor transfer evaluation driver (paper §4.6 / §5.2).
#
# Evaluates each SINGLE-SOURCE model from the backbone comparison (§5.1) —
# the UAV-trained models and the pooled 15 cm-trained models — ZERO-SHOT
# (no weights updated) across its row of the transfer matrix:
#
#   UAV-trained       -> UAV (5 cm, in-domain), UAV_15sim, UAV_30sim,
#                        GE (15 cm), Aerial (15 cm)
#   15 cm-trained     -> GE, Aerial (in-domain), UAV (5 cm), UAV_15sim, UAV_30sim
#
# DESIGN — reuse, do not reimplement
# ----------------------------------
# Metric computation is delegated verbatim to the validated stack:
#   * metrics_engine.patch_config   — dataloader/evaluator rewrite,
#   * metrics_engine.PalmBenchmarkMetric (registered on import),
#   * metrics_engine.write_csv      — identical CSV schema to Stages A/B/C,
#   * evaluate_model.register_modules / resolve_device — backbone registration
#                                     and device handling reused as-is.
# The only addition over evaluate_model.py is per-sensor pipeline injection:
# a sensor whose registry entry carries 'test_pipeline' (the resampled UAV
# sets) is evaluated with that no-resize pipeline; all other sensors keep the
# config's standard 1024 px pipeline. Consequently the in-domain and transfer
# numbers are produced by byte-identical metric code and are directly
# comparable; the transfer gap is a difference of like-for-like quantities.
#
# Efficiency is NOT measured here (it is architecture-only and stays on the
# single-reference-GPU protocol of §4.5). This driver is accuracy-only.
#
# The experiment is defined by a manifest JSON (see --write-example-manifest),
# so adding or removing a backbone is a manifest edit, not a code change.
# Per (model, sensor) the result is one CSV named <config_stem>__<label>.csv,
# identical to evaluate_model.py output; existing cells are skipped (resume).
#
# Usage
# -----
#   # 1. write a template, fill in configs/checkpoints, then run:
#   python run_cross_transfer.py --write-example-manifest cross_transfer.json
#   python run_cross_transfer.py --manifest cross_transfer.json
#
#   # restrict to one training source or one model while iterating:
#   python run_cross_transfer.py --manifest cross_transfer.json --train-source UAV
#   python run_cross_transfer.py --manifest cross_transfer.json --only ResNet-50
# ==========================================================================

import argparse
import glob
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Optional

# Make sibling evaluation modules importable regardless of CWD.
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from mmengine.config import Config                       # noqa: E402
from mmengine.runner import Runner                       # noqa: E402

# Importing metrics_engine registers PalmBenchmarkMetric in the METRICS registry.
from metrics_engine import (                             # noqa: E402,F401
    PalmBenchmarkMetric,
    patch_config,
    print_summary,
    write_csv,
)
from sensor_registry import available_sensors, get_sensor  # noqa: E402
from evaluate_model import register_modules, resolve_device  # noqa: E402

import copy                                                # noqa: E402

# Dataset wrappers whose payload is a single inner dataset (or a list); the
# test set is rebuilt from the leaf so we never inherit a wrapper's stale
# root or a ConcatDataset's mean-val structure.
_WRAP_SINGLE = {'RepeatDataset', 'ClassBalancedDataset',
                'MultiImageMixDataset', 'ConcatDataset'}


def unwrap_to_leaf(ds: dict) -> dict:
    """Descends dataset wrappers to the underlying single dataset dict."""
    ds = copy.deepcopy(ds)
    seen = 0
    while isinstance(ds, dict) and ds.get('type') in _WRAP_SINGLE and seen < 8:
        if isinstance(ds.get('dataset'), dict):
            ds = ds['dataset']
        elif isinstance(ds.get('datasets'), (list, tuple)) and ds['datasets']:
            ds = ds['datasets'][0]
        else:
            break
        seen += 1
    return ds


# --------------------------------------------------------------------------
# Default transfer matrix (training source -> ordered list of test sensors).
# In-domain cells are included first so the analyzer can compute gaps without
# depending on separately located §5.1 CSVs; they are cheap and resumable.
# --------------------------------------------------------------------------
DEFAULT_MATRIX: Dict[str, List[str]] = {
    'UAV':       ['UAV', 'UAV_15sim', 'UAV_30sim', 'GE', 'Aerial'],
    'GE_Aerial': ['GE', 'Aerial', 'UAV', 'UAV_15sim', 'UAV_30sim'],
}


# --------------------------------------------------------------------------
# Manifest helpers
# --------------------------------------------------------------------------
def write_example_manifest(path: str) -> None:
    """Writes a self-documenting manifest template."""
    example = {
        "_comment": "Cross-resolution/cross-sensor transfer experiment (§4.6/§5.2). "
                    "One entry per single-source model from §5.1. train_source is "
                    "'UAV' or 'GE_Aerial'. applied_score_thr is the F1-optimal "
                    "threshold selected on THAT model's source-domain validation "
                    "set (same value used for its in-domain F1 in §5.1); it affects "
                    "only the F1 column, not mAP. Omit it to sweep on the test set.",
        "results_dir": "results/cross_transfer",
        "protocol": {
            "score_thr": 0.05,
            "iou_thr": 0.50,
            "max_dets": 500,
            "batch_size_native": 2,
            "batch_size_noresize": 1,
            "num_workers": 2
        },
        "matrix": DEFAULT_MATRIX,
        "models": [
            {
                "label": "ResNet-50",
                "family": "CNN",
                "train_source": "UAV",
                "config": "configs/Custom/1_single_sensor_uav_5cm/maskrcnn_r50_uav5cm.py",
                "checkpoint": "work_dirs/Stage_A/maskrcnn_r50_uav5cm/best_coco_segm_mAP_50_iter_*.pth",
                "applied_score_thr": None
            },
            {
                "label": "Swin-S",
                "family": "Transformer",
                "train_source": "GE_Aerial",
                "config": "configs/Custom/2_pooled_15cm_ge_aerial/maskrcnn_swin_s_ms15.py",
                "checkpoint": "work_dirs/Stage_B/maskrcnn_swin_s_ms15/best_coco_segm_mAP_50_iter_*.pth",
                "applied_score_thr": None
            }
        ]
    }
    with open(path, 'w') as f:
        json.dump(example, f, indent=2)
    print(f'[manifest] example written -> {path}\n'
          f'           edit configs/checkpoints/labels, then run with --manifest.')


def load_manifest(path: str) -> dict:
    with open(path) as f:
        m = json.load(f)
    m.setdefault('results_dir', 'results/cross_transfer')
    m.setdefault('protocol', {})
    m.setdefault('matrix', DEFAULT_MATRIX)
    p = m['protocol']
    p.setdefault('score_thr', 0.05)
    p.setdefault('iou_thr', 0.50)
    p.setdefault('max_dets', 300)
    p.setdefault('batch_size_native', 2)
    p.setdefault('batch_size_noresize', 1)
    p.setdefault('num_workers', 2)
    if 'models' not in m or not m['models']:
        sys.exit(f'[ERROR] manifest {path} has no "models".')
    return m


def resolve_glob(pattern: str, label: str) -> str:
    """Expands a checkpoint glob to a single .pth; aborts on zero/ambiguous."""
    if any(ch in pattern for ch in '*?[]'):
        hits = sorted(glob.glob(pattern))
        if not hits:
            sys.exit(f'[ERROR] {label}: no checkpoint matches {pattern}')
        if len(hits) > 1:
            sys.exit(f'[ERROR] {label}: glob {pattern} matched {len(hits)} '
                     f'files; tighten it: {hits}')
        return hits[0]
    return pattern


# --------------------------------------------------------------------------
# One (model, sensor) cell
# --------------------------------------------------------------------------
def evaluate_cell(base_cfg: Config,
                  sensor: str,
                  checkpoint: str,
                  config_stem: str,
                  protocol: dict,
                  applied_score_thr: Optional[float],
                  device: str,
                  results_dir: str) -> Optional[Dict[str, float]]:
    """Evaluates one checkpoint on one sensor; writes one CSV. Returns the
    cleaned metric dict, or None if the test set is missing."""
    entry = get_sensor(sensor)
    ann_file = entry['ann_file']
    img_prefix = entry['img_prefix']
    label = entry['label']
    no_resize = 'test_pipeline' in entry

    if not osp.isfile(ann_file):
        print(f'[skip] {sensor}: annotation missing: {ann_file}')
        return None
    if not osp.isdir(img_prefix):
        print(f'[skip] {sensor}: image dir missing: {img_prefix}')
        return None

    batch_size = (protocol['batch_size_noresize'] if no_resize
                  else protocol['batch_size_native'])
    work_dir = osp.join('/tmp', f'xtransfer_{config_stem}__{label}')
    os.makedirs(work_dir, exist_ok=True)

    cfg = patch_config(
        base_cfg,
        ann_file=ann_file,
        img_prefix=img_prefix,
        batch_size=batch_size,
        num_workers=protocol['num_workers'],
        score_thr=protocol['score_thr'],
        iou_thr=protocol['iou_thr'],
        work_dir=work_dir,
        applied_score_thr=applied_score_thr,
        max_dets=protocol['max_dets'],
    )

    # --- robustly define the test set, independent of the config's shape --
    # 1) Build nothing but the test set: null the train/val machinery so a
    #    stale train/val dataloader baked into the config is never instantiated.
    for k in ('train_dataloader', 'train_cfg', 'optim_wrapper', 'param_scheduler',
              'val_dataloader', 'val_cfg', 'val_evaluator'):
        if k in cfg:
            cfg[k] = None
    # 2) Unwrap any dataset wrapper (RepeatDataset / ConcatDataset / ...) to its
    #    leaf, then rebuild the test dataset from the registry. This preserves
    #    the leaf's dataset type and metainfo (classes) while overriding the
    #    paths, so a wrapper's stale root or a mean-val concat cannot leak in.
    leaf = unwrap_to_leaf(cfg.test_dataloader['dataset'])
    std_pipeline = leaf.get('pipeline') or cfg.get('test_pipeline')
    leaf['data_root'] = ''
    leaf['ann_file'] = ann_file
    leaf['data_prefix'] = dict(img=img_prefix)
    leaf['test_mode'] = True
    leaf['pipeline'] = entry['test_pipeline'] if no_resize else std_pipeline
    if leaf.get('pipeline') is None:
        sys.exit(f'[ERROR] {config_stem} -> {sensor}: no test pipeline found in '
                 f'config; cannot evaluate.')
    cfg.test_dataloader['dataset'] = leaf

    cfg.load_from = checkpoint
    cfg.resume = False           # load weights only; never take the resume path
    cfg.launcher = 'none'
    cfg.device = device
    env_cfg = dict(cfg.get('env_cfg', {}))
    # cuDNN autotune helps only at a fixed input size; disable it for the
    # variable-size no-resize sensors.
    env_cfg['cudnn_benchmark'] = device.startswith('cuda') and not no_resize
    cfg.env_cfg = env_cfg

    print(f'\n{"=" * 76}\n  {config_stem}  ->  {sensor}  '
          f'({"no-resize" if no_resize else "standard"} pipeline, '
          f'batch {batch_size})\n  ckpt: {osp.basename(checkpoint)}\n{"=" * 76}')

    runner = Runner.from_cfg(cfg)
    try:
        runner.model.to(device)
    except Exception as exc:                              # noqa: BLE001
        raise RuntimeError(f'could not place model on {device}: {exc}') from exc

    metrics = runner.test()
    metrics_clean = {k.split('/', 1)[-1] if '/' in k else k: v
                     for k, v in metrics.items()}

    print_summary(metrics_clean, config_path=config_stem, ckpt_path=checkpoint,
                  ann_path=ann_file, score_thr=protocol['score_thr'],
                  iou_thr=protocol['iou_thr'])
    out_csv = osp.join(results_dir, f'{config_stem}__{label}.csv')
    write_csv(metrics_clean, out_path=out_csv, config_stem=config_stem,
              ann_stem=label, score_thr=protocol['score_thr'],
              iou_thr=protocol['iou_thr'])

    # Free VRAM between cells; SSM backbones are memory-heavy.
    import torch
    del runner
    torch.cuda.empty_cache()
    return metrics_clean


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description='Cross-resolution/cross-sensor transfer driver (§4.6/§5.2).',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', help='experiment manifest JSON')
    ap.add_argument('--write-example-manifest', metavar='PATH',
                    help='write a template manifest and exit')
    ap.add_argument('--train-source', choices=['UAV', 'GE_Aerial'],
                    help='restrict to models of this training source')
    ap.add_argument('--only', metavar='LABEL',
                    help='restrict to the single model with this label')
    ap.add_argument('--device', default='auto',
                    help='auto (default) / cuda / cuda:N / cpu')
    ap.add_argument('--overwrite', action='store_true',
                    help='re-evaluate cells whose CSV already exists')
    args = ap.parse_args()

    if args.write_example_manifest:
        write_example_manifest(args.write_example_manifest)
        return
    if not args.manifest:
        sys.exit('[ERROR] provide --manifest PATH (or --write-example-manifest).')

    m = load_manifest(args.manifest)
    results_dir = m['results_dir']
    protocol = m['protocol']
    matrix = m['matrix']
    os.makedirs(results_dir, exist_ok=True)

    # Validate that every sensor referenced by the matrix is registered.
    valid = set(available_sensors())
    for src, sensors in matrix.items():
        unknown = [s for s in sensors if s not in valid]
        if unknown:
            sys.exit(f'[ERROR] matrix["{src}"] references unregistered sensor(s) '
                     f'{unknown}. Registered: {sorted(valid)}. Add the missing '
                     f'entries to the SENSORS table in sensor_registry.py.')

    device = resolve_device(args.device)

    models = m['models']
    if args.train_source:
        models = [x for x in models if x.get('train_source') == args.train_source]
    if args.only:
        models = [x for x in models if x.get('label') == args.only]
    if not models:
        sys.exit('[ERROR] no models selected after filtering.')

    # Single-instance lock: refuse to start if another run is active. This
    # prevents concurrent launches from racing on the GPU and results dir.
    lock = osp.join(results_dir, '.run.lock')
    if osp.isfile(lock):
        with open(lock) as f:
            who = f.read().strip()
        sys.exit(f'[ERROR] a run lock exists ({lock}, {who}). Another driver may '
                 f'be active. If none is running, delete the lock and retry.')
    with open(lock, 'w') as f:
        f.write(f'pid={os.getpid()}')

    import torch
    index = {'results_dir': results_dir, 'protocol': protocol, 'cells': []}
    try:
        for model in models:
            label = model.get('label', '?')
            src = model['train_source']
            if src not in matrix:
                sys.exit(f'[ERROR] model "{label}" train_source "{src}" absent from matrix.')
            config = model['config']
            checkpoint_pat = model['checkpoint']
            config_stem = osp.splitext(osp.basename(config))[0]

            # A bad config/checkpoint for ONE model must not abort the whole run.
            if (not osp.isfile(config) or str(config).startswith('UNRESOLVED') or
                    str(checkpoint_pat).startswith('UNRESOLVED')):
                print(f'[ERROR] {label}: unresolved config/checkpoint; skipping model.')
                index['cells'].append({'label': label, 'config_stem': config_stem,
                                       'status': 'unresolved'})
                continue
            try:
                checkpoint = resolve_glob(checkpoint_pat, label)
                base_cfg = Config.fromfile(config)
                register_modules(base_cfg)
            except SystemExit:
                raise
            except Exception as exc:                      # noqa: BLE001
                print(f'[ERROR] {label}: cannot load config/checkpoint ({exc}); skipping model.')
                index['cells'].append({'label': label, 'config_stem': config_stem,
                                       'status': 'load_error', 'error': str(exc)[:300]})
                continue
            applied = model.get('applied_score_thr', None)

            for sensor in matrix[src]:
                out_csv = osp.join(results_dir,
                                   f'{config_stem}__{get_sensor(sensor)["label"]}.csv')
                cell = {'label': label, 'family': model.get('family', ''),
                        'train_source': src, 'config_stem': config_stem,
                        'sensor': sensor, 'csv': out_csv}
                if osp.isfile(out_csv) and not args.overwrite:
                    print(f'[resume] {config_stem} -> {sensor}: exists, skipping.')
                    cell['status'] = 'resumed'
                    index['cells'].append(cell)
                    continue
                try:
                    res = evaluate_cell(base_cfg, sensor, checkpoint, config_stem,
                                        protocol, applied, device, results_dir)
                    cell['status'] = 'done' if res is not None else 'missing_testset'
                except Exception as exc:                  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    cell['status'] = 'error'
                    cell['error'] = str(exc)[:300]
                    print(f'[ERROR] {config_stem} -> {sensor}: {exc} (continuing)')
                    try:
                        torch.cuda.empty_cache()
                    except Exception:                     # noqa: BLE001
                        pass
                index['cells'].append(cell)
                with open(osp.join(results_dir, 'cross_transfer_index.json'), 'w') as f:
                    json.dump(index, f, indent=2)
    finally:
        if osp.isfile(lock):
            os.remove(lock)

    done = sum(1 for c in index['cells'] if c.get('status') == 'done')
    errs = [c for c in index['cells'] if c.get('status') in ('error', 'load_error', 'unresolved')]
    print(f'\n[done] {done} cells evaluated, {len(errs)} problem cell(s). '
          f'index -> {osp.join(results_dir, "cross_transfer_index.json")}')
    for c in errs:
        print(f'    {c.get("config_stem","?")} -> {c.get("sensor","-")}: '
              f'{c.get("status")} {c.get("error","")}')
    print(f'[next] python compile_cross_transfer.py --manifest {args.manifest}')


if __name__ == '__main__':
    main()
