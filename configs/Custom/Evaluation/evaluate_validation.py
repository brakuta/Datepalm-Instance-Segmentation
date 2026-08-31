#!/usr/bin/env python3
# ==========================================================================
# evaluate_validation.py
# --------------------------------------------------------------------------
# Evaluate trained Stage A/B/C checkpoints on their VALIDATION sets and report
# segm/bbox mAP@0.50 and F1 (test-swept optimal threshold) under the SAME
# PalmBenchmarkMetric and protocol used for the held-out test evaluation, so
# validation and test numbers are directly comparable.
#
# Why validation? Model selection (save_best='coco/segm_mAP_50') was performed
# on the validation split; reviewers frequently ask for that validation number.
# This script reproduces it under the unified evaluation protocol rather than
# reading it off the (LoggerHook-formatted) training log.
#
# Design (mirrors run_cross_transfer.py):
#   * The DUMPED config inside each run dir is authoritative: it produced the
#     checkpoint and cannot have drifted from the weights (this is what avoids
#     the d_state-style config/checkpoint mismatch).
#   * The validation image set + pipeline are taken from the config's own
#     val_dataloader; each leaf validation dataset is scored separately
#     (so a pooled/ConcatDataset val yields one row per native sensor, which
#     is both more informative and free of merged-annotation id-alignment risk).
#   * The standard test pipeline (1024 Resize+Pad) and protocol
#     (max_dets=500, score_thr=0.05, iou_thr=0.50) are forced for comparability.
#   * Per-(model,val_set) try/except, JSON index, and resume (skip existing CSV).
#
# Usage
# -----
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   python configs/Custom/Evaluation/evaluate_validation.py \
#       --stage A=work_dirs/Stage_A B=work_dirs/Stage_B C=work_dirs/Stage_C \
#       --results-dir results/validation
#   # resume is automatic; re-run after a crash to fill the rest.
#   # single model:  --only "SpatialMamba-S"
# ==========================================================================

import argparse
import copy
import glob
import json
import os
import os.path as osp
import re
import sys
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# make sibling modules (metrics_engine, evaluate_model) importable when this
# script is launched as `python <dir>/evaluate_validation.py`
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402

# --- register custom modules. These imports trigger @register_module(). ----
try:
    import evaluate_model as _evm            # noqa: F401  (backbones + helpers)
except Exception:
    _evm = None
try:
    from metrics_engine import PalmBenchmarkMetric  # noqa: F401 (registers metric)
except Exception:
    PalmBenchmarkMetric = None
try:
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
except Exception:
    pass

# --------------------------------------------------------------------------
# backbone label / family inference (specific -> general; first match wins)
# --------------------------------------------------------------------------
CANON = [
    (r'(r101|resnet[_-]?101)', 'ResNet-101', 'CNN'),
    (r'(r50|resnet[_-]?50)',   'ResNet-50',  'CNN'),
    (r'convnext.*_s', 'ConvNeXt-S', 'CNN'), (r'convnext', 'ConvNeXt-T', 'CNN'),
    (r'swin.*_s', 'Swin-S', 'Transformer'), (r'swin.*_t', 'Swin-T', 'Transformer'),
    (r'(pvt.*b2|pvtv2[_-]?b2)', 'PVTv2-B2', 'Transformer'),
    (r'uniformer', 'UniFormer-XS', 'Transformer'),
    (r'spatialmamba.*_s', 'SpatialMamba-S', 'Mamba'), (r'spatialmamba.*_t', 'SpatialMamba-T', 'Mamba'),
    (r'groupmamba.*_s', 'GroupMamba-S', 'Mamba'), (r'groupmamba.*_t', 'GroupMamba-T', 'Mamba'),
    (r'efficientvmamba.*_b', 'EfficientVMamba-B', 'Mamba'), (r'efficientvmamba.*_s', 'EfficientVMamba-S', 'Mamba'),
    (r'mambaout.*_s', 'MambaOut-S', 'Mamba'), (r'mambaout.*_t', 'MambaOut-T', 'Mamba'),
    (r'mambavision.*_s', 'MambaVision-S', 'Mamba'), (r'mambavision.*_t', 'MambaVision-T', 'Mamba'),
    (r'vmamba.*_s', 'VMamba-S', 'Mamba'), (r'vmamba.*_t', 'VMamba-T', 'Mamba'),
]


def infer_label_family(stem):
    s = stem.lower()
    for pat, lab, fam in CANON:
        if re.search(pat, s):
            return lab, fam
    return stem, ''


def resolve_config(stem, run_dir):
    """Authoritative config = the dump inside the run dir (matches checkpoint)."""
    cand = osp.join(run_dir, stem + '.py')
    if osp.isfile(cand):
        return cand
    pys = sorted(glob.glob(osp.join(run_dir, '*.py')))
    stem_match = [p for p in pys if osp.splitext(osp.basename(p))[0] == stem]
    if stem_match:
        return stem_match[0]
    return pys[0] if len(pys) == 1 else None


def resolve_checkpoint(run_dir, ckpt_glob):
    """Best checkpoint; tolerant of the non-standard save_best names."""
    def _it(p):
        m = re.search(r'iter_(\d+)', osp.basename(p))
        return int(m.group(1)) if m else -1
    for pat in (ckpt_glob, 'best_*.pth'):
        hits = glob.glob(osp.join(run_dir, pat))
        if hits:
            return max(hits, key=_it)
    lc = osp.join(run_dir, 'last_checkpoint')
    if osp.isfile(lc):
        p = open(lc).read().strip()
        return p if osp.isfile(p) else None
    return None


def _abs(path, root):
    if not path:
        return None
    return path if osp.isabs(path) else osp.normpath(osp.join(root or '', path))


def remap_path(p, remaps):
    """Apply OLD=NEW substring remaps; then try common fixes if still missing."""
    if p is None:
        return p
    for old, new in remaps:
        if old in p:
            p = p.replace(old, new)
    if not osp.isfile(p):
        # automatic fallback: stale '/workspace/project/' -> '/workspace/'
        for cand in (p.replace('/workspace/project/', '/workspace/'),
                     p.replace('/project/', '/')):
            if osp.isfile(cand):
                return cand
    return p


def leaf_val_datasets(ds):
    """Yield leaf CocoDataset specs from a (possibly nested) val dataset cfg."""
    if ds is None:
        return
    t = ds.get('type', '')
    if t == 'ConcatDataset' and 'datasets' in ds:
        for d in ds['datasets']:
            yield from leaf_val_datasets(d)
    elif 'dataset' in ds:                      # RepeatDataset / wrappers
        yield from leaf_val_datasets(ds['dataset'])
    else:
        yield ds


def get_metric(metrics, key):
    """Fetch a metric value tolerant of the 'palm/' prefix and naming."""
    for k in (key, 'palm/' + key, key.replace('_50', '_50'), 'coco/' + key):
        if k in metrics:
            return float(metrics[k])
    # last resort: substring match
    for mk, mv in metrics.items():
        if mk.endswith(key):
            return float(mv)
    return float('nan')


def build_test_cfg(base_cfg, ann, img, ckpt, batch, num_workers,
                   max_dets, score_thr, iou_thr, work_dir):
    """Mutate a loaded training cfg into a test-only cfg on one validation set."""
    c = copy.deepcopy(base_cfg)
    pipeline = c.test_pipeline
    metainfo = dict(classes=('DatePalm',))
    c.test_dataloader = dict(
        batch_size=batch, num_workers=num_workers,
        persistent_workers=False, drop_last=False,
        pin_memory=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='CocoDataset', ann_file=ann, data_root='',
                     data_prefix=dict(img=img), metainfo=metainfo,
                     test_mode=True, pipeline=pipeline),
    )
    c.test_evaluator = dict(
        type='PalmBenchmarkMetric', ann_file=ann,
        metric_items=('bbox', 'segm'),
        iou_thr=iou_thr, max_dets=max_dets, score_thr=score_thr,
    )
    c.test_cfg = dict(type='TestLoop')
    # strip all training/validation scaffolding
    c.train_dataloader = c.train_cfg = c.optim_wrapper = c.param_scheduler = None
    c.val_dataloader = c.val_cfg = c.val_evaluator = None
    c.custom_hooks = []
    c.load_from = ckpt
    c.resume = False
    c.work_dir = work_dir
    c.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='DetVisualizationHook', draw=False),
    )
    return c


def main():
    ap = argparse.ArgumentParser(description='Validation-set evaluation for Stages A/B/C.')
    ap.add_argument('--stage', nargs='+', default=['A=work_dirs/Stage_A',
                                                   'B=work_dirs/Stage_B',
                                                   'C=work_dirs/Stage_C'],
                    help='STAGE=run_dir pairs, e.g. A=work_dirs/Stage_A')
    ap.add_argument('--ckpt-glob', default='best_coco_segm_mAP_50_*.pth')
    ap.add_argument('--results-dir', default='results/validation')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--max-dets', type=int, default=500)
    ap.add_argument('--score-thr', type=float, default=0.05)
    ap.add_argument('--iou-thr', type=float, default=0.50)
    ap.add_argument('--only', default=None, help='evaluate only this backbone label')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--device', default=None, help='cuda:0 | cpu (default: auto)')
    ap.add_argument('--remap', nargs='*', default=None,
                    help='path remaps OLD=NEW applied to val ann/img paths, '
                         'e.g. /workspace/project/=/workspace/  (repeatable)')
    args = ap.parse_args()
    remaps = []
    for r in (args.remap or []):
        if '=' in r:
            o, n = r.split('=', 1); remaps.append((o, n))

    out = args.results_dir
    os.makedirs(out, exist_ok=True)
    index_path = osp.join(out, 'validation_index.json')
    index = json.load(open(index_path)) if osp.isfile(index_path) else {'cells': []}

    if args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.device == 'cpu' else \
            os.environ.get('CUDA_VISIBLE_DEVICES', '0')

    # discover runs
    runs = []
    for spec in args.stage:
        if '=' not in spec:
            print(f'[warn] bad --stage spec (need STAGE=dir): {spec}'); continue
        stage, sdir = spec.split('=', 1)
        if not osp.isdir(sdir):
            print(f'[warn] stage {stage} dir not found: {sdir}'); continue
        for run_dir in sorted(glob.glob(osp.join(sdir, '*'))):
            if osp.isdir(run_dir):
                runs.append((stage, run_dir))
    if not runs:
        sys.exit('[ERROR] no run dirs discovered. Check --stage paths.')

    done = {(c['stage'], c['model'], c['val_set']): c for c in index['cells']
            if c.get('status') == 'done'}
    n_ok = n_skip = n_err = 0

    for stage, run_dir in runs:
        stem = osp.basename(run_dir)
        label, family = infer_label_family(stem)
        if args.only and label != args.only:
            continue
        cfg_path = resolve_config(stem, run_dir)
        ckpt = resolve_checkpoint(run_dir, args.ckpt_glob)
        if cfg_path is None or ckpt is None:
            print(f'[skip] {stage}:{label} unresolved '
                  f'{"config" if cfg_path is None else ""} '
                  f'{"checkpoint" if ckpt is None else ""}')
            continue
        try:
            cfg = Config.fromfile(cfg_path)
        except Exception as e:
            print(f'[error] {stage}:{label} cfg load: {e}')
            continue

        # validation leaves from the config's own val_dataloader
        vdl = cfg.get('val_dataloader')
        if not vdl:
            print(f'[skip] {stage}:{label} has no val_dataloader')
            continue
        leaves = list(leaf_val_datasets(vdl['dataset']))
        if not leaves:
            print(f'[skip] {stage}:{label} no leaf val dataset')
            continue

        for leaf in leaves:
            root = leaf.get('data_root', '')
            ann = remap_path(_abs(leaf.get('ann_file'), root), remaps)
            imgp = leaf.get('data_prefix', {}).get('img')
            img = remap_path(_abs(imgp, root), remaps)
            if not ann or not osp.isfile(ann):
                print(f'[skip] {stage}:{label} val ann missing: {ann}')
                continue
            val_set = osp.splitext(osp.basename(ann))[0]      # e.g. val_UAV
            key = (stage, label, val_set)
            csv_path = osp.join(out, f'{stem}__{val_set}.csv')
            if not args.overwrite and osp.isfile(csv_path) and key in done:
                n_skip += 1
                print(f'[skip] {stage}:{label} on {val_set} (exists)')
                continue

            print('\n' + '=' * 76)
            print(f'  STAGE {stage}  |  {label}  ->  {val_set}')
            print(f'  ckpt: {osp.basename(ckpt)}')
            print('=' * 76)
            try:
                c = build_test_cfg(cfg, ann, img, ckpt, args.batch,
                                   args.num_workers, args.max_dets,
                                   args.score_thr, args.iou_thr,
                                   work_dir=f'/tmp/valeval_{stem}__{val_set}')
                runner = Runner.from_cfg(c)
                metrics = runner.test()

                row = dict(
                    stage=stage, model=label, family=family, val_set=val_set,
                    checkpoint=osp.basename(ckpt), ann_file=ann,
                    segm_mAP50=get_metric(metrics, 'segm_mAP_50'),
                    segm_mAP=get_metric(metrics, 'segm_mAP'),
                    segm_F1_opt=get_metric(metrics, 'segm_f1_opt'),
                    segm_P_opt=get_metric(metrics, 'segm_precision_opt'),
                    segm_R_opt=get_metric(metrics, 'segm_recall_opt'),
                    segm_best_thr=get_metric(metrics, 'segm_best_score_thr'),
                    bbox_mAP50=get_metric(metrics, 'bbox_mAP_50'),
                    bbox_mAP=get_metric(metrics, 'bbox_mAP'),
                    bbox_F1_opt=get_metric(metrics, 'bbox_f1_opt'),
                    bbox_best_thr=get_metric(metrics, 'bbox_best_score_thr'),
                )
                # write per-cell CSV
                import csv as _csv
                with open(csv_path, 'w', newline='') as f:
                    w = _csv.DictWriter(f, fieldnames=list(row.keys()))
                    w.writeheader(); w.writerow(row)

                index['cells'] = [x for x in index['cells']
                                  if (x['stage'], x['model'], x['val_set']) != key]
                index['cells'].append(dict(stage=stage, model=label,
                                           val_set=val_set, status='done',
                                           csv=osp.basename(csv_path)))
                json.dump(index, open(index_path, 'w'), indent=2)
                n_ok += 1
                print(f'  segm mAP@50={row["segm_mAP50"]:.4f}  '
                      f'segm F1={row["segm_F1_opt"]:.4f}  -> {csv_path}')
            except Exception as e:
                n_err += 1
                print(f'[ERROR] {stage}:{label} on {val_set}: {e}')
                traceback.print_exc()
                index['cells'] = [x for x in index['cells']
                                  if (x['stage'], x['model'], x['val_set']) != key]
                index['cells'].append(dict(stage=stage, model=label,
                                           val_set=val_set, status='error',
                                           error=str(e)))
                json.dump(index, open(index_path, 'w'), indent=2)
            finally:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f'\n[done] ok={n_ok}  skipped={n_skip}  errors={n_err}')
    print(f'[next] python {osp.basename(__file__).replace("evaluate","compile")} '
          f'--results-dir {out}')


if __name__ == '__main__':
    main()
