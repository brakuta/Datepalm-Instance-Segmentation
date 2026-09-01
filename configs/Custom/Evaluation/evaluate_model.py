# ==========================================================================
# evaluate_model.py
# ==========================================================================
# Per-model evaluation runner for the date palm detection benchmark.
#
# Evaluates ONE trained Mask R-CNN model on ONE OR MORE test sets in a
# single launch, using PalmBenchmarkMetric from metrics_engine.py.
# Produces one CSV per sensor.
#
# DESIGN
# ------
# This script holds NO dataset paths. Every test set is defined once in
# sensor_registry.py; --sensors accepts any key declared there (UAV, GE,
# Aerial, Sat, or any future source), so covering a new sensor needs no
# code change. Evaluation runs on the GPU (see --device) and applies the
# locked protocol defaults score_thr=0.05, iou_thr=0.50.
#
# --------------------------------------------------------------------------
# LOCATION
#   configs/Custom/Evaluation/   (alongside metrics_engine.py and
#                                 sensor_registry.py)
#
# Usage
# -----
#   python configs/Custom/Evaluation/evaluate_model.py \
#       --config     configs/Custom/2_pooled_15cm_ge_aerial/maskrcnn_r50_ms15.py \
#       --checkpoint work_dirs/Stage_B/maskrcnn_r50_ms15/best_coco_segm_mAP_50_iter_25000.pth \
#       --sensors    GE Aerial \
#       --results-dir results/stage_b
#
# Single-sensor or other-stage examples:
#   --sensors GE
#   --sensors UAV
#   --sensors GE Aerial UAV
# ==========================================================================

import argparse
import copy
import os
import os.path as osp
import sys
from typing import Dict

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# Make this folder importable so the sibling modules resolve regardless
# of the working directory the script is launched from.
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from metrics_engine import (              # noqa: E402
    PalmBenchmarkMetric,
    patch_config,
    print_summary,
    write_csv,
)
from sensor_registry import (                # noqa: E402
    available_sensors,
    get_sensor,
)
from efficiency import (                      # noqa: E402
    profile_model,
    EFF_WARMUP,
    EFF_ITERS,
)
import csv as _csv                            # noqa: E402


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------
def register_modules(cfg: Config) -> None:
    """Registers every backbone/module class needed to build the model.

    The model is built along TWO paths in this script: through
    ``Runner.from_cfg`` (the accuracy pass, which honours
    ``cfg.custom_imports``) and through a bare ``MODELS.build`` (the
    efficiency profile, which does NOT). Training relied on
    ``tools/train.py`` importing the backbone packages ambiently; the
    evaluation entry point does not, so unscoped ``type=`` names such as
    ``'ConvNeXt'`` are absent from the registry and the build raises a
    KeyError. This registers, idempotently and import-failure tolerant:

      * mmdet's own modules (ResNet, FPN, Mask R-CNN, SwinTransformer);
      * mmpretrain's modules — ConvNeXt and other classification backbones
        referenced by unscoped ``type=`` names. Importing the package is
        sufficient: mmengine resolves an unscoped key by searching the
        whole registry tree from the root, so a class registered in the
        mmpretrain registry is found from the mmdet registry;
      * every module string declared in ``cfg.custom_imports`` — the local
        Mamba/SSM backbones, the SensorBalancedSampler, and custom hooks —
        so the efficiency build (which bypasses Runner) sees the same
        registrations Runner.from_cfg would have performed.
    """
    try:
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=False)
    except Exception as exc:                       # noqa: BLE001
        print(f'[register] mmdet register_all_modules skipped: {exc}')

    try:
        import mmpretrain.models  # noqa: F401
    except ImportError:
        print('[register] mmpretrain not importable; ConvNeXt and related '
              'classification backbones may be unavailable.')

    ci = cfg.get('custom_imports')
    if ci and ci.get('imports'):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(
            ci['imports'],
            allow_failed_imports=ci.get('allow_failed_imports', False))
        print(f'[register] custom_imports: {list(ci["imports"])}')

    # Scope mmpretrain-sourced backbone types. Some mmengine builds do not
    # resolve an unscoped key (e.g. 'ConvNeXt') from the mmdet registry
    # across the mmpretrain boundary, even after the package is imported;
    # the build then raises KeyError. Backbones that live in mmdet core
    # (ResNet, SwinTransformer, PVTv2, ...) resolve on the bare name and
    # are left untouched. A bare type that is absent from the mmdet
    # registry but present in the mmpretrain registry is rewritten to its
    # scoped form 'mmpretrain.<Type>', which mmengine resolves
    # deterministically. The rewrite is applied to base_cfg in place, so
    # both the Runner accuracy pass and the bare-build efficiency pass
    # (which each deep-copy base_cfg afterwards) inherit it.
    _scope_mmpretrain_backbone(cfg)


def _scope_mmpretrain_backbone(cfg: Config) -> None:
    """Rewrites a bare mmpretrain backbone type to its scoped form."""
    model = cfg.get('model')
    if not model:
        return
    backbone = model.get('backbone')
    if not backbone or not isinstance(backbone.get('type'), str):
        return
    bb_type = backbone['type']
    if '.' in bb_type:                              # already scoped
        return

    from mmdet.registry import MODELS as MMDET_MODELS
    if bb_type in MMDET_MODELS:                      # in mmdet core; leave it
        return

    try:
        from mmpretrain.registry import MODELS as MMPRETRAIN_MODELS
    except ImportError:
        return
    if bb_type in MMPRETRAIN_MODELS:
        backbone['type'] = f'mmpretrain.{bb_type}'
        print(f'[register] backbone type "{bb_type}" not in mmdet registry; '
              f'scoped to "mmpretrain.{bb_type}".')


# ---------------------------------------------------------------------------
# GPU / device resolution
# ---------------------------------------------------------------------------
def resolve_device(requested: str) -> str:
    """Resolves and validates the compute device for inference.

    requested:
        'auto' -> use CUDA if available, else abort (production default:
                  a silent CPU fallback would run inference ~1-2 orders
                  of magnitude slower, which is almost never intended).
        'cuda' / 'cuda:N' -> require that CUDA device; abort if absent.
        'cpu'  -> explicit opt-in to CPU (only for debugging on a host
                  with no GPU).

    Returns the resolved device string. Aborts via SystemExit with a
    clear message if a GPU was required but is unavailable.
    """
    import torch

    cuda_ok = torch.cuda.is_available()

    if requested == 'cpu':
        print('[device] CPU explicitly requested. Inference will be slow.')
        return 'cpu'

    if requested == 'auto':
        if not cuda_ok:
            sys.exit('[ERROR] no CUDA device available and --device is '
                     '"auto". Inference on CPU is far slower and is not '
                     'done implicitly. Pass --device cpu to override, or '
                     'check that the container sees the GPU '
                     '(nvidia-smi, torch.cuda.is_available()).')
        device = 'cuda:0'
    else:  # explicit 'cuda' or 'cuda:N'
        if not cuda_ok:
            sys.exit(f'[ERROR] --device {requested} requested but CUDA is '
                     f'not available in this environment.')
        device = 'cuda:0' if requested == 'cuda' else requested

    idx = int(device.split(':')[1]) if ':' in device else 0
    n = torch.cuda.device_count()
    if idx >= n:
        sys.exit(f'[ERROR] --device {requested} requested but only {n} '
                 f'CUDA device(s) are visible (valid indices 0..{n - 1}).')

    name = torch.cuda.get_device_name(idx)
    total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    print(f'[device] using {device}  ({name}, {total_gb:.1f} GB)')
    return device


# ---------------------------------------------------------------------------
# Efficiency phase (sensor-independent; run once per model)
# ---------------------------------------------------------------------------
def _load_model_for_efficiency(base_cfg: Config,
                               args: argparse.Namespace) -> object:
    """Builds the detector and loads the checkpoint, ready for profiling.

    A dedicated minimal runner is used rather than reusing an accuracy
    pass runner, so the efficiency measurement is independent of any
    sensor's dataloader, evaluator, and work_dir state. Mirrors the device
    pinning of the accuracy pass so latency is measured on the same GPU.
    """
    from mmengine.runner import load_checkpoint

    cfg = copy.deepcopy(base_cfg) if 'copy' in globals() else base_cfg
    # We only need the model; build it directly via the registry rather
    # than spinning up a full Runner (no dataloader, no hooks).
    from mmdet.registry import MODELS
    from mmengine.registry import init_default_scope
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.eff_checkpoint, map_location='cpu')
    model.to(args.resolved_device)
    model.eval()
    return model


def run_efficiency(base_cfg: Config,
                   args: argparse.Namespace,
                   config_stem: str,
                   results_dir: str) -> Dict[str, object]:
    """Profiles one model and writes <config_stem>__efficiency.csv.

    Returns the flat efficiency dict. Sensor-independent: called once per
    model launch, after the accuracy passes. Skips gracefully (with a
    warning) if the device is CPU, since CPU latency is not comparable to
    the reported GPU figures.
    """
    if not args.resolved_device.startswith('cuda'):
        print('\n[efficiency] device is CPU; efficiency metrics skipped '
              '(GPU-only by protocol).')
        return {}

    print(f'\n{"=" * 76}')
    print(f'  EFFICIENCY PROFILE  ({config_stem})')
    print(f'  regime: batch 1, {args.eff_input}x{args.eff_input}, fp32, '
          f'warmup={args.eff_warmup}, timed_iters={args.eff_iters}')
    print(f'{"=" * 76}')

    model = _load_model_for_efficiency(base_cfg, args)

    # Some backbones break PyTorch's JIT tracer that fvcore relies on
    # (MambaVision's HuggingFace forward aborts with a C++ INTERNAL ASSERT
    # in FixupTraceScopeBlocks — a SIGABRT that no Python try/except can
    # catch). For those, skip the FLOP trace rather than let it kill the
    # process. FLOPs for these backbones are non-comparable regardless,
    # since fvcore cannot count their selective-scan operators. Params,
    # latency, FPS, and peak VRAM are unaffected and still reported.
    skip_flops = args.no_flops or any(
        tok.lower() in config_stem.lower() for tok in args.flops_skip)
    if skip_flops and not args.no_flops:
        print(f'[efficiency] FLOP trace skipped for {config_stem} '
              f'(matches --flops-skip; tracer-incompatible backbone).')

    profile = profile_model(
        model,
        device=args.resolved_device,
        input_hw=(args.eff_input, args.eff_input),
        warmup=args.eff_warmup,
        iters=args.eff_iters,
        measure_flops_too=not skip_flops,
    )
    if skip_flops and 'gflops' not in profile:
        profile.update(dict(
            gflops=None,
            flops_uncounted=True,
            flops_uncounted_ops=['skipped:tracer-incompatible'],
            flops_note=('FLOP trace skipped: backbone is incompatible with '
                        'the fvcore/JIT tracer. Params and latency are '
                        'unaffected.'),
        ))

    # Free the dedicated model before any further sensor work.
    del model
    import torch
    torch.cuda.empty_cache()

    # Console summary.
    gf = profile.get('gflops')
    gf_str = (f'{gf:.2f}' if isinstance(gf, float) else 'N/A')
    if profile.get('flops_uncounted'):
        gf_str += ' (undercount)'
    print(f'  Params (M)        : '
          f'{profile.get("params_total", 0) / 1e6:.2f}')
    print(f'  GFLOPs            : {gf_str}')
    print(f'  Latency mean (ms) : {profile.get("latency_mean_ms", 0):.2f} '
          f'+/- {profile.get("latency_std_ms", 0):.2f}')
    print(f'  FPS               : {profile.get("fps", 0):.2f}')
    print(f'  Peak VRAM (MB)    : {profile.get("peak_vram_mb", 0):.1f}')
    if profile.get('flops_note'):
        print(f'  NOTE              : {profile["flops_note"]}')
    print(f'{"=" * 76}')

    write_efficiency_csv(profile, results_dir, config_stem,
                         device_name=args.device_name)
    return profile


def write_efficiency_csv(profile: Dict[str, object],
                         results_dir: str,
                         config_stem: str,
                         device_name: str) -> None:
    """Writes a one-row, wide-form efficiency CSV for one model.

    Schema (single data row):
        model, params_total_M, params_trainable_M, gflops, flops_uncounted,
        flops_uncounted_ops, latency_mean_ms, latency_std_ms,
        latency_p50_ms, latency_p90_ms, fps, peak_vram_mb,
        eff_dtype, eff_input_h, eff_input_w, eff_warmup, eff_iters,
        device_name, flops_note
    Consumed by compile_results.py, which left-joins it onto every sensor
    row for this backbone.
    """
    out_path = osp.join(results_dir, f'{config_stem}__efficiency.csv')
    os.makedirs(results_dir, exist_ok=True)

    ops = profile.get('flops_uncounted_ops', []) or []
    row = {
        'model': config_stem,
        'params_total_M': f'{profile.get("params_total", 0) / 1e6:.4f}',
        'params_trainable_M':
            f'{profile.get("params_trainable", 0) / 1e6:.4f}',
        'gflops': ('' if profile.get('gflops') is None
                   else f'{profile["gflops"]:.4f}'),
        'flops_uncounted': int(bool(profile.get('flops_uncounted', False))),
        'flops_uncounted_ops': ';'.join(ops),
        'flops_scope': profile.get('flops_scope', ''),
        'latency_mean_ms': f'{profile.get("latency_mean_ms", 0):.4f}',
        'latency_std_ms': f'{profile.get("latency_std_ms", 0):.4f}',
        'latency_p50_ms': f'{profile.get("latency_p50_ms", 0):.4f}',
        'latency_p90_ms': f'{profile.get("latency_p90_ms", 0):.4f}',
        'fps': f'{profile.get("fps", 0):.4f}',
        'peak_vram_mb': f'{profile.get("peak_vram_mb", 0):.2f}',
        'eff_dtype': profile.get('eff_dtype', 'float32'),
        'eff_input_h': profile.get('eff_input_h', ''),
        'eff_input_w': profile.get('eff_input_w', ''),
        'eff_warmup': profile.get('eff_warmup', ''),
        'eff_iters': profile.get('eff_iters', ''),
        'device_name': device_name,
        'flops_note': profile.get('flops_note', ''),
    }
    with open(out_path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f'\n  Efficiency saved -> {out_path}\n')


# ---------------------------------------------------------------------------
# Single-sensor evaluation pass
# ---------------------------------------------------------------------------
def evaluate_one_sensor(
    base_cfg: Config,
    sensor: str,
    args: argparse.Namespace,
    config_stem: str,
    results_dir: str,
) -> Dict[str, float]:
    """Runs PalmBenchmarkMetric on one sensor's test set.

    The sensor's paths are taken from sensor_registry.py. Returns the
    cleaned metric dict, or an empty dict if the test set is missing.
    """
    entry = get_sensor(sensor)
    ann_file = entry['ann_file']
    img_prefix = entry['img_prefix']
    ann_stem = entry['label']

    # This sensor's designated checkpoint (Stage C diagonal protocol):
    # UAV<-best_UAV, GE<-best_GE, Aerial<-best_GE. For single-checkpoint
    # stages (A/B/D) every sensor maps to the same path.
    sensor_ckpt = args.ckpt_map[sensor]

    # Verify the test set exists before launching mmengine.
    if not osp.isfile(ann_file):
        print(f'\n[ERROR] {sensor}: test annotation not found: {ann_file}')
        print(f'        Check the path in sensor_registry.py. '
              f'Skipping {sensor}.\n')
        return {}
    if not osp.isdir(img_prefix):
        print(f'\n[ERROR] {sensor}: test image directory not found: '
              f'{img_prefix}')
        print(f'        Check the path in sensor_registry.py. '
              f'Skipping {sensor}.\n')
        return {}

    work_dir = (
        args.work_dir
        if args.work_dir
        else osp.join('/tmp', f'eval_{config_stem}__{ann_stem}')
    )
    os.makedirs(work_dir, exist_ok=True)

    # Patch a fresh copy of the config for this sensor.
    cfg = patch_config(
        base_cfg,
        ann_file=ann_file,
        img_prefix=img_prefix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        work_dir=work_dir,
        max_dets=args.max_dets,
        applied_score_thr=args.applied_score_thr,
    )
    cfg.load_from = sensor_ckpt

    # ---- Device / GPU setup --------------------------------------------
    # Pin the runner to the resolved device, and enable cudnn benchmark
    # so cuDNN selects the fastest convolution algorithms for the fixed
    # 1024x1024 inference input. resolved_device is set once in main().
    cfg.launcher = 'none'                       # single-process inference
    cfg.device = args.resolved_device
    env_cfg = dict(cfg.get('env_cfg', {}))
    env_cfg['cudnn_benchmark'] = (
        args.resolved_device.startswith('cuda'))
    cfg.env_cfg = env_cfg

    print(f'\n{"=" * 76}')
    print(f'  EVALUATING ON {sensor.upper()} TEST SET  '
          f'(stage: {entry.get("stage", "?")})')
    print(f'{"=" * 76}')
    print(f'  config     : {args.config}')
    print(f'  checkpoint : {sensor_ckpt}')
    print(f'  ann_file   : {ann_file}')
    print(f'  img_prefix : {img_prefix}')
    print(f'  batch_size : {cfg.test_dataloader["batch_size"]}')
    print(f'  num_workers: {cfg.test_dataloader["num_workers"]}')
    print(f'  device     : {args.resolved_device}')
    print(f'{"=" * 76}\n')

    runner = Runner.from_cfg(cfg)

    # Explicit device placement: ensure the model sits on the resolved
    # device irrespective of how the installed MMEngine version handles
    # cfg.device. resolve_device() has already guaranteed the device is
    # valid and available, so this cannot fail silently to CPU.
    try:
        runner.model.to(args.resolved_device)
    except Exception as exc:                       # noqa: BLE001
        sys.exit(f'[ERROR] failed to place model on '
                 f'{args.resolved_device}: {exc}')

    metrics = runner.test()

    # Strip the "palm/" prefix that mmengine prepends to metric keys.
    metrics_clean: Dict[str, float] = {}
    for k, v in metrics.items():
        bare = k.split('/', 1)[-1] if '/' in k else k
        metrics_clean[bare] = v

    print_summary(
        metrics_clean,
        config_path=args.config,
        ckpt_path=sensor_ckpt,
        ann_path=ann_file,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
    )

    out_path = osp.join(results_dir, f'{config_stem}__{ann_stem}.csv')
    write_csv(
        metrics_clean,
        out_path=out_path,
        config_stem=config_stem,
        ann_stem=ann_stem,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
    )
    return metrics_clean


# ---------------------------------------------------------------------------
# Cross-sensor summary
# ---------------------------------------------------------------------------
def print_cross_sensor_summary(
    sensor_metrics: Dict[str, Dict[str, float]],
    config_stem: str,
) -> None:
    """Side-by-side table of primary metrics across all evaluated sensors.

    Handles an arbitrary number of sensors, since the registry may
    declare more than two.
    """
    populated = {s: m for s, m in sensor_metrics.items() if m}
    if len(populated) < 2:
        return

    metrics_to_show = [
        ('mAP @ IoU 0.50',            'bbox_mAP_50',        'segm_mAP_50'),
        ('F1  @ best score, IoU 0.50', 'bbox_f1_opt',       'segm_f1_opt'),
        ('Precision @ best score',    'bbox_precision_opt', 'segm_precision_opt'),
        ('Recall    @ best score',    'bbox_recall_opt',    'segm_recall_opt'),
        ('Best score threshold',      'bbox_best_score_thr', 'segm_best_score_thr'),
    ]

    sensors = list(populated.keys())
    col_w = 13
    line1 = '=' * (40 + col_w * 2 * len(sensors))
    line2 = '-' * len(line1)

    print(f'\n{line1}')
    print(f'  CROSS-SENSOR SUMMARY -- {config_stem}')
    print(line2)
    header = f'  {"Metric (PRIMARY)":<38}'
    for s in sensors:
        header += f'{s + " bbox":>{col_w}}{s + " segm":>{col_w}}'
    print(header)
    print(line2)

    for label, bk, sk in metrics_to_show:
        row = f'  {label:<38}'
        for s in sensors:
            m = populated[s]
            row += (f'{m.get(bk, float("nan")):>{col_w}.4f}'
                    f'{m.get(sk, float("nan")):>{col_w}.4f}')
        print(row)
    print(line1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    valid = available_sensors()
    p = argparse.ArgumentParser(
        description='Evaluate one trained model on one or more test sets '
                    '(sensors defined in sensor_registry.py).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--config', required=True,
                   help='MMDetection config (.py) used for training.')
    p.add_argument('--checkpoint', required=False, default=None,
                   help='Best-checkpoint .pth from the training run. Applied '
                        'to every --sensor unless overridden per sensor by '
                        '--sensor-checkpoints. Required unless '
                        '--sensor-checkpoints covers all requested sensors.')
    p.add_argument('--sensor-checkpoints', nargs='+', default=None,
                   metavar='SENSOR=PATH',
                   help='Per-sensor checkpoint overrides for the Stage C '
                        'diagonal protocol, e.g. '
                        'UAV=/.../best_UAV_*.pth GE=/.../best_GE_*.pth '
                        'Aerial=/.../best_GE_*.pth . A sensor not listed '
                        'here falls back to --checkpoint. Each sensor is '
                        'evaluated against exactly one checkpoint, so the '
                        'reported cell is the intended diagonal entry.')
    p.add_argument('--sensors', nargs='+', default=['GE', 'Aerial'],
                   metavar='SENSOR',
                   help=f'Which test sets to evaluate. Valid keys come '
                        f'from sensor_registry.py: {valid}. '
                        f'Default: GE Aerial.')
    p.add_argument('--batch-size', type=int, default=None,
                   help='Test DataLoader batch size (default: from config).')
    p.add_argument('--num-workers', type=int, default=None,
                   help='Test DataLoader workers (default: from config). '
                        'Use 0 on a memory-constrained container.')
    p.add_argument('--device', default='auto',
                   metavar='DEVICE',
                   help='Compute device for inference. "auto" (default) '
                        'uses the GPU and aborts if none is found; '
                        '"cuda" / "cuda:N" target a specific GPU; "cpu" '
                        'forces CPU (debugging only, much slower).')
    p.add_argument('--score-thr', type=float, default=0.05,
                   help='Fixed score threshold for protocol traceability. '
                        'Locked at 0.05.')
    p.add_argument('--iou-thr', type=float, default=0.50,
                   help='IoU threshold for TP matching. Locked at 0.50.')
    p.add_argument('--max-dets', type=int, default=100,
                   help='Per-image detection cap for both COCO passes '
                        '(mAP and P/R/F1), forwarded to '
                        'PalmBenchmarkMetric. COCO default 100 caps recall '
                        'on dense plantation tiles; set to the densest '
                        'tile instance count (<= model test_cfg '
                        'max_per_img) and use ONE value across all stages '
                        'and sensors for comparability.')
    p.add_argument('--applied-score-thr', type=float, default=None,
                   help='Fixed operating threshold for the reported '
                        'P/R/F1 (optimal) block, typically selected on a '
                        'held-out validation set. When omitted, the '
                        'operating point is swept on the test set.')
    p.add_argument('--results-dir', default='results/stage_b',
                   help='Directory for CSV outputs. Default: results/stage_b')
    p.add_argument('--eff-input', type=int, default=1024,
                   help='Square input size (px) for the efficiency '
                        'forward pass. Locked to the benchmark inference '
                        'size 1024.')
    p.add_argument('--eff-warmup', type=int, default=EFF_WARMUP,
                   help=f'Warm-up iterations discarded before timing '
                        f'(default {EFF_WARMUP}).')
    p.add_argument('--eff-iters', type=int, default=EFF_ITERS,
                   help=f'Timed iterations for latency statistics '
                        f'(default {EFF_ITERS}).')
    p.add_argument('--no-flops', action='store_true',
                   help='Skip FLOPs estimation (fvcore). Latency, FPS, '
                        'params, and VRAM are still measured.')
    p.add_argument('--flops-skip', nargs='*', default=['mambavision'],
                   metavar='SUBSTR',
                   help='Config-stem substrings whose FLOP trace is skipped '
                        '(tracer-incompatible backbones). Default: '
                        'mambavision, whose HuggingFace forward aborts the '
                        'JIT tracer. Other metrics are unaffected. Pass an '
                        'empty list to disable skipping.')
    p.add_argument('--no-efficiency', action='store_true',
                   help='Skip the efficiency phase entirely; produce only '
                        'the per-sensor accuracy CSVs.')
    p.add_argument('--work-dir', default=None,
                   help='Runner work_dir (default: /tmp/eval_<stem>__<sensor>).')
    p.add_argument('--cfg-options', nargs='+', action=DictAction,
                   help='Additional config overrides.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate requested sensors against the registry up front, so an
    # unknown key fails immediately with the list of valid keys.
    valid = available_sensors()
    unknown = [s for s in args.sensors if s not in valid]
    if unknown:
        sys.exit(f'[ERROR] unknown sensor(s) {unknown}. '
                 f'Valid keys in sensor_registry.py: {valid}')

    if abs(args.score_thr - 0.05) > 1e-6:
        print(f'\n[WARN] score_thr = {args.score_thr} deviates from the '
              f'protocol-locked value of 0.05.\n')
    if abs(args.iou_thr - 0.50) > 1e-6:
        print(f'\n[WARN] iou_thr = {args.iou_thr} deviates from the '
              f'protocol-locked value of 0.50.\n')

    # ---- Resolve the per-sensor checkpoint mapping ----------------------
    # Builds args.ckpt_map: {sensor -> resolved .pth path}. Each sensor is
    # evaluated against exactly one checkpoint. --sensor-checkpoints entries
    # override --checkpoint; any sensor not overridden falls back to it.
    import glob as _glob

    def _resolve_glob(pattern: str, sensor: str) -> str:
        """Expands a glob to a single .pth; aborts on zero or ambiguous match."""
        if any(ch in pattern for ch in '*?[]'):
            hits = sorted(_glob.glob(pattern))
            if not hits:
                sys.exit(f'[ERROR] {sensor}: no checkpoint matches glob '
                         f'{pattern}')
            if len(hits) > 1:
                sys.exit(f'[ERROR] {sensor}: glob {pattern} is ambiguous, '
                         f'matched {len(hits)} files: {hits}. Tighten it.')
            return hits[0]
        return pattern

    ckpt_map: Dict[str, str] = {}
    overrides: Dict[str, str] = {}
    if args.sensor_checkpoints:
        for item in args.sensor_checkpoints:
            if '=' not in item:
                sys.exit(f'[ERROR] --sensor-checkpoints entry "{item}" must '
                         f'be SENSOR=PATH.')
            s, path = item.split('=', 1)
            if s not in valid:
                sys.exit(f'[ERROR] --sensor-checkpoints sensor "{s}" is not '
                         f'a registry key. Valid: {valid}')
            overrides[s] = path

    for sensor in args.sensors:
        if sensor in overrides:
            ckpt_map[sensor] = _resolve_glob(overrides[sensor], sensor)
        elif args.checkpoint is not None:
            ckpt_map[sensor] = _resolve_glob(args.checkpoint, sensor)
        else:
            sys.exit(f'[ERROR] no checkpoint for sensor "{sensor}": it is '
                     f'not in --sensor-checkpoints and --checkpoint was not '
                     f'given.')
        if not osp.isfile(ckpt_map[sensor]):
            sys.exit(f'[ERROR] {sensor}: checkpoint not found: '
                     f'{ckpt_map[sensor]}')

    args.ckpt_map = ckpt_map
    # Efficiency is a property of the architecture, not of which checkpoint
    # iteration is loaded; the forward-pass cost, parameter count, and FLOPs
    # are identical across the two Stage C operating points. The first
    # requested sensor's checkpoint is used deterministically for the
    # efficiency profile.
    args.eff_checkpoint = ckpt_map[args.sensors[0]]
    print('\n[checkpoint mapping] each sensor -> its evaluated checkpoint:')
    for sensor in args.sensors:
        print(f'    {sensor:8s} <- {osp.basename(ckpt_map[sensor])}')
    print(f'    [efficiency uses] {osp.basename(args.eff_checkpoint)}')

    if not osp.isfile(args.config):
        sys.exit(f'[ERROR] config not found: {args.config}')

    # Resolve and validate the compute device once, up front, so an
    # unavailable GPU aborts before any model is loaded.
    args.resolved_device = resolve_device(args.device)

    # Capture the GPU name for efficiency-CSV provenance (single reference
    # machine for all efficiency figures — TITAN RTX per protocol).
    if args.resolved_device.startswith('cuda'):
        import torch as _torch
        _idx = (int(args.resolved_device.split(':')[1])
                if ':' in args.resolved_device else 0)
        args.device_name = _torch.cuda.get_device_name(_idx)
    else:
        args.device_name = 'cpu'

    base_cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        base_cfg.merge_from_dict(args.cfg_options)

    # Register every backbone/module class the config references, on both
    # the Runner path and the bare-build efficiency path, before any model
    # is constructed. Mirrors the ambient registration tools/train.py did.
    register_modules(base_cfg)

    config_stem = osp.splitext(osp.basename(args.config))[0]
    os.makedirs(args.results_dir, exist_ok=True)

    sensor_metrics: Dict[str, Dict[str, float]] = {}
    for sensor in args.sensors:
        sensor_metrics[sensor] = evaluate_one_sensor(
            base_cfg=base_cfg,
            sensor=sensor,
            args=args,
            config_stem=config_stem,
            results_dir=args.results_dir,
        )

    print_cross_sensor_summary(sensor_metrics, config_stem)

    # ---- Efficiency phase: once per model, after all accuracy passes ----
    if not args.no_efficiency:
        run_efficiency(
            base_cfg=base_cfg,
            args=args,
            config_stem=config_stem,
            results_dir=args.results_dir,
        )


if __name__ == '__main__':
    main()
