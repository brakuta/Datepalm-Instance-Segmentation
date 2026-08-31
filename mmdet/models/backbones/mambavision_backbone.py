# ==========================================================================
# mambavision_backbone.py — OPTIMIZED for TITAN RTX 24GB
# ==========================================================================
# Location: /workspace/mmdetection/mmdet/models/backbones/mambavision_backbone.py
#
# CHANGES vs ORIGINAL:
#   1. Eliminated HuggingFace AutoModel per-forward overhead
#   2. Removed per-eval() channel re-validation (was ~20 wasted checks)
#   3. Added native gradient checkpointing support
#   4. Probing + validation deferred to FIRST GPU forward (Mamba SSM
#      kernels are CUDA-only and cannot run on CPU)
#   5. Both probing and validation run exactly ONCE, then never again
# ==========================================================================
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmdet.registry import MODELS

__all__: list[str] = []


def _patch_timm_if_needed() -> None:
    """Patch missing ImageNetInfo in older timm versions. Idempotent."""
    try:
        import timm.data
        if not hasattr(timm.data, 'ImageNetInfo'):
            class ImageNetInfo:
                def __init__(self, *args, **kwargs):
                    pass
            timm.data.ImageNetInfo = ImageNetInfo
    except ImportError:
        pass


_MAMBAVISION_REGISTRY: dict[str, dict] = {
    'mamba_tiny_vision_timm': {
        'model_id': 'nvidia/MambaVision-T-1K',
        'out_channels': (80, 160, 320, 640),
    },
    'mamba_small_vision_timm': {
        'model_id': 'nvidia/MambaVision-S-1K',
        'out_channels': (96, 192, 384, 768),
    },
    'mamba_base_vision_timm': {
        'model_id': 'nvidia/MambaVision-B-1K',
        'out_channels': (128, 256, 512, 1024),
    },
}

_DROP_PATH_KEY_PRIORITY = ('drop_path_rate', 'drop_path', 'drop_path_rates')


def _get_logger() -> MMLogger:
    """Returns MMLogger or falls back to stdlib logger."""
    try:
        return MMLogger.get_current_instance()
    except RuntimeError:
        import logging
        return logging.getLogger(__name__)


class MambaVisionBase(BaseModule):
    """
    Optimized wrapper for MambaVision → MMDetection FPN.

    Key improvements over the original:
      - Probing + channel validation run ONCE on first forward (on GPU),
        then never again. Mamba's selective_scan_cuda kernel requires
        CUDA tensors, so CPU-based init-time validation is impossible.
      - No per-eval() re-validation overhead
      - Gradient checkpointing support for memory savings
      - Frozen stages handled cleanly

    Args:
        model_name (str): HuggingFace model ID or local path.
        expected_channels (tuple[int,...]): Expected C per stage (4 values).
        pretrained (bool): Load pretrained weights. Default True.
        drop_path_rate (float): Stochastic depth. Default 0.0.
        out_indices (tuple[int,...]): Stages to return. Default (0,1,2,3).
        frozen_stages (int): Freeze stem + first N stages. -1 = none.
        local_files_only (bool): No HF Hub contact. Default False.
        gradient_checkpointing (bool): Checkpoint each level's forward.
            Saves ~30-40% activation memory. Recommended for 24GB GPU.
        init_cfg: MMEngine init config (unused when pretrained=True).
    """

    def __init__(
        self,
        model_name: str,
        expected_channels: tuple,
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
        out_indices: tuple = (0, 1, 2, 3),
        frozen_stages: int = -1,
        local_files_only: bool = False,
        gradient_checkpointing: bool = False,
        in_chans: int = 3,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)

        if kwargs:
            raise ValueError(
                f"[MambaVision] Unexpected kwargs: {sorted(kwargs.keys())}. "
                f"Check config for typos."
            )
        if not all(0 <= i <= 3 for i in out_indices):
            raise ValueError(
                f"[MambaVision] out_indices must be subset of (0,1,2,3), "
                f"got {out_indices}."
            )

        self.model_name = model_name
        self.out_indices = tuple(sorted(out_indices))
        self.expected_channels = expected_channels
        self.frozen_stages = frozen_stages
        self.use_grad_ckpt = gradient_checkpointing
        self.in_chans = in_chans

        # Deferred to first forward() call (needs CUDA)
        self._use_hidden_states: bool | None = None
        self._initialized = False

        _patch_timm_if_needed()

        logger = _get_logger()
        logger.info(
            f"[MambaVision] Loading: {model_name} | pretrained={pretrained} | "
            f"drop_path={drop_path_rate} | frozen_stages={frozen_stages} | "
            f"grad_ckpt={gradient_checkpointing}"
        )

        # ---- Load model via HuggingFace (once) ----
        from transformers import AutoConfig, AutoModel
        hf_kwargs = dict(
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        config = AutoConfig.from_pretrained(model_name, **hf_kwargs)
        self._apply_drop_path(config, drop_path_rate, logger)

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                model_name, config=config, **hf_kwargs
            )
        else:
            self.backbone = AutoModel.from_config(
                config, trust_remote_code=True
            )
            logger.warning("[MambaVision] pretrained=False — random weights!")

        # ---- Multispectral: widen the stem, keeping the pretrained RGB ----
        # MambaVision loads through HuggingFace AutoModel, which has no
        # in_chans argument, so the usual route (create the model at N
        # channels, then load an inflated checkpoint) is unavailable. Instead
        # the 3-channel model is loaded normally and its stem convolution is
        # replaced afterwards -- which is strictly better than checkpoint
        # surgery here, because it needs no HF cache path and cannot go stale
        # when the cache is refreshed.
        #
        # Inflation rule is identical to
        # configs/Custom/tools_staged/inflate_stem_to_nband.py --mode mean:
        # the pretrained filters occupy the first three input channels and the
        # remainder start at their mean. Keep the two in step -- a backbone
        # inflated by a different rule than its peers is not comparable to
        # them.
        if in_chans != 3:
            self._inflate_stem(in_chans, logger)

        # ---- Enable gradient checkpointing if supported ----
        # NOTE: HF's gradient_checkpointing_enable() exists on the base
        # class but raises ValueError for models that don't support it
        # (including MambaVision). We must try/except, not just hasattr.
        self._manual_grad_ckpt = False
        if gradient_checkpointing:
            hf_ckpt_ok = False
            if hasattr(self.backbone, 'gradient_checkpointing_enable'):
                try:
                    self.backbone.gradient_checkpointing_enable()
                    hf_ckpt_ok = True
                    logger.info("[MambaVision] HF gradient_checkpointing enabled.")
                except ValueError:
                    logger.info(
                        "[MambaVision] HF gradient_checkpointing not supported "
                        "by this model. Trying manual per-level checkpointing."
                    )

            # Check both wrapper structures: backbone.model.levels (HF typical)
            # or backbone.levels (some versions). Must match _freeze_stages logic.
            inner = self._get_inner()
            if not hf_ckpt_ok and hasattr(inner, 'levels'):
                self._manual_grad_ckpt = True
                logger.info(
                    "[MambaVision] Manual per-level gradient checkpointing "
                    "enabled. Saves ~30-40%% activation memory."
                )
            elif not hf_ckpt_ok:
                logger.warning(
                    "[MambaVision] gradient_checkpointing requested but "
                    "no supported mechanism found. Ignored."
                )

        # ---- Freeze stages ----
        self._freeze_stages(logger)

    # ------------------------------------------------------------------
    # First-forward probing & validation (runs ONCE on GPU, then cached)
    # ------------------------------------------------------------------
    def _first_forward_init(self, x: torch.Tensor) -> None:
        """
        Called exactly ONCE on the first forward() call.
        Probes output_hidden_states and validates channel dimensions.
        Must run on GPU because Mamba's selective_scan_cuda kernel
        requires CUDA tensors — cannot be done at __init__ time on CPU.
        """
        logger = _get_logger()
        device = x.device

        # 1) Probe output_hidden_states support
        # Channel count comes from the actual input, not a hardcoded 3: with
        # in_chans=8 the inflated stem rejects a 3-channel probe, and the
        # except clause below catches only TypeError, so the resulting
        # RuntimeError escapes and aborts the run before iteration 1.
        dummy = torch.zeros(1, x.shape[1], 64, 64, device=device)
        try:
            with torch.no_grad():
                self.backbone(dummy, output_hidden_states=True)
            self._use_hidden_states = True
            logger.info("[MambaVision] output_hidden_states: supported")
        except TypeError:
            self._use_hidden_states = False
            logger.info("[MambaVision] output_hidden_states: not supported")

        # 2) Validate channel dimensions
        with torch.no_grad():
            if self._use_hidden_states:
                outs = self.backbone(dummy, output_hidden_states=True)
            else:
                outs = self.backbone(dummy)
        features = self._extract_features(outs)

        if len(features) < 4:
            raise RuntimeError(
                f"[MambaVision] Expected >=4 stages, got {len(features)}."
            )
        for i, expected_c in enumerate(self.expected_channels):
            actual_c = features[i].shape[1]
            if actual_c != expected_c:
                raise RuntimeError(
                    f"[MambaVision] Stage {i} channel mismatch: "
                    f"expected {expected_c}, got {actual_c}. "
                    f"Check FPN in_channels."
                )

        shapes = [features[i].shape for i in self.out_indices]
        logger.info(f"[MambaVision] Channel validation passed. Shapes: {shapes}")
        self._initialized = True

    # ------------------------------------------------------------------
    # Helper: find inner model (HF wraps as backbone.model.X or backbone.X)
    # ------------------------------------------------------------------
    def _get_inner(self):
        """Find the inner model that has patch_embed and levels."""
        if hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'levels'):
            return self.backbone.model
        return self.backbone

    # ------------------------------------------------------------------
    # Stage freezing
    # ------------------------------------------------------------------
    def _freeze_stages(self, logger) -> None:
        """Freeze stem + first N stages. Sets eval() and requires_grad=False."""
        if self.frozen_stages < 0:
            return

        inner = self._get_inner()

        # Freeze stem / patch_embed
        if hasattr(inner, 'patch_embed'):
            inner.patch_embed.eval()
            for p in inner.patch_embed.parameters():
                p.requires_grad = False
            logger.info("[MambaVision] Froze patch_embed (stem).")

        # Freeze levels
        if hasattr(inner, 'levels'):
            n = min(self.frozen_stages, len(inner.levels))
            for i in range(n):
                inner.levels[i].eval()
                for p in inner.levels[i].parameters():
                    p.requires_grad = False
                logger.info(f"[MambaVision] Froze stage {i}.")

    def train(self, mode: bool = True):
        """Override to keep frozen stages in eval() mode."""
        result = super().train(mode)
        if mode and self.frozen_stages >= 0:
            inner = self._get_inner()
            if hasattr(inner, 'patch_embed'):
                inner.patch_embed.eval()
                for p in inner.patch_embed.parameters():
                    p.requires_grad = False
            if hasattr(inner, 'levels'):
                n = min(self.frozen_stages, len(inner.levels))
                for i in range(n):
                    inner.levels[i].eval()
                    for p in inner.levels[i].parameters():
                        p.requires_grad = False
        return result

    # ------------------------------------------------------------------
    # Feature extraction (static, no logging in hot path)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_features(outs) -> list | tuple:
        """Extract 4-stage features from HF model output. No logging."""
        if hasattr(outs, 'hidden_states') and outs.hidden_states is not None:
            return outs.hidden_states

        if (isinstance(outs, tuple) and len(outs) == 2
                and isinstance(outs[1], (list, tuple))):
            return outs[1]

        if isinstance(outs, (list, tuple)) and len(outs) >= 4:
            return outs[-4:]

        raise RuntimeError(
            f"[MambaVision] Cannot extract features from "
            f"{type(outs).__name__} (len={getattr(outs, '__len__', 'N/A')})"
        )

    # ------------------------------------------------------------------
    # Forward — clean hot path after first-call init
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple:
        # One-time init on first forward (needs CUDA for Mamba kernels)
        if not self._initialized:
            self._first_forward_init(x)

        # Manual per-level gradient checkpointing:
        # Instead of letting backbone.forward() run all levels normally,
        # we iterate levels ourselves and wrap each in torch checkpoint.
        # This recomputes activations during backward, saving ~30-40% VRAM.
        if self._manual_grad_ckpt and self.training:
            return self._forward_with_checkpointing(x)

        if self._use_hidden_states:
            outs = self.backbone(x, output_hidden_states=True)
        else:
            outs = self.backbone(x)

        feature_maps = self._extract_features(outs)
        return tuple(feature_maps[i] for i in self.out_indices)

    def _forward_with_checkpointing(self, x: torch.Tensor) -> tuple:
        """
        Manual gradient checkpointing: run each MambaVision level inside
        torch.utils.checkpoint.checkpoint() to trade compute for memory.
        """
        inner = self._get_inner()

        if not hasattr(inner, 'levels'):
            # Fallback: can't find levels, disable and use normal forward
            _get_logger().warning(
                "[MambaVision] Cannot find levels for manual checkpointing. "
                "Falling back to normal forward."
            )
            self._manual_grad_ckpt = False
            outs = self.backbone(x)
            return tuple(self._extract_features(outs)[i] for i in self.out_indices)

        # Patch embed (stem) — small, no need to checkpoint
        x = inner.patch_embed(x)

        # Run each level with gradient checkpointing
        feature_maps = []
        for level in inner.levels:
            def _level_forward(_x, _level=level):
                return _level(_x)

            x, xo = grad_checkpoint(
                _level_forward, x, use_reentrant=False
            )
            feature_maps.append(xo)

        return tuple(feature_maps[i] for i in self.out_indices)

    def __repr__(self) -> str:
        out_ch = tuple(self.expected_channels[i] for i in self.out_indices)
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"frozen_stages={self.frozen_stages}, "
            f"out_indices={self.out_indices}, "
            f"out_channels={out_ch}, "
            f"grad_ckpt={self.use_grad_ckpt})"
        )

    # ------------------------------------------------------------------
    # Stem inflation (multispectral input)
    # ------------------------------------------------------------------
    def _inflate_stem(self, in_chans: int, logger) -> None:
        """Replace the first 3-channel conv with an `in_chans` one."""
        import torch
        import torch.nn as nn

        target = None
        for name, m in self.backbone.named_modules():
            if isinstance(m, nn.Conv2d) and m.in_channels == 3:
                target = (name, m)
                break
        if target is None:
            raise RuntimeError(
                f"[MambaVision] in_chans={in_chans} requested but no 3-channel "
                f"Conv2d was found to widen. The model structure has changed; "
                f"inspect named_modules() before assuming this is safe -- "
                f"running anyway would train a multispectral arm whose stem "
                f"still reads three bands.")

        name, old = target
        new = nn.Conv2d(in_chans, old.out_channels, old.kernel_size,
                        stride=old.stride, padding=old.padding,
                        dilation=old.dilation, groups=old.groups,
                        bias=old.bias is not None)
        with torch.no_grad():
            w = old.weight.data
            new.weight.data.zero_()
            new.weight.data[:, :3] = w
            if in_chans > 3:
                new.weight.data[:, 3:] = w.mean(dim=1, keepdim=True).repeat(
                    1, in_chans - 3, 1, 1)
            if old.bias is not None:
                new.bias.data.copy_(old.bias.data)

        parent = self.backbone
        parts = name.split('.')
        for q in parts[:-1]:
            parent = getattr(parent, q)
        setattr(parent, parts[-1], new)
        logger.info(
            f"[MambaVision] stem '{name}' inflated 3 -> {in_chans} channels "
            f"(pretrained RGB kept, extra channels = RGB mean)")

    # ------------------------------------------------------------------
    # Drop path utility
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_drop_path(config, drop_path_rate: float, logger) -> None:
        config_keys = set(vars(config).keys())

        for key in _DROP_PATH_KEY_PRIORITY:
            if key in config_keys:
                old_val = getattr(config, key)
                setattr(config, key, drop_path_rate)
                logger.info(
                    f"[MambaVision] config.{key}: {old_val} -> {drop_path_rate}"
                )
                return

        fallback_keys = sorted(
            k for k in config_keys
            if 'drop' in k.lower() and 'path' in k.lower()
        )
        if fallback_keys:
            key = fallback_keys[0]
            old_val = getattr(config, key)
            setattr(config, key, drop_path_rate)
            logger.warning(
                f"[MambaVision] Fallback drop_path key '{key}' "
                f"({old_val} -> {drop_path_rate})."
            )
        else:
            logger.warning(
                f"[MambaVision] No drop_path key found in config. "
                f"rate={drop_path_rate} NOT applied."
            )


# ======================================================================
# Variant registration — explicit classes for clarity
# ======================================================================
def _make_mambavision_class(type_name: str, spec: dict) -> type:
    """Factory: create, document, and register a MambaVision variant."""

    def __init__(self, **kwargs):
        super(cls, self).__init__(
            model_name=spec['model_id'],
            expected_channels=spec['out_channels'],
            **kwargs,
        )

    cls = type(type_name, (MambaVisionBase,), {'__init__': __init__})
    cls.__doc__ = (
        f"MambaVision backbone wrapper.\n\n"
        f"  HuggingFace ID : {spec['model_id']}\n"
        f"  FPN in_channels: {spec['out_channels']}\n\n"
        f"See MambaVisionBase for full docs."
    )
    MODELS.register_module()(cls)
    return cls


for _name, _spec in _MAMBAVISION_REGISTRY.items():
    globals()[_name] = _make_mambavision_class(_name, _spec)
    __all__.append(_name)

__all__.append('MambaVisionBase')