# =============================================================================
# File: mmdet/models/backbones/groupmamba_backbone.py
#
# MMDetection wrapper registering GroupMamba (Shaker et al., CVPR 2025) as
# "MM_GroupMamba".
#
# Notes:
#   * GroupMamba upstream only ships a classification model. We subclass it
#     minimally to collect the 4 per-stage NCHW feature maps needed by FPN.
#   * Source files live in /opt/groupmamba/classification/models/. We load
#     them as a synthetic package `group_mamba_src` by absolute path so that
#     (a) relative imports inside the repo resolve, and (b) no name collision
#     occurs with VMamba's identically-named csms6s.py under /opt/vmamba.
#   * Requires the CUDA kernel `selective_scan_cuda_core`: upstream csms6s.py
#     calls it directly, so the oflex variant alone is not enough. The
#     Dockerfile builds it from VMamba's kernels/selective_scan.
# =============================================================================

import sys
import types
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmdet.registry import MODELS


# -----------------------------------------------------------------------------
# Synthetic-package loader: group_mamba_src
# -----------------------------------------------------------------------------
_GM_DIR   = Path('/opt/groupmamba/classification/models')
_PKG_NAME = 'group_mamba_src'

_IMPORT_OK  = False
_IMPORT_ERR = None
GroupMamba = None


def _load_gm_module(mod_fname: str, mod_name: str):
    """Load a file under _GM_DIR as `_PKG_NAME.<mod_name>` submodule."""
    fpath = _GM_DIR / mod_fname
    if not fpath.is_file():
        raise FileNotFoundError(f"Missing: {fpath}")

    fqmn = f'{_PKG_NAME}.{mod_name}'
    if fqmn in sys.modules:
        return sys.modules[fqmn]

    spec = importlib.util.spec_from_file_location(
        fqmn, str(fpath), submodule_search_locations=[str(_GM_DIR)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqmn] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    # 1) Parent synthetic package
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_GM_DIR)]
        sys.modules[_PKG_NAME] = _pkg

    # 2) Load siblings first so relative imports inside groupmamba.py work.
    #    groupmamba.py's Block_mamba uses GroupMambaLayer which depends on
    #    ss2d / csms6s.
    _csms6s = _load_gm_module('csms6s.py', 'csms6s')
    _ss2d   = _load_gm_module('ss2d.py',   'ss2d')

    # 3) Load main model file
    _gm_mod = _load_gm_module('groupmamba.py', 'groupmamba')

    GroupMamba = getattr(_gm_mod, 'GroupMamba')
    _IMPORT_OK = True
except Exception as _e:
    _IMPORT_ERR = _e


# -----------------------------------------------------------------------------
# Backbone subclass: collects 4-stage NCHW features for FPN
# -----------------------------------------------------------------------------
class _Backbone_GroupMamba(nn.Module if GroupMamba is None else GroupMamba):
    """Subclass of GroupMamba that returns per-stage NCHW feature maps.

    Reimplements forward_features to:
      * apply norm{i} and reshape to NCHW for EVERY out_indices stage
        (upstream only does this for stages 0..num_stages-1; the final stage
         is kept as (B, N, C) for classification — we undo that).
      * skip the post_network (classification-only) path entirely.
    """

    def __init__(self, *args, out_indices=(0, 1, 2, 3), **kwargs):
        # Upstream always builds a classifier head; we'll just ignore it.
        # Keep num_classes > 0 to avoid nn.Identity() shape surprises
        # downstream, but we won't use the head.
        kwargs.setdefault('num_classes', 1000)
        super().__init__(*args, **kwargs)
        self.out_indices = tuple(out_indices)

    def forward_features(self, x):  # type: ignore[override]
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block       = getattr(self, f"block{i + 1}")
            norm        = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)

            # Apply norm, reshape to NCHW. (Upstream skips this on the last
            # stage for its classifier; we need NCHW at every out_indices.)
            x_norm = norm(x)
            feat   = x_norm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outs.append(feat)

            # Prepare x for the next stage's patch_embed, which expects NCHW.
            if i != self.num_stages - 1:
                x = feat
        return outs

    def forward(self, x):  # type: ignore[override]
        return self.forward_features(x)


# -----------------------------------------------------------------------------
# MMDetection-registered wrapper
# -----------------------------------------------------------------------------
@MODELS.register_module()
class MM_GroupMamba(BaseModule):
    """Wrap GroupMamba (Shaker et al., CVPR 2025) for MMDetection.

    Args:
        embed_dims (list[int]): channel sizes per stage. Tiny/Small share
            [64,128,348,_], Base uses [96,192,424,512].
        depths (list[int]): number of Block_mamba per stage.
        mlp_ratios (list[int]): MLP expansion ratio per stage. Default
            [8,8,4,4] (matches all three variants upstream).
        stem_hidden_dim (int): stem width (32 for Tiny, 64 for Small/Base).
        drop_path_rate (float): stochastic depth rate.
        out_indices (tuple[int]): which stages to return (default all 4).
        pretrained (str | None): path to ImageNet .pth to load.
        frozen_stages (int): freeze stages with index < frozen_stages. -1
            means nothing frozen.
    """

    def __init__(self,
                 embed_dims=(64, 128, 348, 448),
                 depths=(3, 4, 9, 3),
                 mlp_ratios=(8, 8, 4, 4),
                 stem_hidden_dim=32,
                 drop_path_rate=0.1,
                 out_indices=(0, 1, 2, 3),
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        if not _IMPORT_OK:
            raise RuntimeError(
                f"Could not load GroupMamba from {_GM_DIR}. "
                f"Underlying error: {_IMPORT_ERR}\n"
                f"Make sure /opt/groupmamba is present and "
                f"selective_scan_cuda_oflex is importable "
                f"(LD_LIBRARY_PATH must include torch/lib)."
            )

        self.logger = MMLogger.get_current_instance()
        self.frozen_stages = frozen_stages

        # Don't pass distillation head related kwargs - we strip heads anyway.
        kwargs.pop('num_classes', None)

        self.logger.info(
            f"[MM_GroupMamba] Building GroupMamba backbone: "
            f"embed_dims={list(embed_dims)}, depths={list(depths)}, "
            f"stem_hidden_dim={stem_hidden_dim}, dpr={drop_path_rate}, "
            f"out_indices={out_indices}, pretrained={pretrained}"
        )

        self.backbone = _Backbone_GroupMamba(
            in_chans=3,
            num_classes=1000,          # built but not used
            stem_hidden_dim=stem_hidden_dim,
            embed_dims=list(embed_dims),
            mlp_ratios=list(mlp_ratios),
            depths=list(depths),
            drop_path_rate=drop_path_rate,
            num_stages=4,
            distillation=True,
            out_indices=tuple(out_indices),
            **kwargs,
        )

        # Remove classification heads and post_network — they waste memory
        # and don't participate in detection training.
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        if hasattr(self.backbone, 'dist_head'):
            self.backbone.dist_head = nn.Identity()
        if hasattr(self.backbone, 'post_network'):
            self.backbone.post_network = nn.ModuleList()

        if pretrained:
            self._load_pretrained(pretrained)

        self._freeze_stages()

    # ------------------------------------------------------------------ #
    def _load_pretrained(self, ckpt_path):
        self.logger.info(f"[MM_GroupMamba] Loading pretrained: {ckpt_path}")
        state = torch.load(ckpt_path, map_location='cpu')
        # Official GroupMamba checkpoints usually store under 'model' key.
        for k in ('model', 'state_dict', 'module'):
            if isinstance(state, dict) and k in state and isinstance(state[k], dict):
                state = state[k]
                break

        # Strip 'backbone.' or 'module.' prefixes if present.
        new_state = {}
        for k, v in state.items():
            nk = k
            for prefix in ('backbone.', 'module.'):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            new_state[nk] = v

        # Drop classifier / distillation / post_network keys — we removed those.
        drop = [k for k in new_state
                if k.startswith(('head.', 'dist_head.', 'post_network.'))]
        for k in drop:
            new_state.pop(k)

        missing, unexpected = self.backbone.load_state_dict(new_state, strict=False)
        self.logger.info(
            f"[MM_GroupMamba] pretrained loaded. "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            self.logger.info(f"[MM_GroupMamba] first missing keys: {missing[:5]}")
        if unexpected:
            self.logger.info(f"[MM_GroupMamba] first unexpected keys: {unexpected[:5]}")

    # ------------------------------------------------------------------ #
    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return
        for i in range(1, self.frozen_stages + 1):
            pe = getattr(self.backbone, f'patch_embed{i}', None)
            bl = getattr(self.backbone, f'block{i}', None)
            nm = getattr(self.backbone, f'norm{i}', None)
            for mod in (pe, bl, nm):
                if mod is None:
                    continue
                mod.eval()
                for p in mod.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        return self

    # ------------------------------------------------------------------ #
    def forward(self, x):
        outs = self.backbone(x)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        return tuple(outs)
