# =============================================================================
# File: mmdet/models/backbones/msvmamba_backbone.py
#
# MMDetection wrapper for MSVMamba (Shi et al., NeurIPS 2024).
#
# NOT USED by any published config or reported result -- kept because the
# environment carries the /opt/msvmamba clone (see THIRD_PARTY.md) and the
# wrapper documents how it would be registered without colliding with
# VMamba. Safe to ignore.
#
# IMPORTANT: MSVMamba's source reuses the filename `vmamba.py` and the class
# names `VSSM` / `Backbone_VSSM` from the original VMamba repo, but the
# IMPLEMENTATIONS ARE DIFFERENT — MSVMamba invokes multi-scale scan blocks
# via forward_type="v3noz" and the utilities in utils.py.
#
# To avoid silent collisions with the original VMamba wrapper (which also
# registers as "MM_VSSM"), this file:
#   * Loads MSVMamba source as a synthetic package `ms_vmamba_src`
#   * Registers the wrapper under a DIFFERENT name: "MM_MSVSSM"
#
# Configs for MSVMamba must use type='MM_MSVSSM' — not 'MM_VSSM'.
#
# Kernel dependency: selective_scan_cuda_oflex (same kernel as VMamba; must
# already be built and LD_LIBRARY_PATH must include torch/lib).
# =============================================================================

# Import torch FIRST so libc10.so is resolvable when the kernel is loaded
# transitively by MSVMamba's utils.py.
import torch
import torch.nn as nn

import sys
import types
import importlib.util
from pathlib import Path

from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmdet.registry import MODELS


# -----------------------------------------------------------------------------
# Synthetic-package loader: ms_vmamba_src
# -----------------------------------------------------------------------------
_MSV_DIR  = Path('/opt/msvmamba/classification/models')
_PKG_NAME = 'ms_vmamba_src'

_IMPORT_OK  = False
_IMPORT_ERR = None
Backbone_VSSM = None


def _load_msv_module(mod_fname: str, mod_name: str):
    """Load a file under _MSV_DIR as `_PKG_NAME.<mod_name>` submodule."""
    fpath = _MSV_DIR / mod_fname
    if not fpath.is_file():
        raise FileNotFoundError(f"Missing: {fpath}")

    fqmn = f'{_PKG_NAME}.{mod_name}'
    if fqmn in sys.modules:
        return sys.modules[fqmn]

    spec = importlib.util.spec_from_file_location(
        fqmn, str(fpath), submodule_search_locations=[str(_MSV_DIR)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqmn] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    # 1) Parent synthetic package
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_MSV_DIR)]
        sys.modules[_PKG_NAME] = _pkg

    # 2) Load utils.py FIRST (vmamba.py line 16: `from .utils import ...`)
    _utils = _load_msv_module('utils.py', 'utils')

    # 3) Load csm_triton.py (may be imported by utils or vmamba)
    _csm_triton = _load_msv_module('csm_triton.py', 'csm_triton')

    # 4) Load the main vmamba.py (which contains VSSM and Backbone_VSSM)
    _mod = _load_msv_module('vmamba.py', 'vmamba')

    Backbone_VSSM = getattr(_mod, 'Backbone_VSSM')
    _IMPORT_OK = True
except Exception as _e:
    _IMPORT_ERR = _e


# -----------------------------------------------------------------------------
# MMDetection-registered wrapper  —  registered as "MM_MSVSSM"
# -----------------------------------------------------------------------------
@MODELS.register_module(name='MM_MSVSSM')
class MM_MSVSSM(BaseModule):
    """Wrap MSVMamba's Backbone_VSSM for MMDetection.

    MSVMamba is selected through the same VSSM class the original VMamba uses,
    but with MSVMamba-specific kwargs:
        forward_type="v3noz", ssm_ratio=2.0, ssm_d_state=1,
        downsample_version="v3", patchembed_version="v2", mlp_ratio=4.0

    Per-variant kwargs (from /opt/msvmamba/detection/configs/vssm1/):
        Tiny:  dims=96,  depths=(2,2,5,2),  drop_path_rate=0.2
        Small: dims=96,  depths=(2,2,15,2), drop_path_rate=0.3
        Base:  dims=128, depths=(2,2,15,2), drop_path_rate=0.6

    Args:
        out_indices (tuple[int]): stages whose features are returned.
        pretrained (str | None): path to ImageNet .pth. The upstream
            Backbone_VSSM.load_pretrained expects the state_dict under key
            'model' — handled automatically.
        frozen_stages (int): freeze stages with index < frozen_stages. -1
            means nothing frozen.
        **backbone_kwargs: forwarded to Backbone_VSSM. Typical keys: dims,
            depths, ssm_d_state, ssm_dt_rank, ssm_ratio, ssm_conv,
            ssm_conv_bias, forward_type, mlp_ratio, downsample_version,
            patchembed_version, drop_path_rate.
    """

    def __init__(self,
                 out_indices=(0, 1, 2, 3),
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None,
                 **backbone_kwargs):
        super().__init__(init_cfg=init_cfg)

        if not _IMPORT_OK:
            raise RuntimeError(
                f"Could not load MSVMamba Backbone_VSSM from {_MSV_DIR}. "
                f"Underlying error: {_IMPORT_ERR}\n"
                f"Ensure /opt/msvmamba is present and "
                f"selective_scan_cuda_oflex is importable."
            )

        self.logger = MMLogger.get_current_instance()
        self.frozen_stages = frozen_stages

        backbone_kwargs.pop('num_classes', None)
        if pretrained == "":
            pretrained = None

        self.logger.info(
            f"[MM_MSVSSM] Building MSVMamba Backbone_VSSM: "
            f"out_indices={out_indices}, pretrained={pretrained}, "
            f"kwargs={backbone_kwargs}"
        )

        self.backbone = Backbone_VSSM(
            out_indices=out_indices,
            pretrained=pretrained,
            **backbone_kwargs,
        )

        self._freeze_stages()

    # ------------------------------------------------------------------ #
    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return

        if hasattr(self.backbone, 'patch_embed'):
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False
            self.backbone.patch_embed.eval()

        if hasattr(self.backbone, 'layers'):
            n = min(self.frozen_stages, len(self.backbone.layers))
            for i in range(n):
                stage = self.backbone.layers[i]
                stage.eval()
                for p in stage.parameters():
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
