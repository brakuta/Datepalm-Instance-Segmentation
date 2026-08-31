# =============================================================================
# File: mmdet/models/backbones/efficientvmamba_backbone.py
#
# MMDetection wrapper for EfficientVMamba (Pei et al., AAAI 2025).
#
# Loads /opt/efficientvmamba/classification/models/vmamba_efficient.py as a
# synthetic package `eff_vmamba_src` to avoid polluting sys.path or colliding
# with the original VMamba wrapper.
#
# IMPORTANT: vmamba_efficient.py uses an ABSOLUTE import:
#     from lib.models.operations import InvertedResidual
# where `lib` is a sibling directory at /opt/efficientvmamba/classification/lib.
# We therefore add /opt/efficientvmamba/classification to sys.path BEFORE
# loading vmamba_efficient.py so this import resolves.
#
# Registered name : MM_EFFVSSM
# Kernel dependency: selective_scan_cuda_oflex (already built from VMamba).
#
# Stage output channels for dims=96: [96, 192, 384, 768]
#
# Variant kwargs (from /opt/efficientvmamba/detection/configs/efficient/):
#   Small : depths=(2,2,4,2), dims=96, drop_path=0.2
#   Base  : depths=(2,2,9,2), dims=96, drop_path=0.2
# Common : ssm_d_state=16, ssm_dt_rank='auto', ssm_ratio=2.0, mlp_ratio=0.0,
#          downsample_version='v1', patchembed_version='v1', window_size=2
# =============================================================================

# Import torch FIRST so libc10.so is resolvable when the kernel loads.
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
# Synthetic-package loader: eff_vmamba_src
# -----------------------------------------------------------------------------
_EFF_CLASSIFICATION_DIR = Path('/opt/efficientvmamba/classification')
_EFF_MODELS_DIR         = _EFF_CLASSIFICATION_DIR / 'models'
_PKG_NAME               = 'eff_vmamba_src'

_IMPORT_OK              = False
_IMPORT_ERR             = None
Backbone_EfficientVSSM  = None


def _load_eff_module(mod_fname: str, mod_name: str):
    """Load a file under _EFF_MODELS_DIR as `_PKG_NAME.<mod_name>` submodule."""
    fpath = _EFF_MODELS_DIR / mod_fname
    if not fpath.is_file():
        raise FileNotFoundError(f"Missing: {fpath}")

    fqmn = f'{_PKG_NAME}.{mod_name}'
    if fqmn in sys.modules:
        return sys.modules[fqmn]

    spec = importlib.util.spec_from_file_location(
        fqmn, str(fpath), submodule_search_locations=[str(_EFF_MODELS_DIR)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqmn] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    # 1) Add /opt/efficientvmamba/classification to sys.path so that the
    #    absolute import `from lib.models.operations import InvertedResidual`
    #    inside vmamba_efficient.py resolves to
    #    /opt/efficientvmamba/classification/lib/models/operations.py.
    _classification_str = str(_EFF_CLASSIFICATION_DIR)
    if _classification_str not in sys.path:
        sys.path.insert(0, _classification_str)

    # 2) Parent synthetic package
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_EFF_MODELS_DIR)]
        sys.modules[_PKG_NAME] = _pkg

    # 3) Load vmamba.py first (vmamba_efficient.py imports from it)
    _load_eff_module('vmamba.py', 'vmamba')

    # 4) Load vmamba_efficient.py (contains EfficientVSSM + Backbone_EfficientVSSM)
    _mod = _load_eff_module('vmamba_efficient.py', 'vmamba_efficient')

    Backbone_EfficientVSSM = getattr(_mod, 'Backbone_EfficientVSSM')
    _IMPORT_OK = True

except Exception as _e:
    _IMPORT_ERR = _e


# -----------------------------------------------------------------------------
# MMDetection-registered wrapper  —  registered as "MM_EFFVSSM"
# -----------------------------------------------------------------------------
@MODELS.register_module(name='MM_EFFVSSM')
class MM_EFFVSSM(BaseModule):
    """Wrap EfficientVMamba's Backbone_EfficientVSSM for MMDetection.

    Args:
        out_indices (tuple[int]): stages whose features are returned.
        pretrained (str | None): path to ImageNet .pth/.ckpt. The upstream
            Backbone_EfficientVSSM.load_pretrained expects 'state_dict' key
            and strips 'module.module.' prefix automatically.
        frozen_stages (int): freeze stages with index < frozen_stages. -1
            means nothing frozen.
        **backbone_kwargs: forwarded to Backbone_EfficientVSSM. Typical keys:
            depths, dims, ssm_d_state, ssm_dt_rank, ssm_ratio, mlp_ratio,
            downsample_version, patchembed_version, window_size,
            drop_path_rate.
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
                f"Could not load EfficientVMamba Backbone_EfficientVSSM from "
                f"{_EFF_MODELS_DIR}.\n"
                f"Underlying error: {_IMPORT_ERR}\n"
                f"Ensure /opt/efficientvmamba is present and "
                f"selective_scan_cuda_oflex is importable."
            )

        self.logger = MMLogger.get_current_instance()
        self.frozen_stages = frozen_stages

        backbone_kwargs.pop('num_classes', None)
        if pretrained == "":
            pretrained = None

        self.logger.info(
            f"[MM_EFFVSSM] Building EfficientVMamba Backbone_EfficientVSSM: "
            f"out_indices={out_indices}, pretrained={pretrained}, "
            f"kwargs={backbone_kwargs}"
        )

        self.backbone = Backbone_EfficientVSSM(
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