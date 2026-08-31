# =============================================================================
# File: mmdet/models/backbones/spatialmamba_backbone.py
#
# MMDetection wrapper registering Spatial-Mamba's Backbone_SpatialMamba as
# "MM_SpatialMamba".
#
# We load the upstream source files as a SYNTHETIC PACKAGE named
# `spatial_mamba_src`, with `utils` as a sibling submodule of `spatialmamba`.
# This makes the relative import inside spatialmamba.py line 27:
#     from .utils import selective_scan_state_flop_jit, ...
# resolve correctly, without polluting the global sys.path/sys.modules with a
# bare `utils` name that could collide with VMamba / MambaVision / others
# mounted in the same container.
#
# Required CUDA kernels:
#   * selective_scan_cuda_oflex_rh  (/opt/spatial_mamba/kernels/selective_scan)
#   * dwconv2d                      (/opt/spatial_mamba/kernels/dwconv2d)
#
# The dwconv2d kernel needs TWO things, and they are easy to confuse:
#   1. the compiled extension `dwconv2d`, which exposes only the raw entry
#      points dwconv2d_fp32 / dwconv2dbias_fp32; and
#   2. the pure-Python autograd wrapper `Dwconv.dwconv_layer`, which defines
#      DepthwiseFunction on top of them.
# spatialmamba.py imports (2), not (1), inside a try/except that binds
# DepthwiseFunction = None on failure. A machine with only the .so therefore
# imports cleanly, BUILDS THE MODEL SUCCESSFULLY, and then dies on the first
# forward pass with 'NoneType' object has no attribute 'apply' -- after any
# expensive setup the caller has already done. We put the wrapper's directory
# on sys.path here so a mounted /opt/spatial_mamba is sufficient on its own,
# and we verify the binding below so the failure lands at construction time
# with a message that names the cause.
# =============================================================================

import sys
import types
import importlib.util
from pathlib import Path

from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmdet.registry import MODELS


# -----------------------------------------------------------------------------
# Synthetic-package loader for /opt/spatial_mamba/classification/models/
# -----------------------------------------------------------------------------
_SM_DIR   = Path('/opt/spatial_mamba/classification/models')
_DWC_DIR  = Path('/opt/spatial_mamba/kernels/dwconv2d')   # parent of Dwconv/
_UTILS_FP = _SM_DIR / 'utils.py'
_SPM_FP   = _SM_DIR / 'spatialmamba.py'
_PKG_NAME = 'spatial_mamba_src'       # synthetic package
_UTILS_MN = f'{_PKG_NAME}.utils'       # spatial_mamba_src.utils
_SPM_MN   = f'{_PKG_NAME}.spatialmamba'  # spatial_mamba_src.spatialmamba

_IMPORT_OK  = False
_IMPORT_ERR = None
Backbone_SpatialMamba = None

try:
    if not _UTILS_FP.is_file():
        raise FileNotFoundError(f"Missing: {_UTILS_FP}")
    if not _SPM_FP.is_file():
        raise FileNotFoundError(f"Missing: {_SPM_FP}")

    # 0) Make `Dwconv.dwconv_layer` importable BEFORE spatialmamba.py is
    #    executed, since its guarded import runs at module-exec time and a
    #    later path fix cannot undo the None binding. Appended, not inserted,
    #    so it can never shadow an installed copy on a machine that has one.
    if _DWC_DIR.is_dir() and str(_DWC_DIR) not in sys.path:
        sys.path.append(str(_DWC_DIR))

    # 1) Create an empty parent package pointing at the real folder, so
    #    relative imports inside the loaded files can find siblings.
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_SM_DIR)]
        sys.modules[_PKG_NAME] = _pkg

    # 2) Load utils.py as spatial_mamba_src.utils
    if _UTILS_MN not in sys.modules:
        _u_spec = importlib.util.spec_from_file_location(
            _UTILS_MN, str(_UTILS_FP),
            submodule_search_locations=[str(_SM_DIR)],
        )
        _u_mod = importlib.util.module_from_spec(_u_spec)
        sys.modules[_UTILS_MN] = _u_mod
        _u_spec.loader.exec_module(_u_mod)

    # 3) Load spatialmamba.py as spatial_mamba_src.spatialmamba.
    #    Its `from .utils import ...` will now resolve to the module above.
    if _SPM_MN not in sys.modules:
        _s_spec = importlib.util.spec_from_file_location(
            _SPM_MN, str(_SPM_FP),
            submodule_search_locations=[str(_SM_DIR)],
        )
        _s_mod = importlib.util.module_from_spec(_s_spec)
        sys.modules[_SPM_MN] = _s_mod
        _s_spec.loader.exec_module(_s_mod)
    else:
        _s_mod = sys.modules[_SPM_MN]

    # 4) Verify the depthwise kernel actually bound. spatialmamba.py swallows
    #    this import, so without an explicit check the model builds and only
    #    fails deep inside the first forward pass.
    if getattr(_s_mod, 'DepthwiseFunction', None) is None:
        try:
            from Dwconv.dwconv_layer import DepthwiseFunction  # noqa: F401
            _why = ("it imports now, so spatialmamba.py was executed before "
                    f"{_DWC_DIR} reached sys.path")
        except Exception as _de:
            _why = f"{type(_de).__name__}: {_de}"
        raise ImportError(
            "spatialmamba.py bound DepthwiseFunction = None, so every "
            "forward pass would fail. It imports "
            "`from Dwconv.dwconv_layer import DepthwiseFunction` -- the "
            "pure-Python autograd wrapper, NOT the compiled `dwconv2d` "
            "extension. Copying dwconv2d*.so alone is not enough. "
            f"Expected the wrapper under {_DWC_DIR}/Dwconv/. Cause: {_why}")

    Backbone_SpatialMamba = getattr(_s_mod, 'Backbone_SpatialMamba')
    _IMPORT_OK = True
except Exception as _e:
    _IMPORT_ERR = _e


# -----------------------------------------------------------------------------
# MMDetection-registered wrapper
# -----------------------------------------------------------------------------
@MODELS.register_module()
class MM_SpatialMamba(BaseModule):
    """Wrap Spatial-Mamba's Backbone_SpatialMamba for MMDetection."""

    def __init__(self,
                 out_indices=(0, 1, 2, 3),
                 pretrained=None,
                 frozen_stages=-1,
                 norm_layer='ln',
                 init_cfg=None,
                 **backbone_kwargs):
        super().__init__(init_cfg=init_cfg)

        if not _IMPORT_OK:
            raise RuntimeError(
                f"Could not load Backbone_SpatialMamba from {_SPM_FP}. "
                f"Underlying error: {_IMPORT_ERR}\n"
                f"Make sure /opt/spatial_mamba is mounted and the CUDA "
                f"kernels selective_scan_cuda_oflex_rh and dwconv2d are built."
            )

        self.logger = MMLogger.get_current_instance()
        self.frozen_stages = frozen_stages

        backbone_kwargs.pop('num_classes', None)
        if pretrained == "":
            pretrained = None

        self.logger.info(
            f"[MM_SpatialMamba] Building Backbone_SpatialMamba: "
            f"out_indices={out_indices}, norm_layer={norm_layer}, "
            f"pretrained={pretrained}, kwargs={backbone_kwargs}"
        )

        self.backbone = Backbone_SpatialMamba(
            out_indices=out_indices,
            pretrained=pretrained,
            norm_layer=norm_layer,
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
