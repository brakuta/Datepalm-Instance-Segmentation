# =============================================================================
# File: mmdet/models/backbones/vmamba_backbone.py
#
# MMDetection wrapper that registers the official VMamba Backbone_VSSM class.
# Backbone_VSSM lives in /opt/vmamba/classification/models/vmamba.py and
# already handles: multi-stage feature output, outnorm layers, NHWC->NCHW
# conversion, and pretrained checkpoint loading.
# =============================================================================

import sys
import torch.nn as nn

from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmdet.registry import MODELS


# -----------------------------------------------------------------------------
# Path setup: point to the official VMamba repo at /opt/vmamba
# -----------------------------------------------------------------------------
for _p in ('/opt/vmamba', '/opt/vmamba/classification'):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from classification.models.vmamba import Backbone_VSSM
    _IMPORT_OK = True
    _IMPORT_ERR = None
except ImportError as _e:
    _IMPORT_OK = False
    _IMPORT_ERR = _e
    Backbone_VSSM = None


# -----------------------------------------------------------------------------
# Thin MMDetection wrapper
# -----------------------------------------------------------------------------
@MODELS.register_module()
class MM_VSSM(BaseModule):
    """Register VMamba's own Backbone_VSSM with MMDetection.

    All keyword arguments are forwarded directly to Backbone_VSSM.
    Backbone_VSSM handles:
      * multi-stage feature extraction (out_indices)
      * per-stage output LayerNorms
      * NHWC -> NCHW conversion for FPN
      * pretrained checkpoint loading
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
                f"Could not import Backbone_VSSM from /opt/vmamba. "
                f"Underlying error: {_IMPORT_ERR}"
            )

        self.logger = MMLogger.get_current_instance()
        self.frozen_stages = frozen_stages

        # Do not pass 'num_classes' to a detection backbone.
        backbone_kwargs.pop('num_classes', None)

        self.logger.info(
            f"[MM_VSSM] Building Backbone_VSSM with out_indices={out_indices}, "
            f"pretrained={pretrained}, kwargs={backbone_kwargs}"
        )

        # Backbone_VSSM itself owns: patch_embed, layers, outnormN, forward().
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
        # Backbone_VSSM returns a list of NCHW feature maps, one per out_index.
        outs = self.backbone(x)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        return tuple(outs)
