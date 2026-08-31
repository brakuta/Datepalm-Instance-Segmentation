# ==========================================================================
# mmdet/models/backbones/mambaout_backbone.py
# --------------------------------------------------------------------------
# Thin mmdet wrapper around the OFFICIAL MambaOut implementation hosted on
# Hugging Face by the `timm` team (e.g. `timm/mambaout_tiny.in1k`,
# `timm/mambaout_small.in1k`, `timm/mambaout_base.in1k`).
#
# Same philosophy as the MambaVision wrapper:
#   - Architecture is loaded verbatim from the upstream library.
#   - No re-implementation, no monkey-patching, no head replacement logic
#     beyond what `features_only=True` already does.
#   - Wrapper only registers the model with mmdet and forwards the four
#     stage feature maps to FPN.
#
# Upstream entry point (HF-hosted, official):
#   timm.create_model('mambaout_tiny.in1k', pretrained=True,
#                     features_only=True, out_indices=(0,1,2,3))
#
# Stage strides : {4, 8, 16, 32}   ->  FPN P2..P5
# Stage channels: [96, 192, 384, 576]  for tiny / small / base
# ==========================================================================

import timm
from mmengine.model import BaseModule

from mmdet.registry import MODELS


_VARIANT_TO_TIMM_NAME = {
    "tiny":  "mambaout_tiny.in1k",
    "small": "mambaout_small.in1k",
    "base":  "mambaout_base.in1k",
}


@MODELS.register_module()
class MambaOutBackbone(BaseModule):
    """MambaOut backbone for mmdet.

    Args:
        variant (str): one of {'tiny', 'small', 'base'}. Maps to the
            corresponding `timm/mambaout_*.in1k` Hugging Face model.
        pretrained (bool): load the official ImageNet-1k weights from
            Hugging Face via timm. Default: True.
        init_cfg (dict | None): standard mmengine init config (left for
            user-supplied overrides; pretrained loading is handled by
            timm at construction time).
    """

    def __init__(self, variant="tiny", pretrained=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if variant not in _VARIANT_TO_TIMM_NAME:
            raise ValueError(
                f"Unknown MambaOut variant '{variant}'. "
                f"Expected one of {list(_VARIANT_TO_TIMM_NAME.keys())}."
            )

        # Upstream architecture, untouched. `features_only=True` returns
        # the four stage feature maps in NCHW with strides {4,8,16,32}.
        self.model = timm.create_model(
            _VARIANT_TO_TIMM_NAME[variant],
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # Channel info from timm — useful for verifying FPN in_channels.
        self.feature_info = self.model.feature_info

    def forward(self, x):
        # timm MambaOut returns features in NHWC; FPN expects NCHW.
        return tuple(f.permute(0, 3, 1, 2).contiguous() for f in self.model(x))
