# ==========================================================================
# ms_data_preprocessor.py
# --------------------------------------------------------------------------
# N-band (>3 channel) data preprocessor for the multispectral Stage D arm.
#
# WHY THIS EXISTS
#   mmengine's ImgDataPreprocessor.__init__ hard-asserts len(mean) in (1, 3),
#   "to be compatible with RGB or gray image": every MS Stage D config
#   passes an 8-length mean/std (the measured MS_MEAN/MS_STD from
#   ms_pipelines.py), so the stock DetDataPreprocessor cannot even be
#   constructed. All four backbones fail identically at MODEL.build(),
#   before the backbone, the checkpoint, or any data is touched:
#     AssertionError: `mean` should have 1 or 3 values, to be compatible
#     with RGB or gray image, but got 8 values
#   The stem inflation, the multispectral loader and the multispectral
#   photometric augmentation were all built for N-band input; this is the
#   one piece of the pipeline that was not, and training cannot start
#   without it.
#
#   The length check exists because the assertion also gates the bgr<->rgb
#   channel-swap logic, which is only meaningful for a 3-channel image.
#   Everything else in the base class is already channel-count agnostic:
#   buffer registration (`torch.tensor(mean).view(-1, 1, 1)`), the
#   normalisation broadcast in forward(), and mmengine's stack_batch() all
#   work unmodified for N channels -- confirmed by reading
#   mmengine/model/base_model/data_preprocessor.py and
#   mmengine/model/utils.py directly rather than assumed. The only
#   3-channel-specific branch in forward() is gated on
#   `self.mean.shape[0] == 3`, so it is simply skipped for N=8.
#
# WHAT THIS CLASS DOES
#   Subclasses mmdet's DetDataPreprocessor and re-implements only the
#   mean/std buffer registration, bypassing ImgDataPreprocessor's length
#   assertion. Every other feature (mask padding, batch augments, box-type
#   conversion, pad_size_divisor) is inherited verbatim by constructing the
#   parent with mean=std=None (which also skips its assertion, since that
#   only fires `if mean is not None`) and then registering the real N-band
#   buffers afterwards.
#
#   bgr_to_rgb / rgb_to_bgr are rejected outright rather than silently
#   accepted: every MS config already sets both False (an RGB<->BGR swap
#   would scramble the first three channels away from the ImageNet
#   statistics the inflated stem expects), and accepting them here without
#   enforcing that would let a future config silently reintroduce the bug
#   this file exists to prevent.
#
# USAGE
#   In a config: data_preprocessor=dict(
#       type='MultispectralDetDataPreprocessor',
#       mean=MS_MEAN, std=MS_STD, bgr_to_rgb=False, ...)
#   Add 'configs.Custom._base_palm.ms_data_preprocessor' to custom_imports.
# ==========================================================================

from numbers import Number
from typing import List, Optional, Sequence, Union

import torch

from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.registry import MODELS


@MODELS.register_module()
class MultispectralDetDataPreprocessor(DetDataPreprocessor):
    """DetDataPreprocessor without the RGB/gray-only mean/std length check."""

    def __init__(self,
                mean: Optional[Sequence[Number]] = None,
                std: Optional[Sequence[Number]] = None,
                pad_size_divisor: int = 1,
                pad_value: Union[float, int] = 0,
                pad_mask: bool = False,
                mask_pad_value: int = 0,
                pad_seg: bool = False,
                seg_pad_value: int = 255,
                bgr_to_rgb: bool = False,
                rgb_to_bgr: bool = False,
                boxtype2tensor: bool = True,
                non_blocking: Optional[bool] = False,
                batch_augments: Optional[List[dict]] = None):
        if bgr_to_rgb or rgb_to_bgr:
            raise ValueError(
                'MultispectralDetDataPreprocessor does not support '
                'bgr_to_rgb or rgb_to_bgr. A channel swap is only '
                'meaningful for a 3-channel image; on an N-band input it '
                'would scramble the first three channels away from the '
                'statistics describing them. Pass bgr_to_rgb=False, '
                'rgb_to_bgr=False and keep the channel order the tiler '
                'wrote.')
        if (mean is None) != (std is None):
            raise ValueError('mean and std must be both None or both given')

        # Build the parent with mean=std=None so ImgDataPreprocessor's
        # length-restricted assertion is never reached. That also skips
        # buffer registration there, which is done below instead, without
        # the length restriction.
        super().__init__(
            mean=None, std=None,
            pad_size_divisor=pad_size_divisor, pad_value=pad_value,
            pad_mask=pad_mask, mask_pad_value=mask_pad_value,
            pad_seg=pad_seg, seg_pad_value=seg_pad_value,
            bgr_to_rgb=False, rgb_to_bgr=False,
            boxtype2tensor=boxtype2tensor, non_blocking=non_blocking,
            batch_augments=batch_augments)

        if mean is not None:
            self.register_buffer(
                'mean',
                torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1),
                False)
            self.register_buffer(
                'std',
                torch.tensor(std, dtype=torch.float32).view(-1, 1, 1),
                False)
            self._enable_normalize = True
        else:
            self._enable_normalize = False
