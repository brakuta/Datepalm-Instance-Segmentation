# =============================================================================
# _base_palm/loading_multispectral.py
# -----------------------------------------------------------------------------
# Transform for reading N-band GeoTIFF tiles.
#
# WHY THIS EXISTS
#   MMDetection's LoadImageFromFile goes through mmcv.imread, which returns a
#   3-channel BGR array whatever the file holds. Point it at an 8-band
#   WorldView-3 tile and it silently hands the model three bands -- no error,
#   no warning, and a "multispectral" experiment that is quietly RGB. That
#   failure is invisible in the metrics, which is what makes it dangerous.
#
# CHANNEL ORDER IS THE FILE'S ORDER
#   No BGR conversion happens here, and `bgr_to_rgb` MUST be False in the
#   data_preprocessor when this transform is used -- otherwise the first three
#   channels get reversed and channel 0 of the mean/std vector no longer
#   describes channel 0 of the data.
#
#   The tiler writes bands in the order the job requested, so a job with
#   "bands": [5, 3, 2, 7] on a WorldView-3 8-band product yields tiles whose
#   channels are literally R, G, B, NIR1. Keep the job's band list and the
#   data_preprocessor mean/std in the same order, always.
# =============================================================================

from typing import Optional

import numpy as np

from mmcv.transforms.base import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadMultispectralImageFromFile(BaseTransform):
    """Read an N-band raster as HxWxC.

    Args:
        to_float32: cast to float32 on load. Leave False for uint8 tiles;
            DetDataPreprocessor casts and normalises anyway.
        expected_channels: if set, the transform RAISES when a file does not
            have exactly this many bands. Strongly recommended. A dataset
            where a handful of tiles were written with a different band
            selection is otherwise impossible to detect from the metrics --
            the model just trains on inconsistent inputs.
        color_type: accepted and ignored, so this can be dropped into a
            pipeline in place of LoadImageFromFile without editing siblings.
    """

    def __init__(self,
                 to_float32: bool = False,
                 expected_channels: Optional[int] = None,
                 color_type: str = 'unchanged',
                 backend_args: Optional[dict] = None) -> None:
        self.to_float32 = to_float32
        self.expected_channels = expected_channels
        self.color_type = color_type
        self.backend_args = backend_args

    def transform(self, results: dict) -> dict:
        import rasterio

        path = results['img_path']
        with rasterio.open(path) as src:
            arr = src.read()                       # (C, H, W)
        img = np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))

        if self.expected_channels is not None and \
                img.shape[2] != self.expected_channels:
            raise ValueError(
                f'{path}: expected {self.expected_channels} bands, found '
                f'{img.shape[2]}. Every tile in a split must carry the same '
                f'bands in the same order, or the mean/std vector no longer '
                f'describes the data it is applied to.')

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'to_float32={self.to_float32}, '
                f'expected_channels={self.expected_channels})')


@TRANSFORMS.register_module()
class MultispectralPhotoMetricDistortion(BaseTransform):
    """PhotoMetricDistortion for N-band imagery.

    WHY THE STOCK TRANSFORM CANNOT BE USED
      mmdet's PhotoMetricDistortion converts BGR to HSV with cv2 to apply
      saturation and hue. cv2 refuses anything but 3 channels, so on a 4-band
      tile the pipeline raises -- and the tempting fix, dropping the transform
      from the multispectral pipeline, silently gives the MS arm WEAKER
      augmentation than the RGB arm it is being compared against. The
      comparison would then confound spectral content with regularisation.

    WHAT THIS DOES INSTEAD
      Brightness and contrast are applied to EVERY channel with the same
      sampled values, which is what a change in illumination physically does:
      it scales all bands together, near-infrared included.

      Saturation and hue are applied to the first three channels only, since
      they are defined in a colour space that has no meaning for NIR. The RGB
      channels therefore receive exactly the treatment they receive in the RGB
      arm, and the extra bands receive the part of it that transfers.

      Defaults match dataset_sat_30cm_staged.py so the two arms differ in
      spectral content and nothing else that can be helped.
    """

    def __init__(self,
                 brightness_delta: int = 48,
                 contrast_range: tuple = (0.4, 1.6),
                 saturation_range: tuple = (0.4, 1.6),
                 hue_delta: int = 24) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def transform(self, results: dict) -> dict:
        import cv2
        img = results['img'].astype(np.float32)
        n_c = img.shape[2]

        # --- brightness and contrast: all bands, shared values -------------
        if np.random.randint(2):
            img += np.random.uniform(-self.brightness_delta,
                                     self.brightness_delta)
        contrast_first = np.random.randint(2)
        if contrast_first and np.random.randint(2):
            img *= np.random.uniform(self.contrast_lower, self.contrast_upper)

        # --- saturation and hue: RGB channels only -------------------------
        if n_c >= 3:
            rgb = np.clip(img[:, :, :3], 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            if np.random.randint(2):
                hsv[:, :, 1] *= np.random.uniform(self.saturation_lower,
                                                  self.saturation_upper)
            if np.random.randint(2):
                hsv[:, :, 0] = (hsv[:, :, 0].astype(int) +
                                np.random.randint(-self.hue_delta,
                                                  self.hue_delta)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img[:, :, :3] = cv2.cvtColor(
                np.clip(hsv, 0, 255).astype(np.uint8),
                cv2.COLOR_HSV2RGB).astype(np.float32)

        if not contrast_first and np.random.randint(2):
            img *= np.random.uniform(self.contrast_lower, self.contrast_upper)

        results['img'] = np.clip(img, 0, 255).astype(np.uint8)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'brightness_delta={self.brightness_delta}, '
                f'contrast_range=({self.contrast_lower}, {self.contrast_upper}), '
                f'saturation_range=({self.saturation_lower}, '
                f'{self.saturation_upper}), hue_delta={self.hue_delta})')


# =============================================================================
# N-band padding
# -----------------------------------------------------------------------------
# WHY THIS EXISTS
#   mmcv's Pad routes through cv2.copyMakeBorder, whose constant border value
#   is an OpenCV Scalar and therefore holds at most 4 components (BGRA). An
#   8-band tile padded with an 8-tuple dies on the first training batch:
#
#     cv2.error: (-5:Bad argument) in function 'copyMakeBorder'
#     > Scalar value for argument 'value' is longer than 4
#
#   Passing a plain int instead does not help: mmcv's Pad expands an int to a
#   per-channel tuple before calling impad, so it arrives at OpenCV the same
#   way. The limit is in OpenCV, not in how the value is written, so the fix
#   has to bypass cv2 for the padding itself.
#
#   Dropping the Pad from the MS pipeline was the other option and is worse:
#   the RGB arm pads identically, and the two arms must differ in spectral
#   content and nothing else, or an MS-vs-RGB difference stops being
#   attributable to the extra bands.
# =============================================================================


def pad_to_size(img: np.ndarray, target_hw, pad_val) -> np.ndarray:
    """Pads `img` to `target_hw` at the right and bottom, any channel count.

    Mirrors mmcv.impad(shape=...): padding goes to the right and bottom only,
    and a target smaller than the image in either axis leaves that axis
    untouched rather than cropping.

    pad_val may be a scalar (applied to every channel) or one value per
    channel.
    """
    h, w = img.shape[:2]
    target_h = max(int(target_hw[0]), h)
    target_w = max(int(target_hw[1]), w)

    if img.ndim == 2:
        fill = np.asarray(pad_val, dtype=img.dtype).reshape(-1)
        if fill.size != 1:
            raise ValueError(
                f'2-D image needs a scalar pad_val, got {fill.size} values')
        out = np.empty((target_h, target_w), dtype=img.dtype)
        out[...] = fill[0]
        out[:h, :w] = img
        return out

    c = img.shape[2]
    fill = np.asarray(pad_val, dtype=img.dtype).reshape(-1)
    if fill.size == 1:
        fill = np.repeat(fill, c)
    if fill.size != c:
        raise ValueError(
            f'pad_val has {fill.size} values but the image has {c} channels')
    out = np.empty((target_h, target_w, c), dtype=img.dtype)
    out[...] = fill          # broadcasts (C,) across (H, W, C)
    out[:h, :w] = img
    return out


try:                                    # pragma: no cover - import shim
    from mmdet.datasets.transforms.transforms import Pad as _MMDetPad
except Exception:                       # mmdet unavailable (e.g. doc build)
    _MMDetPad = None


if _MMDetPad is not None:

    @TRANSFORMS.register_module()
    class MultispectralPad(_MMDetPad):
        """Pad that supports >4 channels by padding with numpy, not cv2.

        Only image padding is overridden. Mask and segmentation padding are
        inherited unchanged -- those already go through numpy and are
        channel-count agnostic.
        """

        def _pad_img(self, results: dict) -> None:
            img = results['img']

            # <=4 channels is exactly what cv2 handles, so defer to the
            # parent there and keep this class safe to use in any pipeline.
            if img.ndim == 3 and img.shape[2] <= 4:
                return super()._pad_img(results)

            pad_val = self.pad_val.get('img', 0)

            # Target-size resolution, matching mmcv.transforms.Pad._pad_img.
            size = None
            if self.pad_to_square:
                max_size = max(img.shape[:2])
                size = (max_size, max_size)
            if self.size_divisor is not None:
                if size is None:
                    size = (img.shape[0], img.shape[1])
                pad_h = int(np.ceil(
                    size[0] / self.size_divisor)) * self.size_divisor
                pad_w = int(np.ceil(
                    size[1] / self.size_divisor)) * self.size_divisor
                size = (pad_h, pad_w)
            elif self.size is not None:
                size = self.size[::-1]      # config gives (w, h)

            padded = pad_to_size(img, size, pad_val)

            results['img'] = padded
            results['pad_shape'] = padded.shape
            results['pad_fixed_size'] = self.size
            results['pad_size_divisor'] = self.size_divisor
            results['img_shape'] = padded.shape[:2]
