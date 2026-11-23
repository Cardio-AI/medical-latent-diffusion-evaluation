"""B32 FRDv1 extraction primitives with narrowly documented input repairs.

Parts adapt frd-score/frd.py by Richard Osuala, Nick Konz and Preeti Verma,
https://github.com/RichardObi/frd-score, revision
bf05c71fc44e461ab123bd273e58390b90f9cf54, under Apache-2.0; see
LICENSE-APACHE-2.0 in this directory. This version adds the study's strict
image-geometry checks, CT common-field cropping/padding and intensity clipping,
RGBA conversion, and in-memory whole-image-mask extraction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from frd_score import frd


LUNA_GENERATED_SPACING = (0.699999988079071, 0.699999988079071, 1.25)
LUNA_TARGET_SIZE = (512, 512, 256)
LUNA_TARGET_ARRAY_SHAPE = LUNA_TARGET_SIZE[::-1]
IDENTITY_DIRECTION_2D = (1.0, 0.0, 0.0, 1.0)
IDENTITY_DIRECTION_3D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
GENERATED_LUNA_ARCHIVE_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


def feature_name_digest(names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def feature_value_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes(order="C")).hexdigest()


def get_extractor(image_dim: int):
    return frd.get_feature_extractor(frd_version="v1", image_dim=image_dim)


def center_crop_pad(
    array: np.ndarray, target: tuple[int, ...], pad_value: float
) -> np.ndarray:
    slices = []
    for size, wanted in zip(array.shape, target, strict=True):
        start = max(0, (size - wanted) // 2)
        slices.append(slice(start, start + min(size, wanted)))
    array = array[tuple(slices)]
    pads = []
    for size, wanted in zip(array.shape, target, strict=True):
        total = max(0, wanted - size)
        pads.append((total // 2, total - total // 2))
    return np.pad(array, pads, mode="constant", constant_values=pad_value)


def _close_tuple(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and bool(np.allclose(left, right, rtol=0.0, atol=1e-7))


def read_image(
    image_path: Path,
    *,
    dataset: str,
    generated: bool,
    color_handling: str = "grayscale",
) -> tuple[sitk.Image, str]:
    """Read an image as float32, repairing only known archive metadata/encoding issues."""
    if dataset == "ChestXray" and color_handling not in ("grayscale", "legacy-as-run"):
        raise ValueError("color_handling must be grayscale or legacy-as-run")
    if dataset == "ChestXray" and color_handling == "grayscale":
        image = sitk.ReadImage(str(image_path))
    else:
        image = sitk.ReadImage(str(image_path), outputPixelType=sitk.sitkFloat32)
    repair = "none"
    if dataset == "ChestXray":
        if image.GetDimension() != 2 or image.GetSize() != (1024, 1024):
            raise ValueError("ChestXray geometry mismatch: expected scalar 1024x1024 2D image")
        if not _close_tuple(image.GetSpacing(), (1.0, 1.0)):
            raise ValueError("ChestXray spacing mismatch: expected 1x1 archive units")
        if not _close_tuple(image.GetDirection(), IDENTITY_DIRECTION_2D):
            raise ValueError("ChestXray direction mismatch: expected identity")
        if color_handling == "grayscale":
            # Inspect native pixels before any scalar cast can alter color data.
            with Image.open(image_path) as pil_image:
                if (image.GetNumberOfComponentsPerPixel() != 1 or
                        pil_image.mode in ("P", "PA") or len(pil_image.getbands()) > 1):
                    array = np.asarray(pil_image.convert("L"), dtype=np.float32)
                    image = sitk.GetImageFromArray(array)
                    repair = "multichannel_to_l_pillow"
                else:
                    image = sitk.Cast(image, sitk.sitkFloat32)
        elif image.GetNumberOfComponentsPerPixel() != 1:
            # Preserve the original post-ReadImage component check verbatim.
            with Image.open(image_path) as pil_image:
                array = np.asarray(pil_image.convert("L"), dtype=np.float32)
            image = sitk.GetImageFromArray(array)
            repair = "rgba_to_l_pillow"
    if dataset == "Luna":
        if image.GetDimension() != 3:
            raise ValueError("Luna geometry mismatch: expected a 3D volume")
        if generated:
            if image.GetSize() != LUNA_TARGET_SIZE:
                raise ValueError("generated Luna geometry mismatch: expected 512x512x256")
            if not _close_tuple(image.GetSpacing(), (1.0, 1.0, 1.0)):
                raise ValueError("generated Luna archive spacing mismatch: expected 1x1x1 header")
            if not _close_tuple(image.GetDirection(), GENERATED_LUNA_ARCHIVE_DIRECTION):
                raise ValueError("generated Luna archive direction mismatch")
            repair = "luna_generated_common_fov_hu_clip_canonical_geometry"
        else:
            if not _close_tuple(image.GetSpacing(), LUNA_GENERATED_SPACING):
                raise ValueError("real Luna spacing mismatch: expected 0.7x0.7x1.25 mm")
            if not _close_tuple(image.GetDirection(), IDENTITY_DIRECTION_3D):
                raise ValueError("real Luna direction mismatch: expected identity")
            repair = "luna_real_common_fov_hu_clip_canonical_geometry"
        array = sitk.GetArrayFromImage(image)
        array = np.clip(array, -1000.0, 1000.0)
        array = center_crop_pad(array, LUNA_TARGET_ARRAY_SHAPE, pad_value=-1000.0)
        image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
        image.SetSpacing(LUNA_GENERATED_SPACING)
        if image.GetSize() != LUNA_TARGET_SIZE:
            raise RuntimeError("Luna common-FOV preprocessing failed to produce 512x512x256")
    return image, repair


def process_image(
    image_path: Path,
    *,
    dataset: str,
    generated: bool,
    feature_extractor,
    color_handling: str = "grayscale",
) -> tuple[np.ndarray, list[str], str]:
    """Mirror frd-score v1 process_image_mask_pair with in-memory repairs."""
    sitk_image, repair = read_image(image_path, dataset=dataset, generated=generated,
                                    color_handling=color_handling)

    mask_array = np.ones(sitk_image.GetSize()[::-1], dtype=np.uint8)
    if mask_array.ndim == 2:
        mask_array[0, 0] = 0
    elif mask_array.ndim >= 3:
        mask_array[0, 0, 0] = 0
    sitk_mask = sitk.GetImageFromArray(mask_array)
    sitk_mask.CopyInformation(sitk_image)

    image_array = sitk.GetArrayFromImage(sitk_image)
    if image_array.ndim == 2:
        original_spacing = sitk_image.GetSpacing()
        sitk_image_3d = sitk.GetImageFromArray(image_array[np.newaxis, ...])
        sitk_image_3d.SetSpacing((*original_spacing, 1.0))
        sitk_image = sitk_image_3d

        mask_array = sitk.GetArrayFromImage(sitk_mask)
        sitk_mask_3d = sitk.GetImageFromArray(mask_array[np.newaxis, ...])
        sitk_mask_3d.SetSpacing((*original_spacing, 1.0))
        sitk_mask = sitk_mask_3d

    output = feature_extractor.execute(sitk_image, sitk_mask)
    features: dict[str, float] = {}
    for feature_name in output:
        if "diagnostics" not in feature_name:
            features[feature_name.replace("original_", "")] = float(output[feature_name])
        else:
            try:
                features[feature_name] = float(output[feature_name])
            except (ValueError, TypeError):
                pass
    return np.asarray(list(features.values()), dtype=np.float64), list(features), repair
