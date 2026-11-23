#!/usr/bin/env python3
"""Deterministic corruption operators; arrays remain in physical evaluation units."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

from .common import resolve_data_path, stable_seed


@dataclass(frozen=True)
class Sample:
    data: np.ndarray
    dataset: str
    source_id: str
    recorded_path: str
    resolved_path: str
    spacing: tuple[float, ...]
    metadata: dict[str, Any]


def centre_crop_or_pad(array: np.ndarray, shape: Sequence[int], fill: float) -> np.ndarray:
    target = tuple(map(int, shape))
    output = np.full(target, fill, dtype=np.float32)
    input_slices = []
    output_slices = []
    for current, wanted in zip(array.shape, target):
        take = min(current, wanted)
        input_start = (current - take) // 2
        output_start = (wanted - take) // 2
        input_slices.append(slice(input_start, input_start + take))
        output_slices.append(slice(output_start, output_start + take))
    output[tuple(output_slices)] = np.asarray(array[tuple(input_slices)], dtype=np.float32)
    return output


def load_luna(recorded_path: str, source_id: str) -> Sample:
    path = resolve_data_path(recorded_path)
    image = nib.load(path)
    xyz = np.asarray(image.dataobj, dtype=np.float32)
    dhw = np.transpose(xyz, (2, 0, 1))
    dhw = np.clip(dhw, -1000.0, 1000.0)
    zoom_xyz = tuple(map(float, image.header.get_zooms()[:3]))
    return Sample(
        data=dhw,
        dataset="luna",
        source_id=source_id,
        recorded_path=recorded_path,
        resolved_path=str(path),
        spacing=(zoom_xyz[2], zoom_xyz[0], zoom_xyz[1]),
        metadata={
            "source_shape_xyz": list(image.shape[:3]),
            "source_spacing_xyz": list(zoom_xyz),
            "source_affine": np.asarray(image.affine).tolist(),
            "evaluation_affine": np.asarray(image.affine).tolist(),
            "evaluation_shape_dhw": list(dhw.shape),
        },
    )


def load_chestxray(
    recorded_path: str,
    source_id: str,
    original_geometry: tuple[float, float, float, float] | None = None,
) -> Sample:
    path = resolve_data_path(recorded_path)
    with Image.open(path) as image:
        raw = np.asarray(image)
        mode = image.mode
        size = image.size
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw = np.asarray(raw, dtype=np.float32)
    low = float(np.min(raw))
    high = float(np.max(raw))
    data = np.zeros_like(raw, dtype=np.float32) if high <= low else (raw - low) / (high - low)
    spacing_yx = (1.0, 1.0)
    geometry_metadata: dict[str, Any] = {"physical_spacing_available": False}
    if original_geometry is not None:
        original_width, original_height, spacing_x, spacing_y = map(float, original_geometry)
        if all(np.isfinite(value) and value > 0 for value in original_geometry):
            spacing_yx = (spacing_y * original_height / raw.shape[0], spacing_x * original_width / raw.shape[1])
            geometry_metadata = {
                "physical_spacing_available": True,
                "original_width": original_width,
                "original_height": original_height,
                "original_spacing_x_mm": spacing_x,
                "original_spacing_y_mm": spacing_y,
                "effective_spacing_y_mm": spacing_yx[0],
                "effective_spacing_x_mm": spacing_yx[1],
            }
    return Sample(
        data=np.clip(data, 0.0, 1.0),
        dataset="chestxray",
        source_id=source_id,
        recorded_path=recorded_path,
        resolved_path=str(path),
        spacing=spacing_yx,
        metadata={
            "source_shape_yx": list(raw.shape),
            "source_mode": mode,
            "source_size_xy": list(size),
            **geometry_metadata,
        },
    )


def valid_clip(data: np.ndarray, dataset: str) -> np.ndarray:
    bounds = (-1000.0, 1000.0) if dataset == "luna" else (0.0, 1.0)
    return np.clip(data, *bounds).astype(np.float32, copy=False)


def intensity_offset(data: np.ndarray, dataset: str, amount: float) -> np.ndarray:
    return valid_clip(np.asarray(data, dtype=np.float32) + float(amount), dataset)


def contrast(data: np.ndarray, dataset: str, factor: float, centre: float) -> np.ndarray:
    output = float(centre) + float(factor) * (np.asarray(data, dtype=np.float32) - float(centre))
    return valid_clip(output, dataset)


def window_narrow(data: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(np.asarray(data, dtype=np.float32), float(lower), float(upper)).astype(np.float32)


def gamma_distort(data: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(np.asarray(data, dtype=np.float32), 0.0, 1.0) ** float(gamma)


def gaussian_blur(data: np.ndarray, sigma_physical: float, spacing: Sequence[float]) -> np.ndarray:
    if float(sigma_physical) == 0:
        return np.asarray(data, dtype=np.float32).copy()
    sigma = tuple(float(sigma_physical) / float(value) for value in spacing)
    return ndimage.gaussian_filter(np.asarray(data, dtype=np.float32), sigma=sigma, mode="nearest").astype(np.float32)


def gaussian_noise(
    data: np.ndarray,
    dataset: str,
    std_normalised_range: float,
    seed_parts: Sequence[object],
) -> np.ndarray:
    if float(std_normalised_range) == 0:
        return np.asarray(data, dtype=np.float32).copy()
    physical_range = 2000.0 if dataset == "luna" else 1.0
    rng = np.random.default_rng(stable_seed(*seed_parts))
    noise = rng.normal(0.0, float(std_normalised_range) * physical_range, size=data.shape).astype(np.float32)
    return valid_clip(np.asarray(data, dtype=np.float32) + noise, dataset)


def attenuate_high_frequency(
    data: np.ndarray,
    retained_gain: float,
    low_pass_sigma_physical: float,
    spacing: Sequence[float],
) -> np.ndarray:
    """Fourier-equivalent Gaussian low-pass with controlled residual gain."""
    original = np.asarray(data, dtype=np.float32)
    if float(retained_gain) == 1.0:
        return original.copy()
    low = gaussian_blur(original, low_pass_sigma_physical, spacing)
    return (low + float(retained_gain) * (original - low)).astype(np.float32)


def checkerboard_patch(
    data: np.ndarray,
    dataset: str,
    amplitude_normalised_range: float,
    block_shape: Sequence[int],
) -> np.ndarray:
    source = np.asarray(data, dtype=np.float32)
    if float(amplitude_normalised_range) == 0:
        return source.copy()
    slices = tuple(slice(size * 3 // 8, size * 5 // 8) for size in source.shape)
    patch_shape = tuple(value.stop - value.start for value in slices)
    coordinates = np.indices(patch_shape, sparse=True)
    parity = np.zeros(patch_shape, dtype=np.int8)
    for coordinate, block in zip(coordinates, block_shape):
        parity ^= ((coordinate // int(block)) % 2).astype(np.int8)
    pattern = np.where(parity == 0, -1.0, 1.0).astype(np.float32)
    physical_range = 2000.0 if dataset == "luna" else 1.0
    output = source.copy()
    output[slices] += float(amplitude_normalised_range) * physical_range * pattern
    return valid_clip(output, dataset)


def translate_no_wrap(data: np.ndarray, offsets: Sequence[int], fill: float) -> np.ndarray:
    return ndimage.shift(
        np.asarray(data, dtype=np.float32),
        shift=tuple(map(int, offsets)),
        order=1,
        mode="constant",
        cval=float(fill),
        prefilter=False,
    ).astype(np.float32)


def near_copy(data: np.ndarray, dataset: str, source_id: str, seed: int) -> np.ndarray:
    if dataset == "luna":
        shifted = translate_no_wrap(data, (1, 1, 1), -1000.0)
        shifted = intensity_offset(shifted, dataset, 5.0)
        return gaussian_noise(shifted, dataset, 0.001, ("near-copy", source_id, seed))
    shifted = translate_no_wrap(data, (1, 1), 0.0)
    shifted = intensity_offset(shifted, dataset, 0.005)
    return gaussian_noise(shifted, dataset, 0.002, ("near-copy", source_id, seed))


def repeat_slices(data: np.ndarray, fraction: float, source_id: str, seed: int) -> np.ndarray:
    source = np.asarray(data, dtype=np.float32)
    count = int(round(float(fraction) * source.shape[0]))
    if count <= 0:
        return source.copy()
    candidates = np.arange(1, source.shape[0], dtype=np.int64)
    rng = np.random.default_rng(stable_seed("slice-repeat", source_id, seed))
    order = rng.permutation(candidates)
    selected = np.sort(order[: min(count, len(order))])
    output = source.copy()
    output[selected] = source[selected - 1]
    return output


def thoracic_elastic(data: np.ndarray, amplitude_pixels: float) -> np.ndarray:
    source = np.asarray(data, dtype=np.float32)
    if float(amplitude_pixels) == 0:
        return source.copy()
    height, width = source.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    ry, rx = 0.43 * height, 0.40 * width
    radius = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    taper = np.clip(1.0 - radius, 0.0, 1.0) ** 2
    dy = float(amplitude_pixels) * np.sin(2.0 * np.pi * (xx - cx) / width) * taper
    dx = float(amplitude_pixels) * np.sin(2.0 * np.pi * (yy - cy) / height) * taper
    return ndimage.map_coordinates(source, (yy - dy, xx - dx), order=1, mode="nearest").astype(np.float32)


def organ_hu_offset(data: np.ndarray, mask: np.ndarray, amount_hu: float) -> np.ndarray:
    output = np.asarray(data, dtype=np.float32).copy()
    present = np.asarray(mask, dtype=bool)
    output[present] += float(amount_hu)
    return valid_clip(output, "luna")


def organ_geometric_scale(
    data: np.ndarray,
    mask: np.ndarray,
    signed_scale_change: float,
    spacing: Sequence[float],
    blend_sigma_mm: float,
) -> np.ndarray:
    """Smooth local coordinate warp centred on one segmented structure."""
    source = np.asarray(data, dtype=np.float32)
    present = np.asarray(mask, dtype=bool)
    if float(signed_scale_change) == 0 or not np.any(present):
        return source.copy()
    points = np.argwhere(present)
    centre = points.mean(axis=0).astype(np.float32)
    sigma = tuple(float(blend_sigma_mm) / float(value) for value in spacing)
    influence = ndimage.gaussian_filter(present.astype(np.float32), sigma=sigma, mode="constant")
    maximum = float(influence.max())
    if maximum > 0:
        influence /= maximum
    influence = np.clip(influence, 0.0, 1.0)
    # Block along depth so coordinate fields never occupy several full-volume
    # copies in memory. The result is numerically identical to one full field.
    warped = np.empty_like(source, dtype=np.float32)
    depth_block = 16
    for start in range(0, source.shape[0], depth_block):
        stop = min(start + depth_block, source.shape[0])
        local_shape = (stop - start, source.shape[1], source.shape[2])
        coordinates = np.indices(local_shape, dtype=np.float32)
        coordinates[0] += start
        local_scale = 1.0 + float(signed_scale_change) * influence[start:stop]
        source_coordinates = np.stack(
            [
                centre[axis] + (coordinates[axis] - centre[axis]) / local_scale
                for axis in range(source.ndim)
            ],
            axis=0,
        )
        warped[start:stop] = ndimage.map_coordinates(
            source, source_coordinates, order=1, mode="nearest"
        )
        del coordinates, source_coordinates
    return valid_clip(warped, "luna")


def paired_simple_metrics(clean: np.ndarray, corrupted: np.ndarray, data_range: float) -> dict[str, float]:
    clean64 = np.asarray(clean, dtype=np.float64)
    corrupted64 = np.asarray(corrupted, dtype=np.float64)
    difference = corrupted64 - clean64
    mse = float(np.mean(difference * difference))
    denominator = float(np.mean(clean64 * clean64))
    return {
        "l1": float(np.mean(np.abs(difference))),
        "psnr": float("inf") if mse == 0 else float(10.0 * np.log10((float(data_range) ** 2) / mse)),
        "nmse": 0.0 if mse == 0 else (float("inf") if denominator == 0 else mse / denominator),
    }


def apply_condition(
    sample: Sample,
    condition: Any,
    *,
    transformation_seed: int = 61001,
    structure_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one registry condition; cohort-composition mapping is upstream."""
    subtype = condition.subtype
    parameters = condition.parameters
    if condition.condition_kind != "pixel_transform":
        return sample.data.copy()
    if subtype.startswith("hu_offset_") or subtype.startswith("brightness_"):
        amount = parameters.get("offset_hu", parameters.get("offset"))
        return intensity_offset(sample.data, sample.dataset, amount)
    if subtype.startswith("contrast_"):
        centre = parameters.get("centre_hu", parameters.get("centre"))
        return contrast(sample.data, sample.dataset, parameters["factor"], centre)
    if subtype == "window_narrowing":
        return window_narrow(sample.data, *parameters["clip_hu"])
    if subtype.startswith("gamma_"):
        return gamma_distort(sample.data, parameters["gamma"])
    if subtype == "gaussian_blur":
        return gaussian_blur(sample.data, parameters["sigma_mm"], sample.spacing)
    if subtype == "gaussian_noise":
        return gaussian_noise(
            sample.data,
            sample.dataset,
            parameters["std_normalised_range"],
            (sample.dataset, sample.source_id, subtype, transformation_seed),
        )
    if subtype == "high_frequency_attenuation":
        return attenuate_high_frequency(
            sample.data,
            parameters["retained_high_frequency_gain"],
            parameters["low_pass_sigma_mm"],
            sample.spacing,
        )
    if subtype == "checkerboard_patch":
        block = parameters.get("block_shape_dhw", parameters.get("block_shape_yx"))
        return checkerboard_patch(sample.data, sample.dataset, parameters["amplitude_normalised_range"], block)
    if subtype == "slice_repetition":
        return repeat_slices(
            sample.data, parameters["affected_slice_fraction"], sample.source_id, transformation_seed
        )
    if subtype == "thoracic_elastic_deformation":
        return thoracic_elastic(sample.data, parameters["displacement_pixels"])
    if "_geometric_" in subtype:
        if structure_mask is None:
            raise ValueError(f"{condition.condition_id} requires a structure mask")
        return organ_geometric_scale(
            sample.data,
            structure_mask,
            parameters["signed_scale_change"],
            sample.spacing,
            parameters["blend_sigma_mm"],
        )
    if subtype.endswith("_hu_positive") or subtype.endswith("_hu_negative"):
        if structure_mask is None:
            raise ValueError(f"{condition.condition_id} requires a structure mask")
        return organ_hu_offset(sample.data, structure_mask, parameters["offset_hu"])
    raise ValueError(f"No transform implementation for {condition.condition_id}")
