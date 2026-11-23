#!/usr/bin/env python3
"""Canonical condition registry shared by manifest, extraction, and analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Condition:
    dataset: str
    family: str
    subtype: str
    severity_rank: int
    severity_value: Any
    parameters: dict[str, Any]
    condition_kind: str = "pixel_transform"
    requires_segmentation: bool = False

    @property
    def condition_id(self) -> str:
        value = str(self.severity_value).replace(" ", "").replace("-", "m").replace(".", "p")
        value = value.replace("[", "").replace("]", "").replace(",", "_")
        return f"{self.dataset}__{self.family}__{self.subtype}__s{self.severity_rank}_{value}"

    @property
    def severity_normalised(self) -> float:
        return self.severity_rank / 4.0

    def row(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "dataset": self.dataset,
            "failure_family": self.family,
            "failure_subtype": self.subtype,
            "severity_rank": self.severity_rank,
            "severity_normalised": self.severity_normalised,
            "severity_value": json.dumps(self.severity_value, separators=(",", ":")),
            "transformation_parameters": json.dumps(self.parameters, sort_keys=True, separators=(",", ":")),
            "condition_kind": self.condition_kind,
            "requires_segmentation": self.requires_segmentation,
        }


def _add(
    rows: list[Condition],
    dataset: str,
    family: str,
    subtype: str,
    levels: Iterable[Any],
    parameter: str,
    *,
    kind: str = "pixel_transform",
    requires_segmentation: bool = False,
    fixed: dict[str, Any] | None = None,
) -> None:
    values = list(levels)
    denominator = max(1, len(values) - 1)
    for index, value in enumerate(values):
        rank = int(round(index * 4 / denominator))
        parameters = {parameter: value, **(fixed or {})}
        rows.append(
            Condition(
                dataset=dataset,
                family=family,
                subtype=subtype,
                severity_rank=rank,
                severity_value=value,
                parameters=parameters,
                condition_kind=kind,
                requires_segmentation=requires_segmentation,
            )
        )


def all_conditions() -> list[Condition]:
    rows: list[Condition] = []
    _add(rows, "luna", "global_intensity", "hu_offset_positive", [0, 25, 50, 100, 200], "offset_hu")
    _add(rows, "luna", "global_intensity", "hu_offset_negative", [0, -25, -50, -100, -200], "offset_hu")
    _add(rows, "luna", "global_intensity", "contrast_lower", [1.0, 0.9, 0.75], "factor", fixed={"centre_hu": 0.0})
    _add(rows, "luna", "global_intensity", "contrast_higher", [1.0, 1.1, 1.25], "factor", fixed={"centre_hu": 0.0})
    _add(rows, "luna", "global_intensity", "window_narrowing", [[-1000, 1000], [-900, 900], [-750, 750], [-500, 500], [-250, 250]], "clip_hu")
    _add(rows, "luna", "local_degradation", "gaussian_blur", [0.0, 0.5, 1.0, 2.0, 4.0], "sigma_mm")
    _add(rows, "luna", "local_degradation", "gaussian_noise", [0.0, 0.01, 0.025, 0.05, 0.1], "std_normalised_range")
    _add(rows, "luna", "local_degradation", "high_frequency_attenuation", [1.0, 0.75, 0.5, 0.25, 0.0], "retained_high_frequency_gain", fixed={"low_pass_sigma_mm": 1.0})
    _add(rows, "luna", "local_degradation", "checkerboard_patch", [0.0, 0.025, 0.05, 0.1, 0.2], "amplitude_normalised_range", fixed={"block_shape_dhw": [4, 16, 16]})
    _add(rows, "luna", "anatomical_volumetric_incoherence", "slice_repetition", [0.0, 0.05, 0.1, 0.2, 0.4], "affected_slice_fraction")
    for structure in ("left_lung", "right_lung", "heart", "aorta"):
        for direction, sign in (("contract", -1.0), ("expand", 1.0)):
            levels = [0.0, 0.05 * sign, 0.1 * sign, 0.2 * sign, 0.3 * sign]
            _add(rows, "luna", "anatomical_volumetric_incoherence", f"{structure}_geometric_{direction}", levels, "signed_scale_change", requires_segmentation=True, fixed={"structure": structure, "blend_sigma_mm": 2.0})
        for direction, sign in (("hu_positive", 1), ("hu_negative", -1)):
            levels = [0, 25 * sign, 50 * sign, 100 * sign, 200 * sign]
            _add(rows, "luna", "anatomical_volumetric_incoherence", f"{structure}_{direction}", levels, "offset_hu", requires_segmentation=True, fixed={"structure": structure})
    for subtype in ("common_stratum", "rare_stratum"):
        _add(rows, "luna", "support_loss", subtype, [0.0, 0.25, 0.5, 0.75, 1.0], "removed_fraction", kind="cohort_composition")
    _add(rows, "luna", "diversity_collapse", "exact_duplicate_collapse", [1.0, 0.9, 0.75, 0.5, 0.25], "unique_fraction", kind="cohort_composition")
    for subtype in ("exact_copy_injection", "near_copy_injection"):
        _add(rows, "luna", "memorisation_injection", subtype, [0.0, 0.01, 0.05, 0.1, 0.25, 0.5], "injected_fraction", kind="cohort_composition")

    _add(rows, "chestxray", "global_intensity", "brightness_positive", [0.0, 0.025, 0.05, 0.1, 0.2], "offset")
    _add(rows, "chestxray", "global_intensity", "brightness_negative", [0.0, -0.025, -0.05, -0.1, -0.2], "offset")
    _add(rows, "chestxray", "global_intensity", "gamma_lower", [1.0, 0.9, 0.75, 0.5], "gamma")
    _add(rows, "chestxray", "global_intensity", "gamma_higher", [1.0, 1.1, 1.25, 1.5], "gamma")
    _add(rows, "chestxray", "global_intensity", "contrast_lower", [1.0, 0.9, 0.75, 0.5, 0.25], "factor", fixed={"centre": 0.5})
    _add(rows, "chestxray", "global_intensity", "contrast_higher", [1.0, 1.1, 1.25, 1.5, 1.75], "factor", fixed={"centre": 0.5})
    _add(rows, "chestxray", "local_degradation", "gaussian_blur", [0.0, 0.5, 1.0, 2.0, 4.0], "sigma_mm")
    _add(rows, "chestxray", "local_degradation", "gaussian_noise", [0.0, 0.01, 0.025, 0.05, 0.1], "std_normalised_range")
    _add(rows, "chestxray", "local_degradation", "high_frequency_attenuation", [1.0, 0.75, 0.5, 0.25, 0.0], "retained_high_frequency_gain", fixed={"low_pass_sigma_mm": 2.0})
    _add(rows, "chestxray", "local_degradation", "checkerboard_patch", [0.0, 0.025, 0.05, 0.1, 0.2], "amplitude_normalised_range", fixed={"block_shape_yx": [16, 16]})
    _add(rows, "chestxray", "anatomical_volumetric_incoherence", "thoracic_elastic_deformation", [0.0, 4.0, 8.0, 16.0, 32.0], "displacement_pixels")
    for subtype in ("common_stratum", "rare_stratum"):
        _add(rows, "chestxray", "support_loss", subtype, [0.0, 0.25, 0.5, 0.75, 1.0], "removed_fraction", kind="cohort_composition")
    _add(rows, "chestxray", "diversity_collapse", "exact_duplicate_collapse", [1.0, 0.9, 0.75, 0.5, 0.25], "unique_fraction", kind="cohort_composition")
    for subtype in ("exact_copy_injection", "near_copy_injection"):
        _add(rows, "chestxray", "memorisation_injection", subtype, [0.0, 0.01, 0.05, 0.1, 0.25, 0.5], "injected_fraction", kind="cohort_composition")

    identifiers = [condition.condition_id for condition in rows]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
        raise RuntimeError(f"Duplicate condition IDs: {duplicates[:5]}")
    return rows


def conditions_for(dataset: str) -> list[Condition]:
    return [condition for condition in all_conditions() if condition.dataset == dataset]
