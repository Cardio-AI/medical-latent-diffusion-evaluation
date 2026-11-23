"""Apply one controlled benchmark image condition to a CSV of images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from .conditions import all_conditions
from .transforms import apply_condition, load_chestxray, load_luna, near_copy


def _path(value: str, manifest: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else manifest.parent / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list all condition IDs and exit")
    parser.add_argument("--dataset", choices=("chestxray", "luna"))
    parser.add_argument("--condition", help="full condition ID from --list, or near_copy for donor images")
    parser.add_argument("--manifest", type=Path, help="CSV with image, optional source_id and geometry")
    parser.add_argument("--output-dir", type=Path, help="new directory for arrays and output manifest")
    parser.add_argument("--seed", type=int, default=61001)
    parser.add_argument("--allow-unit-spacing", action="store_true",
                        help="explicitly assume 1 mm/pixel when radiography geometry is missing")
    args = parser.parse_args()
    registry = {condition.condition_id: condition for condition in all_conditions()}
    if args.list:
        for item in registry.values():
            print(f"{item.condition_id}\t{item.condition_kind}\t{item.parameters}")
        return
    if not all((args.dataset, args.condition, args.manifest, args.output_dir)):
        parser.error("--dataset, --condition, --manifest and --output-dir are required")
    is_near_copy = args.condition == "near_copy"
    condition = registry.get(args.condition)
    if not is_near_copy and (condition is None or condition.dataset != args.dataset):
        parser.error("unknown or dataset-mismatched condition; use --list")
    if not is_near_copy and condition.condition_kind != "pixel_transform":
        parser.error("cohort-composition conditions use the composition command")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            parser.error("input CSV must contain an image column")
        records = list(reader)
    if not records:
        parser.error("input CSV is empty")
    ids = [(row.get("source_id") or str(index)).strip() for index, row in enumerate(records)]
    if len(set(ids)) != len(ids) or not all(ids):
        parser.error("source_id values must be non-empty and unique")

    geometry_fields = ("original_width", "original_height", "original_spacing_x_mm", "original_spacing_y_mm")
    args.output_dir.mkdir(parents=True)
    output_rows = []
    for index, (row, source_id) in enumerate(zip(records, ids, strict=True)):
        image_path = _path(row["image"], args.manifest)
        if args.dataset == "luna":
            sample = load_luna(str(image_path), source_id)
        else:
            fields = [row.get(field, "").strip() for field in geometry_fields]
            if any(fields) and not all(fields):
                raise ValueError(f"row {index + 2}: supply all four original geometry fields")
            geometry = tuple(float(v) for v in fields) if all(fields) else None
            if geometry is None and not is_near_copy and condition.subtype in ("gaussian_blur", "high_frequency_attenuation") and not args.allow_unit_spacing:
                raise ValueError(f"row {index + 2}: geometry required for physical Gaussian scales; use --allow-unit-spacing to assume 1 mm/pixel")
            if geometry is not None and not all(np.isfinite(v) and v > 0 for v in geometry):
                raise ValueError(f"row {index + 2}: geometry must be finite and positive")
            sample = load_chestxray(str(image_path), source_id, geometry)
        mask = None
        if not is_near_copy and condition.requires_segmentation:
            if not row.get("mask"):
                raise ValueError(f"row {index + 2}: this condition needs a mask column")
            image = nib.load(str(_path(row["mask"], args.manifest)))
            mask = np.transpose(np.asarray(image.dataobj) > 0, (2, 0, 1))
            if mask.shape != sample.data.shape or not np.allclose(image.affine, sample.metadata["evaluation_affine"]):
                raise ValueError(f"row {index + 2}: mask geometry differs from source image")
        result = (near_copy(sample.data, args.dataset, source_id, args.seed) if is_near_copy else
                  apply_condition(sample, condition, transformation_seed=args.seed, structure_mask=mask))
        name = f"{index:06d}.npz"
        payload = {"data": np.asarray(result, dtype=np.float32), "spacing": np.asarray(sample.spacing),
                   "dataset": np.asarray(args.dataset), "source_id": np.asarray(source_id)}
        if args.dataset == "luna":
            payload["affine"] = np.asarray(sample.metadata["evaluation_affine"])
        np.savez_compressed(args.output_dir / name, **payload)
        output_rows.append({"image": name, "source_id": source_id})
    with (args.output_dir / "manifest.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "source_id"])
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} arrays to {args.output_dir}")


if __name__ == "__main__":
    main()
