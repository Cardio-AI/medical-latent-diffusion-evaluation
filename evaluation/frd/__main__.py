"""Compute FRDv1 from real and generated image manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EXPECTED_HASHES = {
    "frd": "d6c2a8c29ab43090fb4c61cae844dc4ebfafbfbd99dd42fcbf7d5c45fb47dcd9",
    "2d": "eb4235dbbdd719ccad6fde75f2b9e394812bc53890e74ba1631e6ee77a86dbb8",
    "3d": "42d714ef5d5d31d373b7b56531f268ed6859d32f29a0757253fd0c1d9ce4e738",
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(manifest: Path) -> list[Path]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            raise ValueError(f"{manifest}: CSV needs an image column")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"{manifest}: at least two images are required")
    paths = []
    for row in rows:
        path = Path(row["image"]).expanduser()
        path = path if path.is_absolute() else manifest.parent / path
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("chestxray", "luna"), required=True)
    parser.add_argument("--real-csv", type=Path, required=True, help="CSV containing real image paths")
    parser.add_argument("--generated-csv", type=Path, required=True, help="CSV containing generated image paths")
    parser.add_argument("--output", type=Path, required=True, help="new JSON result")
    parser.add_argument("--profile", choices=("paper", "upstream"), default="paper",
                        help="paper geometry/repairs, or unmodified upstream input handling")
    parser.add_argument("--threads", type=int, default=1, help="SimpleITK worker threads")
    parser.add_argument("--color-handling", choices=("grayscale", "legacy-as-run"),
                        help="paper X-ray color handling (default: grayscale)")
    parser.add_argument("--allow-other-frd-source", action="store_true", help="allow different upstream source/config bytes")
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error("output must be new and threads must be positive")
    if args.color_handling and (args.profile != "paper" or args.dataset != "chestxray"):
        parser.error("--color-handling applies only to the paper chestxray profile")
    color_handling = (args.color_handling or "grayscale") if args.profile == "paper" and args.dataset == "chestxray" else None
    real_paths, generated_paths = _paths(args.real_csv), _paths(args.generated_csv)
    try:
        import numpy as np
        import SimpleITK as sitk
        from frd_score import frd
        from . import core
        from .aggregation import compute_pair
    except ImportError as exc:
        parser.error(f"FRD dependency missing: {exc}. Install the optional FRD environment in evaluation/frd/README.md")
    observed = {"frd": _hash(Path(frd.__file__)), "2d": _hash(Path(frd.V1_CONFIG_2D)),
                "3d": _hash(Path(frd.V1_CONFIG_3D))}
    if observed != EXPECTED_HASHES and not args.allow_other_frd_source:
        parser.error("installed frd-score source/config hash differs; use the pinned optional environment")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    dimensionality = 2 if args.dataset == "chestxray" else 3
    extractor = core.get_extractor(dimensionality)
    names = None

    def extract(paths: list[Path], generated: bool):
        nonlocal names
        values = []
        repairs = Counter()
        for path in paths:
            if args.profile == "paper":
                features, current_names, repair = core.process_image(
                    path, dataset="ChestXray" if dimensionality == 2 else "Luna",
                    generated=generated, feature_extractor=extractor,
                    color_handling=color_handling or "grayscale",
                )
            else:
                image = sitk.ReadImage(str(path))
                if image.GetDimension() != dimensionality:
                    raise ValueError(f"{path}: wrong image dimensionality")
                result = frd.process_image_mask_pair((str(path), None), feature_extractor=extractor,
                                                      frd_version="v1")
                if result is None:
                    raise ValueError(f"FRD extraction skipped {path}")
                features, metadata, _ = result
                current_names = list(metadata)
                repair = "upstream_image_handling"
            if names is None:
                names = current_names
            elif current_names != names:
                raise ValueError(f"radiomic feature names or order differ for {path}")
            values.append(np.asarray(features, dtype=np.float64))
            repairs[repair] += 1
        return np.stack(values), dict(sorted(repairs.items()))

    real, real_repairs = extract(real_paths, False)
    generated, generated_repairs = extract(generated_paths, True)
    result = compute_pair(real, generated, names)
    result.update({"dataset": args.dataset, "preprocessing_profile": args.profile,
                   "color_handling": color_handling, "real_repair_counts": real_repairs,
                   "generated_repair_counts": generated_repairs,
                   "normalization": "real-reference z-score", "frd_version": "v1",
                   "frd_source_sha256": observed["frd"], "config_sha256": observed["2d" if dimensionality == 2 else "3d"]})
    for package in ("frd-score", "pyradiomics", "simpleitk"):
        try:
            result[package + "_version"] = version(package)
        except PackageNotFoundError:
            result[package + "_version"] = "unknown"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote FRD to {args.output}")


if __name__ == "__main__":
    main()
