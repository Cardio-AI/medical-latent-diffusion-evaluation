"""Build one cohort's feature matrix from a benchmark composition mapping."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "source_id" not in reader.fieldnames:
            raise ValueError(f"{path}: missing source_id column")
        return list(reader)


def _bank(manifest: Path, features: Path) -> dict[str, np.ndarray]:
    rows, matrix = _rows(manifest), np.load(features, allow_pickle=False)
    ids = [row["source_id"] for row in rows]
    if matrix.ndim != 2 or len(matrix) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{manifest}: unique IDs and one feature row per ID are required")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{features}: non-finite features")
    return dict(zip(ids, matrix, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True, help="CSV produced by benchmark.composition")
    parser.add_argument("--source-manifest", type=Path, required=True, help="full source feature IDs, in feature row order")
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, help="donor feature IDs, in feature row order")
    parser.add_argument("--donor-features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix != ".npy":
        parser.error("output must be a new .npy file")
    if (args.donor_manifest is None) != (args.donor_features is None):
        parser.error("provide both donor manifest and donor features")
    source = _bank(args.source_manifest, args.source_features)
    donor = _bank(args.donor_manifest, args.donor_features) if args.donor_manifest else {}
    mapping = _rows(args.mapping)
    if not mapping:
        parser.error("mapping is empty")
    result = []
    for row in mapping:
        flag = row.get("true_injected", "0").strip()
        if flag not in ("0", "1"):
            raise ValueError("true_injected must be 0 or 1")
        bank = donor if flag == "1" else source
        if row["source_id"] not in bank:
            raise ValueError(f"source ID not present in selected feature bank: {row['source_id']}")
        result.append(bank[row["source_id"]])
    matrix = np.stack(result).astype(np.float32, copy=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.save(stream, matrix, allow_pickle=False)
    print(f"Wrote {matrix.shape} features to {args.output}")


if __name__ == "__main__":
    main()
