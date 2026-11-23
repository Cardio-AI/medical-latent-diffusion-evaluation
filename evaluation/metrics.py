"""Distribution metrics for user-supplied real and generated embeddings."""

from __future__ import annotations

import argparse
import enum
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MEDGEN_FID_BACKEND", "numpy")
if not hasattr(enum, "StrEnum"):
    class _StrEnum(str, enum.Enum):
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

        def __str__(self):
            return str.__str__(self.value)

    enum.StrEnum = _StrEnum
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor" / "medgen_eval_lab"))

from medgen_eval_lab.metrics.distribution.kernels import (  # noqa: E402
    compute_c2st, compute_fid, compute_kid, compute_mmd,
)
from medgen_eval_lab.metrics.prdc.toy import compute_prdc_with_calibration  # noqa: E402
from .wad_div_cache import exact_knn_distance_distribution, wad_div_variants_from_distances  # noqa: E402


def distribution_stack(real: np.ndarray, generated: np.ndarray, *, c2st_seed: int = 0,
                       wad_device: str = "cpu", block_size: int = 128) -> dict:
    """Compute the retained distribution stack from feature rows [N, D]."""
    real = np.asarray(real, dtype=np.float32)
    generated = np.asarray(generated, dtype=np.float32)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("real and generated must be 2D arrays with equal feature dimensions")
    if min(len(real), len(generated)) < 5 or real.shape[1] < 1:
        raise ValueError("each embedding matrix needs at least five rows and one feature")
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("embeddings contain non-finite values")
    r, g = real.tolist(), generated.tolist()
    c2st = compute_c2st(r, g, seed=int(c2st_seed), folds=5)
    prdc = compute_prdc_with_calibration(r, g, k="auto", k_auto_threshold=0.95)
    rd = exact_knn_distance_distribution(real, k=3, block_size=block_size, device=wad_device)
    gd = exact_knn_distance_distribution(generated, k=3, block_size=block_size, device=wad_device)
    zero, empirical = wad_div_variants_from_distances(
        rd, gd, real_sample_count=len(real), generated_sample_count=len(generated),
        embedding_dim=real.shape[1], k=3,
    )
    return {
        "real_count": len(real), "generated_count": len(generated), "feature_dimension": real.shape[1],
        "fid": float(compute_fid(r, g)), "kid": float(compute_kid(r, g)),
        "mmd": float(compute_mmd(r, g)),
        "c2st_absolute_balanced_accuracy_gap": float(c2st["accuracy_gap_from_chance"]),
        "prdc": {name: float(getattr(prdc.values, name)) for name in ("precision", "recall", "density", "coverage")},
        "prdc_k": int(prdc.values.k),
        "prdc_calibration": prdc.k_calibration,
        "prdc_warnings": [*prdc.warnings, *prdc.values.warnings],
        "wad_div_zero": zero, "wad_div_empirical": empirical,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True, help=".npy matrix of real features, shape N x D")
    parser.add_argument("--generated", type=Path, required=True, help=".npy matrix of generated features, shape M x D")
    parser.add_argument("--output", type=Path, required=True, help="JSON result path")
    parser.add_argument("--seed", type=int, default=0, help="five-fold C2ST split seed")
    parser.add_argument("--wad-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = distribution_stack(np.load(args.real, allow_pickle=False),
                                np.load(args.generated, allow_pickle=False),
                                c2st_seed=args.seed, wad_device=args.wad_device,
                                block_size=args.block_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
