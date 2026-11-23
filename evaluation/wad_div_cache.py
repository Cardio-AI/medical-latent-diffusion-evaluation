#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist


WAD_CACHE_VERSION = 1
DistanceDevice = Literal["cpu", "cuda"]


def embedding_digest(embeddings: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(embeddings, dtype=np.float32)
    return "sha256:" + hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def exact_knn_distance_distribution(
    embeddings: np.ndarray,
    *,
    k: int = 3,
    block_size: int = 128,
    device: DistanceDevice = "cpu",
) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) <= k:
        raise ValueError("WAD-DIV embeddings must be a 2D matrix with more than k rows")
    if k <= 0 or block_size <= 0:
        raise ValueError("k and block_size must be positive")
    if not np.isfinite(matrix).all():
        raise ValueError("WAD-DIV embeddings contain non-finite values")
    if device == "cuda":
        return _exact_knn_torch(matrix, k=k, block_size=block_size)
    if device != "cpu":
        raise ValueError(f"unsupported WAD-DIV distance device: {device}")

    result = np.empty((len(matrix), k), dtype=np.float64)
    for start in range(0, len(matrix), block_size):
        stop = min(start + block_size, len(matrix))
        distances = cdist(matrix[start:stop], matrix, metric="euclidean")
        local_rows = np.arange(stop - start)
        distances[local_rows, np.arange(start, stop)] = np.inf
        nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
        nearest.sort(axis=1)
        result[start:stop] = nearest
    return result.reshape(-1)


def _exact_knn_torch(matrix: np.ndarray, *, k: int, block_size: int) -> np.ndarray:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA WAD-DIV distances requested, but CUDA is unavailable")
    device = torch.device("cuda")
    reference = torch.from_numpy(matrix).to(device=device, dtype=torch.float64)
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(matrix), block_size):
            stop = min(start + block_size, len(matrix))
            distances = torch.cdist(reference[start:stop], reference, p=2)
            local_rows = torch.arange(stop - start, device=device)
            global_rows = torch.arange(start, stop, device=device)
            distances[local_rows, global_rows] = torch.inf
            nearest = torch.topk(distances, k=k, dim=1, largest=False, sorted=True).values
            rows.append(nearest.cpu().numpy())
    return np.concatenate(rows, axis=0).reshape(-1)


def load_or_compute_knn_distances(
    cache_path: Path,
    embeddings: np.ndarray,
    *,
    k: int = 3,
    block_size: int = 128,
    device: DistanceDevice = "cpu",
) -> tuple[np.ndarray, bool]:
    digest = embedding_digest(embeddings)
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if (
                    int(cached["cache_version"]) == WAD_CACHE_VERSION
                    and str(cached["embedding_digest"]) == digest
                    and int(cached["k"]) == k
                    and str(cached["distance_metric"]) == "euclidean"
                ):
                    distances = np.asarray(cached["distances"], dtype=np.float64)
                    if distances.shape == (len(embeddings) * k,) and np.isfinite(distances).all():
                        return distances, True
        except (KeyError, OSError, ValueError):
            pass

    distances = exact_knn_distance_distribution(
        embeddings, k=k, block_size=block_size, device=device
    )
    atomic_npz(
        cache_path,
        cache_version=np.asarray(WAD_CACHE_VERSION),
        embedding_digest=np.asarray(digest),
        k=np.asarray(k),
        distance_metric=np.asarray("euclidean"),
        distance_device=np.asarray(device),
        distances=distances,
    )
    return distances, False


def wad_div_variants_from_distances(
    real_distances: np.ndarray,
    generated_distances: np.ndarray,
    *,
    real_sample_count: int,
    generated_sample_count: int,
    embedding_dim: int,
    k: int = 3,
) -> tuple[dict[str, object], dict[str, object]]:
    real = np.asarray(real_distances, dtype=np.float64).tolist()
    generated = np.asarray(generated_distances, dtype=np.float64).tolist()
    anchor = _wasserstein_distance_1d(real, [0.0] * len(real))
    zero_distance = _wasserstein_distance_1d(generated, [0.0] * len(generated))
    empirical_distance = _wasserstein_distance_1d(generated, real)
    zero_normalized, zero_warning = _normalize(zero_distance, anchor, reference="zero")
    empirical_normalized, empirical_warning = _normalize(
        empirical_distance, anchor, reference="empirical"
    )

    common: dict[str, object] = {
        "anchor_wad_div": anchor,
        "real_sample_count": real_sample_count,
        "generated_sample_count": generated_sample_count,
        "embedding_dim": embedding_dim,
        "k": k,
        "knn_distance_count": len(generated),
        "real_knn_distance_count": len(real),
        "distance_metric": "euclidean",
        "normalization_enabled": True,
        "exponential_reference_scale": None,
        "exponential_reference_count": None,
    }
    zero = {
        **common,
        "wad_div": zero_distance,
        "wad_div_norm": zero_normalized,
        "reference": "zero",
        "normalization_warning": zero_warning,
    }
    empirical = {
        **common,
        "wad_div": empirical_distance,
        "wad_div_norm": empirical_normalized,
        "reference": "empirical",
        "normalization_warning": empirical_warning,
    }
    return zero, empirical


def _wasserstein_distance_1d(left: list[float], right: list[float]) -> float:
    if not left or not right:
        raise ValueError("WAD-DIV Wasserstein distance requires non-empty distributions")
    left_sorted = sorted(float(value) for value in left)
    right_sorted = sorted(float(value) for value in right)
    left_count, right_count = len(left_sorted), len(right_sorted)
    left_index = right_index = 0
    left_cdf = right_cdf = 0.0
    previous: float | None = None
    distance = 0.0
    while left_index < left_count or right_index < right_count:
        next_left = left_sorted[left_index] if left_index < left_count else math.inf
        next_right = right_sorted[right_index] if right_index < right_count else math.inf
        current = min(next_left, next_right)
        if previous is not None:
            distance += abs(left_cdf - right_cdf) * (current - previous)
        while left_index < left_count and left_sorted[left_index] == current:
            left_index += 1
        while right_index < right_count and right_sorted[right_index] == current:
            right_index += 1
        left_cdf = left_index / left_count
        right_cdf = right_index / right_count
        previous = current
    return float(distance)


def _normalize(
    wad_div: float, anchor: float, *, reference: Literal["zero", "empirical"]
) -> tuple[float, str]:
    if anchor <= 0.0:
        if wad_div <= 0.0:
            return 1.0, "WAD-DIV normalization anchor is zero; zero WAD-DIV mapped to 1.0"
        return 0.0, "WAD-DIV normalization anchor is zero; non-zero WAD-DIV mapped to 0.0"
    normalized = wad_div / anchor if reference == "zero" else 1.0 - (wad_div / anchor)
    return max(0.0, min(1.0, float(normalized))), ""
