"""Distribution metric kernels (FID, KID, MMD, C2ST) over precomputed embeddings.

These are the production CPU-light distribution metrics used by the public
benchmark. The module deliberately stays self-contained: no image loading,
feature extraction, GPU libraries, or private artifact writes happen here. Inputs
are in-memory numeric embeddings shaped as rows of samples and columns of
features, produced upstream by the configured feature extractor.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from medgen_eval_lab.metrics import statistics as _stats
from medgen_eval_lab.privacy.redaction import contains_private_path, contains_secret
from medgen_eval_lab.schemas.results.metric_result import (
    ConfidenceInterval,
    MetricProvenance,
    MetricResult,
    MetricValues,
    SafetyFlags,
    SignificanceResult,
)
from medgen_eval_lab.schemas.taxonomy import MetricClaim

try:  # NumPy is an optional accelerator; the pure-Python path is the fallback.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only in numpy-free environments.
    _np = None  # type: ignore[assignment]

FID_CITATION = "https://arxiv.org/abs/1706.08500"
KID_CITATION = "https://arxiv.org/abs/1801.01401"
C2ST_CITATION = "https://arxiv.org/abs/1610.06545"
WAD_DIV_CITATION = "https://doi.org/10.1007/978-3-658-51100-5_83"
WAD_DIV_UPSTREAM = "https://github.com/Cardio-AI/wad-div"
BLUEPRINT_REF = "src/medgen_eval_lab/metrics/distribution/kernels.py"

KernelName = Literal["polynomial", "rbf", "linear"]
WADDivReference = Literal["zero", "exponential", "empirical"]
WADDivDistanceMetric = Literal["euclidean", "cosine"]
WADDivK = int | Literal["auto"]


@dataclass(frozen=True)
class DistributionMetricConfig:
    """Common CPU-light distribution metric configuration."""

    feature_backbone_id: str = "toy/precomputed-embeddings"
    feature_layer: str = "embedding"
    preprocessing_fingerprint: str = "none/toy"
    metric_config_hash: str | None = None
    evaluator_git_commit: str | None = None
    input_manifest_ids: tuple[str, ...] = ()
    bootstrap_repetitions: int = 0
    bootstrap_seed: int = 0
    confidence_level: float = 0.95
    sample_size_steps: tuple[int, ...] = ()
    permutation_repetitions: int | None = None
    c2st_folds: int = 5
    real_real_repetitions: int = 20
    sample_size_repetitions: int = 20

    def __post_init__(self) -> None:
        for label, value in {
            "feature_backbone_id": self.feature_backbone_id,
            "feature_layer": self.feature_layer,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
        }.items():
            _validate_config_required_safe_text(value, label=label)
        for label, value in {
            "metric_config_hash": self.metric_config_hash,
            "evaluator_git_commit": self.evaluator_git_commit,
        }.items():
            _validate_config_optional_safe_text(value, label=label)
        _validate_config_safe_text_tuple(
            self.input_manifest_ids,
            label="input_manifest_ids",
            item_label="input_manifest_id",
        )
        _validate_nonnegative_int(
            self.bootstrap_repetitions,
            label="bootstrap_repetitions",
        )
        _validate_int(self.bootstrap_seed, label="bootstrap_seed")
        _validate_confidence_level(self.confidence_level)
        _validate_positive_int_tuple(self.sample_size_steps, label="sample_size_steps")
        if self.permutation_repetitions is not None:
            _validate_nonnegative_int(
                self.permutation_repetitions,
                label="permutation_repetitions",
            )
        _validate_positive_int(self.c2st_folds, label="c2st_folds")
        if self.c2st_folds < 2:
            raise ValueError("c2st_folds must be at least 2")
        _validate_positive_int(self.real_real_repetitions, label="real_real_repetitions")
        _validate_positive_int(self.sample_size_repetitions, label="sample_size_repetitions")


@dataclass(frozen=True)
class WADDivMetricConfig:
    """WAD-DIV configuration over precomputed embedding rows."""

    k: WADDivK = "auto"
    reference: WADDivReference = "zero"
    normalize: bool = True
    distance_metric: WADDivDistanceMetric = "euclidean"
    exponential_reference_scale: float | None = None
    exponential_reference_count: int | None = None

    def __post_init__(self) -> None:
        _validate_wad_div_k(self.k, label="k")
        if self.reference not in {"zero", "exponential", "empirical"}:
            raise ValueError("reference must be one of: zero, exponential, empirical")
        if not isinstance(self.normalize, bool):
            raise ValueError("normalize must be a boolean")
        if self.distance_metric not in {"euclidean", "cosine"}:
            raise ValueError("distance_metric must be one of: euclidean, cosine")
        if self.exponential_reference_scale is not None:
            _validate_positive_float(
                self.exponential_reference_scale,
                label="exponential_reference_scale",
            )
        if self.exponential_reference_count is not None:
            _validate_positive_int(
                self.exponential_reference_count,
                label="exponential_reference_count",
            )


Matrix = list[list[float]]
Vector = list[float]

_FID_BACKEND_ENV = "MEDGEN_FID_BACKEND"
_FID_DEVICE_ENV = "MEDGEN_FID_DEVICE"
# Tile size (rows/cols) for the blocked NumPy MMD/KID Gram sums and the streaming
# FID covariance. Peak extra memory stays O(block^2) for kernels and O(block*dim)
# for covariance instead of O(N^2)/O(N*dim), so real-cohort sizes do not OOM.
# A block >= N reduces to a single tile, matching the unblocked result exactly.
_DISTRIBUTION_BLOCK_ENV = "MEDGEN_DISTRIBUTION_BLOCK_SIZE"
_DEFAULT_DISTRIBUTION_BLOCK = 2048


def _resolve_block_size(block_size: int | None) -> int:
    """Resolve the kernel/covariance tile size from an arg or env override."""

    if block_size is not None:
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        return block_size
    raw = os.environ.get(_DISTRIBUTION_BLOCK_ENV, "").strip()
    if not raw:
        return _DEFAULT_DISTRIBUTION_BLOCK
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"invalid {_DISTRIBUTION_BLOCK_ENV}={raw!r}; expected a positive integer"
        ) from exc
    if value <= 0:
        raise ValueError(f"{_DISTRIBUTION_BLOCK_ENV} must be a positive integer")
    return value


def _validate_kernel_parameters(
    *,
    kernel: object,
    unbiased: object,
    degree: object,
    gamma: object,
    coef0: object,
) -> None:
    if kernel not in {"linear", "polynomial", "rbf"}:
        raise ValueError("kernel must be one of: linear, polynomial, rbf")
    if not isinstance(unbiased, bool):
        raise ValueError("unbiased must be a boolean")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer")
    _positive_finite_float(gamma, name="gamma")
    _finite_float(coef0, name="coef0")


def _positive_finite_float(value: object, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _resolve_fid_backend() -> str:
    """Resolve the FID compute backend from ``MEDGEN_FID_BACKEND``.

    ``auto`` (default) uses NumPy when importable, else the pure-Python path.
    ``python`` forces the dependency-free path even when NumPy is installed.
    ``numpy`` requests NumPy (falling back to pure-Python when unavailable).
    ``torch`` requests the optional Torch/GPU path (device from ``MEDGEN_FID_DEVICE``).
    """

    raw = os.environ.get(_FID_BACKEND_ENV, "auto").strip().lower()
    if raw in ("", "auto"):
        return "numpy" if _np is not None else "python"
    if raw in ("python", "pure", "pure-python"):
        return "python"
    if raw == "numpy":
        return "numpy" if _np is not None else "python"
    if raw == "torch":
        return "torch"
    raise ValueError(
        f"unsupported {_FID_BACKEND_ENV}={raw!r}; expected one of auto|python|numpy|torch"
    )


def _use_numpy_kernels() -> bool:
    """Whether KID/MMD should use the NumPy Gram-matrix path when available."""

    raw = os.environ.get(_FID_BACKEND_ENV, "auto").strip().lower()
    if raw in ("python", "pure", "pure-python"):
        return False
    return _np is not None


def compute_fid(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
) -> float:
    """Compute FID between two small embedding matrices.

    The covariance square-root trace is computed via an eigen decomposition of
    ``sqrt(C_real) @ C_gen @ sqrt(C_real)``, avoiding SciPy while remaining
    appropriate for tiny toy matrices.
    """

    real = _as_matrix(real_embeddings, name="real_embeddings")
    generated = _as_matrix(generated_embeddings, name="generated_embeddings")
    _require_same_dim(real, generated)

    backend = _resolve_fid_backend()
    if backend == "torch":
        return _fid_from_matrices_torch(real, generated)
    if backend == "numpy":
        return _fid_from_matrices_numpy(real, generated)

    mu_real = _mean(real)
    mu_generated = _mean(generated)
    cov_real = _covariance(real)
    cov_generated = _covariance(generated)
    return _fid_from_moments(mu_real, cov_real, mu_generated, cov_generated)


def compute_fid_from_statistics(
    real_mean: Sequence[float],
    real_covariance: Sequence[Sequence[float]],
    generated_mean: Sequence[float],
    generated_covariance: Sequence[Sequence[float]],
) -> float:
    """Compute FID from precomputed embedding means and covariances."""

    parsed_real_mean = _as_vector(real_mean, name="real_mean")
    parsed_generated_mean = _as_vector(generated_mean, name="generated_mean")
    parsed_real_covariance = _as_square_matrix(real_covariance, name="real_covariance")
    parsed_generated_covariance = _as_square_matrix(
        generated_covariance,
        name="generated_covariance",
    )
    _require_same_moment_dim(
        parsed_real_mean,
        parsed_real_covariance,
        parsed_generated_mean,
        parsed_generated_covariance,
    )
    backend = _resolve_fid_backend()
    if backend == "torch":
        return _fid_from_moments_torch(
            parsed_real_mean,
            parsed_real_covariance,
            parsed_generated_mean,
            parsed_generated_covariance,
        )
    if backend == "numpy":
        return _fid_from_moments_numpy(
            parsed_real_mean,
            parsed_real_covariance,
            parsed_generated_mean,
            parsed_generated_covariance,
        )
    return _fid_from_moments(
        parsed_real_mean,
        parsed_real_covariance,
        parsed_generated_mean,
        parsed_generated_covariance,
    )


def compute_kid(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
    block_size: int | None = None,
) -> float:
    """Compute an unbiased KID/MMD² estimate with a polynomial kernel."""

    return compute_mmd(
        real_embeddings,
        generated_embeddings,
        kernel="polynomial",
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        unbiased=True,
        block_size=block_size,
    )


def compute_mmd(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    kernel: KernelName = "rbf",
    unbiased: bool = True,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
    block_size: int | None = None,
) -> float:
    """Compute MMD² between two embedding matrices.

    The NumPy path accumulates the Gram-matrix sums in ``block_size`` tiles, so
    peak memory stays O(block_size²) rather than O(N²) for real-cohort sizes; a
    tile at least as large as the input reduces to the single-Gram computation.
    """

    real = _as_matrix(real_embeddings, name="real_embeddings")
    generated = _as_matrix(generated_embeddings, name="generated_embeddings")
    _require_same_dim(real, generated)
    dim = len(real[0])
    resolved_gamma = gamma if gamma is not None else (1.0 / dim)
    _validate_kernel_parameters(
        kernel=kernel,
        unbiased=unbiased,
        degree=degree,
        gamma=resolved_gamma,
        coef0=coef0,
    )
    if unbiased and (len(real) < 2 or len(generated) < 2):
        raise ValueError("unbiased MMD/KID requires at least two samples per group")

    if _use_numpy_kernels():
        return _mmd_numpy(
            real,
            generated,
            kernel=kernel,
            unbiased=unbiased,
            degree=degree,
            gamma=resolved_gamma,
            coef0=coef0,
            block_size=block_size,
        )

    def k(left: Vector, right: Vector) -> float:
        return _kernel(left, right, kernel=kernel, degree=degree, gamma=resolved_gamma, coef0=coef0)

    if unbiased:
        xx = sum(k(real[i], real[j]) for i in range(len(real)) for j in range(len(real)) if i != j)
        yy = sum(
            k(generated[i], generated[j])
            for i in range(len(generated))
            for j in range(len(generated))
            if i != j
        )
        xy = sum(k(left, right) for left in real for right in generated)
        return (
            xx / (len(real) * (len(real) - 1))
            + yy / (len(generated) * (len(generated) - 1))
            - 2.0 * xy / (len(real) * len(generated))
        )

    xx = sum(k(left, right) for left in real for right in real)
    yy = sum(k(left, right) for left in generated for right in generated)
    xy = sum(k(left, right) for left in real for right in generated)
    return (
        xx / (len(real) ** 2) + yy / (len(generated) ** 2) - 2.0 * xy / (len(real) * len(generated))
    )


def compute_c2st(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    seed: int = 0,
    folds: int = 5,
) -> dict[str, Any]:
    """Compute seeded stratified out-of-fold nearest-centroid C2ST.

    Every prediction is made by a classifier trained without that prediction's
    fold.  Cohorts smaller than ``folds`` use one held-out sample per class per
    fold (the stratified leave-pair-out analogue); fewer than two samples in
    either class is explicitly insufficient.
    """

    real = sorted(_as_matrix(real_embeddings, name="real_embeddings"))
    generated = sorted(_as_matrix(generated_embeddings, name="generated_embeddings"))
    _require_same_dim(real, generated)
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("C2ST requires at least two samples per group")
    _validate_int(seed, label="seed")
    _validate_positive_int(folds, label="folds")
    if folds < 2:
        raise ValueError("folds must be at least 2")

    real_correct = 0.0
    generated_correct = 0.0
    tie_count = 0
    effective_folds = min(folds, len(real), len(generated))
    real_folds = _stratified_fold_indices(real, folds=effective_folds, seed=seed)
    generated_folds = _stratified_fold_indices(
        generated,
        folds=effective_folds,
        seed=seed,
    )
    for fold_index in range(effective_folds):
        held_real = set(real_folds[fold_index])
        held_generated = set(generated_folds[fold_index])
        train_real = [row for index, row in enumerate(real) if index not in held_real]
        train_generated = [
            row for index, row in enumerate(generated) if index not in held_generated
        ]
        real_centroid = _mean(train_real)
        generated_centroid = _mean(train_generated)
        for index in real_folds[fold_index]:
            prediction, credit = _c2st_classify(
                real[index], real_centroid, generated_centroid, true_label="real"
            )
            real_correct += credit
            tie_count += int(prediction == "tie")
        for index in generated_folds[fold_index]:
            prediction, credit = _c2st_classify(
                generated[index], real_centroid, generated_centroid, true_label="generated"
            )
            generated_correct += credit
            tie_count += int(prediction == "tie")

    real_accuracy = real_correct / len(real)
    generated_accuracy = generated_correct / len(generated)
    balanced_accuracy = 0.5 * (real_accuracy + generated_accuracy)
    accuracy = (real_correct + generated_correct) / (len(real) + len(generated))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "real_class_accuracy": real_accuracy,
        "generated_class_accuracy": generated_accuracy,
        "chance_accuracy": 0.5,
        "accuracy_gap_from_chance": abs(balanced_accuracy - 0.5),
        "tie_count": tie_count,
        "evaluation": "stratified_out_of_fold",
        "requested_folds": folds,
        "effective_folds": effective_folds,
        "fold_policy": (
            "stratified_leave_one_per_class_out"
            if effective_folds == len(real) == len(generated)
            else "seeded_stratified_k_fold"
        ),
        "seed": seed,
        "real_class_count": len(real),
        "generated_class_count": len(generated),
    }


def compute_wad_div(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    wad_config: WADDivMetricConfig | Mapping[str, Any] | None = None,
) -> dict[str, float | int | str | bool | None]:
    """Compute WAD-DIV over row-wise kNN distance distributions.

    The implementation follows the documented WAD-Div idea of comparing kNN
    distance distributions in feature space with a 1D Wasserstein distance. Raw
    feature rows and the distance distributions are intentionally not returned.
    """

    cfg = _parse_wad_div_config(wad_config)
    real = _as_matrix(real_embeddings, name="real_embeddings")
    generated = _as_matrix(generated_embeddings, name="generated_embeddings")
    _require_same_dim(real, generated)
    k = _resolve_wad_div_k(cfg.k, real_count=len(real), generated_count=len(generated))
    real_distances = _knn_distance_distribution(real, k=k, distance_metric=cfg.distance_metric)
    generated_distances = _knn_distance_distribution(
        generated,
        k=k,
        distance_metric=cfg.distance_metric,
    )
    reference = _wad_div_reference_distribution(
        cfg,
        target_count=len(generated_distances),
        empirical_distances=real_distances,
    )
    wad_div = _wasserstein_distance_1d(generated_distances, reference)

    # The real cohort anchor is the diversity magnitude of the real kNN-distance
    # distribution against zero. It makes the default zero-reference score a
    # real-normalized ratio and gives non-zero-reference modes a common scale.
    anchor_wad_div = _wasserstein_distance_1d(
        real_distances,
        [0.0 for _ in range(len(real_distances))],
    )
    normalized: float | None = None
    normalization_warning = ""
    if cfg.normalize:
        normalized, normalization_warning = _normalize_wad_div(
            wad_div,
            anchor_wad_div,
            reference=cfg.reference,
        )

    return {
        "wad_div": wad_div,
        "wad_div_norm": normalized,
        "anchor_wad_div": anchor_wad_div if cfg.normalize else None,
        "real_sample_count": len(real),
        "generated_sample_count": len(generated),
        "embedding_dim": len(real[0]),
        "k": k,
        "knn_distance_count": len(generated_distances),
        "real_knn_distance_count": len(real_distances),
        "reference": cfg.reference,
        "distance_metric": cfg.distance_metric,
        "normalization_enabled": cfg.normalize,
        "normalization_warning": normalization_warning,
        "exponential_reference_scale": (
            _wad_div_exponential_scale(cfg, empirical_distances=real_distances)
            if cfg.reference == "exponential"
            else None
        ),
        "exponential_reference_count": (
            len(reference) if cfg.reference == "exponential" else None
        ),
    }


def compute_wad_div_result(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    config: DistributionMetricConfig | Mapping[str, Any] | None = None,
    wad_config: WADDivMetricConfig | Mapping[str, Any] | None = None,
) -> MetricResult:
    """Return normalized ``MetricResult`` for WAD-DIV over embeddings."""

    cfg = _parse_config(config)
    wad_cfg = _parse_wad_div_config(wad_config)
    real = _as_matrix(real_embeddings, name="real_embeddings")
    generated = _as_matrix(generated_embeddings, name="generated_embeddings")
    _require_same_dim(real, generated)
    warnings = _embedding_warnings(real, generated, needs_covariance=False)
    values = compute_wad_div(real, generated, wad_config=wad_cfg)
    normalization_warning = values.get("normalization_warning")
    if isinstance(normalization_warning, str) and normalization_warning:
        warnings.append(normalization_warning)

    scalar_values: dict[str, float | int | bool] = {
        "wad_div": float(values["wad_div"]),
        "real_sample_count": int(values["real_sample_count"]),
        "generated_sample_count": int(values["generated_sample_count"]),
        "embedding_dim": int(values["embedding_dim"]),
        "k": int(values["k"]),
        "knn_distance_count": int(values["knn_distance_count"]),
    }
    if wad_cfg.normalize:
        scalar_values["wad_div_norm"] = float(values["wad_div_norm"])
        scalar_values["anchor_wad_div"] = float(values["anchor_wad_div"])

    status = "warning" if warnings else "ok"
    provenance = MetricProvenance(
        input_manifest_ids=list(cfg.input_manifest_ids),
        real_sample_count=len(real),
        generated_sample_count=len(generated),
        excluded_sample_count=0,
        preprocessing_fingerprint=cfg.preprocessing_fingerprint,
        feature_backbone_id=cfg.feature_backbone_id,
        feature_layer=cfg.feature_layer,
        metric_config_hash=cfg.metric_config_hash
        or _config_hash(
            cfg,
            {
                "wad_div": scalar_values["wad_div"],
                "wad_div_config": wad_cfg,
            },
        ),
        evaluator_git_commit=cfg.evaluator_git_commit,
        hardware_runtime_class="cpu-light",
        citation_urls=[WAD_DIV_CITATION],
        implementation_reference_urls=[BLUEPRINT_REF, WAD_DIV_UPSTREAM],
    )
    return MetricResult(
        metric_id="diversity/wad_div/precomputed_embeddings/0.1.0",
        metric_name="wad_div/precomputed_embeddings/knn_wasserstein_v1",
        metric_family="diversity",
        metric_version="0.1.0",
        status=status,
        status_reason="; ".join(warnings) if warnings else None,
        claim=MetricClaim(
            claim_axis="diversity_coverage",
            claim_kind="mode_collapse_screen",
            reference_kind="real_distribution",
            evidence_level="exploratory",
        ),
        values=MetricValues(
            scalar_values=scalar_values,
            extra_values={
                "distance": wad_cfg.distance_metric,
                "reference": wad_cfg.reference,
                "normalization": {
                    "enabled": wad_cfg.normalize,
                    "mode": (
                        "real_anchor_ratio"
                        if wad_cfg.reference == "zero"
                        else "real_anchor_dissimilarity_complement"
                    ),
                    "anchor_definition": "real cohort W1(real kNN distances, zero)",
                    "bounded_to_unit_interval": wad_cfg.normalize,
                },
                "exponential_reference": {
                    "scale": values["exponential_reference_scale"],
                    "count": values["exponential_reference_count"],
                    "quantiles": "deterministic_midpoint_quantiles",
                }
                if wad_cfg.reference == "exponential"
                else None,
                "input_type": "external precomputed embeddings",
                "artifact_policy": (
                    "feature rows, sample ids, kNN distance distributions, and reference "
                    "distributions remain outside the bundle"
                ),
                "method_note": (
                    "This is a non-official WAD-DIV-style screen computed over "
                    "row-wise kNN distance distributions in feature space using "
                    "a 1D Wasserstein distance."
                ),
                "implementation_status": "non_official_screen",
            },
        ),
        provenance=provenance,
        safety=SafetyFlags(
            contains_private_paths=False,
            contains_images=False,
            safe_for_git=True,
            safe_for_public_export=True,
            redaction_mode="aggregate-only",
        ),
        warnings=warnings,
    )


def compute_fid_result(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    config: DistributionMetricConfig | Mapping[str, Any] | None = None,
) -> MetricResult:
    """Return normalized ``MetricResult`` for FID over precomputed embeddings."""

    cfg = _parse_config(config)
    real = _as_matrix(real_embeddings, name="real_embeddings")
    generated = _as_matrix(generated_embeddings, name="generated_embeddings")
    _require_same_dim(real, generated)
    warnings = _embedding_warnings(real, generated, needs_covariance=True)
    value = compute_fid(real, generated)
    bootstrap_values = _bootstrap(real, generated, compute_fid, cfg)
    baseline_scalars, baseline_ci, baseline_extra = _real_real_baseline(
        real,
        compute_fid,
        value_name="real_real_fid",
        cfg=cfg,
    )
    scalar_values: dict[str, float | int | bool] = {
        "fid": value,
        **baseline_scalars,
        **_count_scalars(real, generated),
    }
    ci = {
        **_confidence_intervals("fid", bootstrap_values, cfg),
        **baseline_ci,
    }
    extra = {
        "covariance_diagnostics": _covariance_diagnostics(real, generated),
        "sample_size_curve": _sample_size_curve(real, generated, compute_fid, cfg),
        "real_real_baseline": baseline_extra,
        "significance_procedure": _permutation_metadata(real, generated, cfg=cfg),
        "feature_space_note": (
            "FID computed over caller-provided embeddings; no feature extraction performed."
        ),
    }
    return _metric_result(
        metric_id="distribution/fid/precomputed_embeddings/0.1.0",
        metric_name="fid/precomputed_embeddings/v1",
        metric_version="0.1.0",
        scalar_values=scalar_values,
        confidence_intervals=ci,
        extra_values=extra,
        warnings=warnings,
        config=cfg,
        real_count=len(real),
        generated_count=len(generated),
        citation_urls=[FID_CITATION],
        significance_tests=_baseline_significance(
            value, real, generated, compute_fid, value_name="fid", cfg=cfg
        ),
    )


def compute_fid_statistics_result(
    *,
    real_mean: Sequence[float],
    real_covariance: Sequence[Sequence[float]],
    generated_mean: Sequence[float],
    generated_covariance: Sequence[Sequence[float]],
    real_sample_count: int,
    generated_sample_count: int,
    config: DistributionMetricConfig | Mapping[str, Any] | None = None,
) -> MetricResult:
    """Return normalized ``MetricResult`` for FID over cached feature statistics."""

    cfg = _parse_config(config)
    value = compute_fid_from_statistics(
        real_mean,
        real_covariance,
        generated_mean,
        generated_covariance,
    )
    parsed_real_mean = _as_vector(real_mean, name="real_mean")
    embedding_dim = len(parsed_real_mean)
    real_count = _positive_count(real_sample_count, name="real_sample_count")
    generated_count = _positive_count(generated_sample_count, name="generated_sample_count")
    warnings = _embedding_count_warnings(
        real_count,
        generated_count,
        embedding_dim,
        needs_covariance=True,
    )
    if cfg.bootstrap_repetitions or cfg.sample_size_steps:
        warnings.append(
            "bootstrap and sample-size diagnostics are unavailable from aggregate statistics"
        )
    return _metric_result(
        metric_id="distribution/fid/precomputed_statistics/0.1.0",
        metric_name="fid/precomputed_feature_statistics/v1",
        metric_version="0.1.0",
        scalar_values={
            "fid": value,
            "real_sample_count": real_count,
            "generated_sample_count": generated_count,
            "embedding_dim": embedding_dim,
        },
        confidence_intervals={},
        extra_values={
            "covariance_diagnostics": _covariance_diagnostics_from_counts(
                real_count,
                generated_count,
                embedding_dim,
            ),
            "feature_space_note": (
                "FID computed over caller-provided aggregate feature statistics; "
                "feature rows were not loaded."
            ),
            "real_real_baseline": {
                "status": "unavailable",
                "reason": "real-real baseline requires feature rows, not only aggregate statistics",
            },
            "statistics_input": True,
        },
        warnings=warnings,
        config=cfg,
        real_count=real_count,
        generated_count=generated_count,
        citation_urls=[FID_CITATION],
    )


def compute_kid_result(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    config: DistributionMetricConfig | Mapping[str, Any] | None = None,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> MetricResult:
    """Return normalized ``MetricResult`` for unbiased polynomial KID."""

    cfg = _parse_config(config)
    real = _as_matrix(real_embeddings, name="real_embeddings")
    generated = _as_matrix(generated_embeddings, name="generated_embeddings")
    _require_same_dim(real, generated)
    warnings = _embedding_warnings(real, generated, needs_covariance=False)

    def metric(a: Matrix, b: Matrix) -> float:
        return compute_kid(a, b, degree=degree, gamma=gamma, coef0=coef0)

    value = metric(real, generated)
    bootstrap_values = _bootstrap(real, generated, metric, cfg)
    baseline_scalars, baseline_ci, baseline_extra = _real_real_baseline(
        real,
        metric,
        value_name="real_real_kid_mmd2_unbiased",
        cfg=cfg,
    )
    return _metric_result(
        metric_id="distribution/kid/precomputed_embeddings/0.1.0",
        metric_name="kid/precomputed_embeddings/polynomial_v1",
        metric_version="0.1.0",
        scalar_values={
            "kid_mmd2_unbiased": value,
            **baseline_scalars,
            **_count_scalars(real, generated),
        },
        confidence_intervals={
            **_confidence_intervals("kid_mmd2_unbiased", bootstrap_values, cfg),
            **baseline_ci,
        },
        extra_values={
            "kernel": "polynomial",
            "degree": degree,
            "gamma": gamma if gamma is not None else 1.0 / len(real[0]),
            "coef0": coef0,
            "estimator": "unbiased_mmd2",
            "real_real_baseline": baseline_extra,
            "sample_size_curve": _sample_size_curve(real, generated, metric, cfg),
            "significance_procedure": _permutation_metadata(real, generated, cfg=cfg),
        },
        warnings=warnings,
        config=cfg,
        real_count=len(real),
        generated_count=len(generated),
        citation_urls=[KID_CITATION],
        significance_tests=_baseline_significance(
            value, real, generated, metric, value_name="kid_mmd2_unbiased", cfg=cfg
        ),
    )


def compute_mmd_result(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    config: DistributionMetricConfig | Mapping[str, Any] | None = None,
    kernel: KernelName = "rbf",
    unbiased: bool = True,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> MetricResult:
    """Return normalized ``MetricResult`` for an MMD² estimate over embeddings."""

    cfg = _parse_config(config)
    real = _as_matrix(real_embeddings, name="real_embeddings")
    generated = _as_matrix(generated_embeddings, name="generated_embeddings")
    _require_same_dim(real, generated)
    warnings = _embedding_warnings(real, generated, needs_covariance=False)

    def metric(a: Matrix, b: Matrix) -> float:
        return compute_mmd(
            a,
            b,
            kernel=kernel,
            unbiased=unbiased,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
        )

    value = metric(real, generated)
    value_name = "mmd2_unbiased" if unbiased else "mmd2_biased"
    baseline_name = f"real_real_{value_name}"
    baseline_scalars, baseline_ci, baseline_extra = _real_real_baseline(
        real,
        metric,
        value_name=baseline_name,
        cfg=cfg,
    )
    return _metric_result(
        metric_id=f"distribution/mmd/{kernel}/precomputed_embeddings/0.1.0",
        metric_name=f"mmd/precomputed_embeddings/{kernel}_v1",
        metric_version="0.1.0",
        scalar_values={value_name: value, **baseline_scalars, **_count_scalars(real, generated)},
        confidence_intervals=_confidence_intervals(
            value_name,
            _bootstrap(real, generated, metric, cfg),
            cfg,
        )
        | baseline_ci,
        extra_values={
            "kernel": kernel,
            "degree": degree if kernel == "polynomial" else None,
            "gamma": gamma if gamma is not None else 1.0 / len(real[0]),
            "coef0": coef0 if kernel == "polynomial" else None,
            "estimator": "unbiased_mmd2" if unbiased else "biased_mmd2",
            "real_real_baseline": baseline_extra,
            "sample_size_curve": _sample_size_curve(real, generated, metric, cfg),
            "significance_procedure": _permutation_metadata(real, generated, cfg=cfg),
        },
        warnings=warnings,
        config=cfg,
        real_count=len(real),
        generated_count=len(generated),
        citation_urls=[KID_CITATION],
        significance_tests=_baseline_significance(
            value, real, generated, metric, value_name=value_name, cfg=cfg
        ),
    )


def compute_c2st_result(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    config: DistributionMetricConfig | Mapping[str, Any] | None = None,
) -> MetricResult:
    """Return normalized ``MetricResult`` for lightweight C2ST over embeddings."""

    cfg = _parse_config(config)
    real = _as_matrix(real_embeddings, name="real_embeddings")
    generated = _as_matrix(generated_embeddings, name="generated_embeddings")
    _require_same_dim(real, generated)
    warnings = _embedding_warnings(real, generated, needs_covariance=False)
    values = compute_c2st(
        real,
        generated,
        seed=cfg.bootstrap_seed,
        folds=cfg.c2st_folds,
    )

    def metric(a: Matrix, b: Matrix) -> float:
        return float(
            compute_c2st(
                a,
                b,
                seed=cfg.bootstrap_seed,
                folds=cfg.c2st_folds,
            )["balanced_accuracy"]
        )

    baseline_scalars, baseline_ci, baseline_extra = _real_real_c2st_baseline(real, cfg=cfg)

    return _metric_result(
        metric_id="distribution/c2st/precomputed_embeddings/0.1.0",
        metric_name="c2st/precomputed_embeddings/nearest_centroid_screen_v1",
        metric_version="0.1.0",
        scalar_values={
            "c2st_accuracy": float(values["accuracy"]),
            "c2st_balanced_accuracy": float(values["balanced_accuracy"]),
            "real_class_accuracy": float(values["real_class_accuracy"]),
            "generated_class_accuracy": float(values["generated_class_accuracy"]),
            "chance_accuracy": 0.5,
            "accuracy_gap_from_chance": float(values["accuracy_gap_from_chance"]),
            "tie_count": int(values["tie_count"]),
            **baseline_scalars,
            **_count_scalars(real, generated),
        },
        confidence_intervals={
            **_confidence_intervals(
                "c2st_balanced_accuracy",
                _bootstrap(real, generated, metric, cfg),
                cfg,
            ),
            **baseline_ci,
        },
        extra_values={
            "classifier": "nearest centroid",
            "headline_statistic": "c2st_balanced_accuracy",
            "implementation_status": "calibrated_out_of_fold_c2st",
            "method_note": (
                "Seeded stratified out-of-fold nearest-centroid classifier two-sample test."
            ),
            "tie_policy": "half credit",
            "public_payload_policy": "aggregate_only_no_row_level_c2st_records",
            "cross_validation": {
                "evaluation": values["evaluation"],
                "fold_policy": values["fold_policy"],
                "requested_folds": values["requested_folds"],
                "effective_folds": values["effective_folds"],
                "seed": values["seed"],
                "real_class_count": values["real_class_count"],
                "generated_class_count": values["generated_class_count"],
            },
            "significance_procedure": _permutation_metadata(real, generated, cfg=cfg)
            | {
                "requested_folds": values["requested_folds"],
                "effective_folds": values["effective_folds"],
                "fold_policy": values["fold_policy"],
            },
            "real_real_baseline": baseline_extra,
            "sample_size_curve": _sample_size_curve(real, generated, metric, cfg),
        },
        warnings=warnings,
        config=cfg,
        real_count=len(real),
        generated_count=len(generated),
        citation_urls=[C2ST_CITATION],
        significance_tests=_baseline_significance(
            float(values["balanced_accuracy"]),
            real,
            generated,
            metric,
            value_name="c2st_balanced_accuracy",
            cfg=cfg,
        ),
    )


def _parse_config(
    config: DistributionMetricConfig | Mapping[str, Any] | None,
) -> DistributionMetricConfig:
    if config is None:
        return DistributionMetricConfig()
    if isinstance(config, DistributionMetricConfig):
        return config
    data = dict(config)
    for key in ("input_manifest_ids", "sample_size_steps"):
        if key in data and not isinstance(data[key], tuple):
            if isinstance(data[key], str) or not isinstance(data[key], Sequence):
                raise ValueError(f"{key} must be an array")
            data[key] = tuple(data[key])
    return DistributionMetricConfig(**data)


def _parse_wad_div_config(
    config: WADDivMetricConfig | Mapping[str, Any] | None,
) -> WADDivMetricConfig:
    if config is None:
        return WADDivMetricConfig()
    if isinstance(config, WADDivMetricConfig):
        return config
    return WADDivMetricConfig(**dict(config))


def _validate_wad_div_k(value: object, *, label: str) -> None:
    if value == "auto":
        return
    _validate_positive_int(value, label=label)


def _validate_positive_int(value: object, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _validate_positive_float(value: object, *, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be a positive finite number")


def _resolve_wad_div_k(
    value: WADDivK,
    *,
    real_count: int,
    generated_count: int,
) -> int:
    required_sample_count = min(real_count, generated_count)
    if required_sample_count < 2:
        raise ValueError("WAD-DIV requires at least two real and two generated samples")
    max_supported_k = required_sample_count - 1
    if value == "auto":
        return min(3, max_supported_k)
    if value > max_supported_k:
        raise ValueError(
            "WAD-DIV k must be smaller than the real and generated sample counts; "
            f"received k={value} with min sample count {required_sample_count}"
        )
    return value


def _knn_distance_distribution(
    matrix: Matrix,
    *,
    k: int,
    distance_metric: WADDivDistanceMetric,
) -> list[float]:
    if len(matrix) <= k:
        raise ValueError("WAD-DIV k must be smaller than the sample count")
    distances: list[float] = []
    for left_index, left in enumerate(matrix):
        row_distances = [
            _embedding_distance(left, right, distance_metric=distance_metric)
            for right_index, right in enumerate(matrix)
            if right_index != left_index
        ]
        row_distances.sort()
        distances.extend(row_distances[:k])
    return distances


def _embedding_distance(
    left: Vector,
    right: Vector,
    *,
    distance_metric: WADDivDistanceMetric,
) -> float:
    if distance_metric == "euclidean":
        return math.sqrt(_squared_norm(_vector_sub(left, right)))
    if distance_metric == "cosine":
        left_norm = math.sqrt(_squared_norm(left))
        right_norm = math.sqrt(_squared_norm(right))
        if left_norm == 0.0 and right_norm == 0.0:
            return 0.0
        if left_norm == 0.0 or right_norm == 0.0:
            return 1.0
        similarity = _dot(left, right) / (left_norm * right_norm)
        return max(0.0, 1.0 - max(-1.0, min(1.0, similarity)))
    raise ValueError(f"unsupported WAD-DIV distance metric: {distance_metric}")


def _wad_div_reference_distribution(
    config: WADDivMetricConfig,
    *,
    target_count: int,
    empirical_distances: Sequence[float],
) -> list[float]:
    if config.reference == "zero":
        return [0.0 for _ in range(target_count)]
    if config.reference == "empirical":
        return [float(value) for value in empirical_distances]
    if config.reference == "exponential":
        count = config.exponential_reference_count or target_count
        scale = _wad_div_exponential_scale(config, empirical_distances=empirical_distances)
        return [
            -scale * math.log1p(-((index + 0.5) / count))
            for index in range(count)
        ]
    raise ValueError(f"unsupported WAD-DIV reference: {config.reference}")


def _wad_div_exponential_scale(
    config: WADDivMetricConfig,
    *,
    empirical_distances: Sequence[float],
) -> float:
    if config.exponential_reference_scale is not None:
        return float(config.exponential_reference_scale)
    if empirical_distances:
        mean_distance = sum(empirical_distances) / len(empirical_distances)
        if mean_distance > 0.0:
            return mean_distance
    return 1.0


def _wasserstein_distance_1d(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        raise ValueError("WAD-DIV Wasserstein distance requires non-empty distributions")
    left_sorted = sorted(float(value) for value in left)
    right_sorted = sorted(float(value) for value in right)
    left_count = len(left_sorted)
    right_count = len(right_sorted)
    left_index = 0
    right_index = 0
    left_cdf = 0.0
    right_cdf = 0.0
    previous: float | None = None
    distance = 0.0
    while left_index < left_count or right_index < right_count:
        next_left = (
            left_sorted[left_index] if left_index < left_count else math.inf
        )
        next_right = (
            right_sorted[right_index] if right_index < right_count else math.inf
        )
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


def _normalize_wad_div(
    wad_div: float,
    anchor_wad_div: float,
    *,
    reference: WADDivReference,
) -> tuple[float, str]:
    if anchor_wad_div <= 0.0:
        if wad_div <= 0.0:
            return 1.0, (
                "WAD-DIV normalization anchor is zero; zero WAD-DIV mapped to 1.0"
            )
        return 0.0, (
            "WAD-DIV normalization anchor is zero; non-zero WAD-DIV mapped to 0.0"
        )
    if reference == "zero":
        normalized = wad_div / anchor_wad_div
    else:
        normalized = 1.0 - (wad_div / anchor_wad_div)
    return max(0.0, min(1.0, float(normalized))), ""


def _validate_config_required_safe_text(value: object, *, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    _validate_config_safe_text(value, label=label)


def _validate_config_optional_safe_text(value: object, *, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    _validate_config_safe_text(value, label=label)


def _validate_config_safe_text(value: str, *, label: str) -> None:
    if contains_private_path(value):
        raise ValueError(f"{label} contains a private-looking path")
    if contains_secret(value):
        raise ValueError(f"{label} contains secret-looking text")


def _validate_config_safe_text_tuple(
    values: object,
    *,
    label: str,
    item_label: str,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{label} must be a tuple of strings")
    for index, value in enumerate(values):
        _validate_config_required_safe_text(value, label=f"{item_label}[{index}]")


def _validate_nonnegative_int(value: object, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _validate_int(value: object, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")


def _validate_confidence_level(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
    ):
        raise ValueError("confidence_level must be between 0 and 1")


def _validate_positive_int_tuple(values: object, *, label: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{label} must be a tuple of positive integers")
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label}[{index}] must be a positive integer")


def _as_matrix(values: Sequence[Sequence[float]], *, name: str) -> Matrix:
    if not isinstance(values, Sequence) or isinstance(values, str) or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of embedding rows")
    matrix: Matrix = []
    expected_dim: int | None = None
    for row_index, row in enumerate(values):
        if not isinstance(row, Sequence) or isinstance(row, str) or len(row) == 0:
            raise ValueError(f"{name}[{row_index}] must be a non-empty numeric row")
        parsed_row: Vector = []
        for col_index, value in enumerate(row):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name}[{row_index}][{col_index}] must be a finite number")
            parsed_row.append(float(value))
        expected_dim = len(parsed_row) if expected_dim is None else expected_dim
        if len(parsed_row) != expected_dim:
            raise ValueError(f"{name} rows must all have the same dimensionality")
        matrix.append(parsed_row)
    return matrix


def _as_vector(values: Sequence[float], *, name: str) -> Vector:
    if not isinstance(values, Sequence) or isinstance(values, str) or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty numeric vector")
    parsed: Vector = []
    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name}[{index}] must be a finite number")
        parsed.append(float(value))
    return parsed


def _as_square_matrix(values: Sequence[Sequence[float]], *, name: str) -> Matrix:
    matrix = _as_matrix(values, name=name)
    dimension = len(matrix)
    if any(len(row) != dimension for row in matrix):
        raise ValueError(f"{name} must be a square matrix")
    return matrix


def _require_same_dim(real: Matrix, generated: Matrix) -> None:
    if len(real[0]) != len(generated[0]):
        raise ValueError("real and generated embeddings must have the same dimensionality")


def _require_same_moment_dim(
    real_mean: Vector,
    real_covariance: Matrix,
    generated_mean: Vector,
    generated_covariance: Matrix,
) -> None:
    dimension = len(real_mean)
    if len(generated_mean) != dimension:
        raise ValueError("real and generated statistic means must have the same dimensionality")
    if len(real_covariance) != dimension or len(generated_covariance) != dimension:
        raise ValueError("statistic covariance dimensions must match mean dimensionality")
    if any(len(row) != dimension for row in (*real_covariance, *generated_covariance)):
        raise ValueError(
            "statistic covariance matrices must be square and match mean dimensionality"
        )


def _mean(matrix: Matrix) -> Vector:
    return [sum(row[col] for row in matrix) / len(matrix) for col in range(len(matrix[0]))]


def _covariance(matrix: Matrix) -> Matrix:
    dim = len(matrix[0])
    if len(matrix) < 2:
        return _zeros(dim, dim)
    mean = _mean(matrix)
    denom = len(matrix) - 1
    cov = _zeros(dim, dim)
    for row in matrix:
        centered = _vector_sub(row, mean)
        for i in range(dim):
            for j in range(dim):
                cov[i][j] += centered[i] * centered[j] / denom
    return cov


def _fid_from_moments(
    real_mean: Vector,
    real_covariance: Matrix,
    generated_mean: Vector,
    generated_covariance: Matrix,
) -> float:
    mean_term = _squared_norm(_vector_sub(real_mean, generated_mean))
    sqrt_cov_real = _symmetric_matrix_sqrt(real_covariance)
    sandwich = _matmul(_matmul(sqrt_cov_real, generated_covariance), sqrt_cov_real)
    covmean_trace = sum(math.sqrt(max(value, 0.0)) for value in _symmetric_eigenvalues(sandwich))
    fid = mean_term + _trace(real_covariance) + _trace(generated_covariance) - 2.0 * covmean_trace
    return max(0.0, float(fid))


def _fid_from_matrices_numpy(
    real: Sequence[Sequence[float]],
    generated: Sequence[Sequence[float]],
) -> float:
    """NumPy FID over embedding rows; equivalent to the pure-Python path."""

    real_array = _np.asarray(real, dtype=_np.float64)
    generated_array = _np.asarray(generated, dtype=_np.float64)
    block = _resolve_block_size(None)
    mu_real = real_array.mean(axis=0)
    mu_generated = generated_array.mean(axis=0)
    cov_real = _covariance_numpy(real_array, mu_real, block=block)
    cov_generated = _covariance_numpy(generated_array, mu_generated, block=block)
    return _fid_from_moments_numpy(mu_real, cov_real, mu_generated, cov_generated)


def _covariance_numpy(array: Any, mean: Any, *, block: int | None = None) -> Any:
    """Sample covariance accumulated in row blocks.

    For ``count <= block`` this is the plain ``centered.T @ centered`` form; for
    larger inputs the centered Gram is summed over row tiles so the full
    ``count×dim`` centered copy is never held, keeping extra memory O(block*dim).
    """

    count = int(array.shape[0])
    dim = int(array.shape[1])
    if count < 2:
        return _np.zeros((dim, dim), dtype=_np.float64)
    tile = _resolve_block_size(block) if block is None else block
    if count <= tile:
        centered = array - mean
        return (centered.T @ centered) / (count - 1)
    gram = _np.zeros((dim, dim), dtype=_np.float64)
    for start in range(0, count, tile):
        centered_tile = array[start : start + tile] - mean
        gram += centered_tile.T @ centered_tile
    return gram / (count - 1)


def _fid_from_moments_numpy(
    real_mean: Any,
    real_covariance: Any,
    generated_mean: Any,
    generated_covariance: Any,
) -> float:
    """FID via the stable PSD identity ``Tr((Cr·Cg)^½)=Σ√eig(Cr^½·Cg·Cr^½)``."""

    mu_real = _np.asarray(real_mean, dtype=_np.float64)
    mu_generated = _np.asarray(generated_mean, dtype=_np.float64)
    cov_real = _np.asarray(real_covariance, dtype=_np.float64)
    cov_generated = _np.asarray(generated_covariance, dtype=_np.float64)
    mean_term = float(_np.sum((mu_real - mu_generated) ** 2))
    sqrt_cov_real = _symmetric_matrix_sqrt_numpy(cov_real)
    sandwich = sqrt_cov_real @ cov_generated @ sqrt_cov_real
    sandwich = 0.5 * (sandwich + sandwich.T)
    eigenvalues = _np.linalg.eigvalsh(sandwich)
    covmean_trace = float(_np.sqrt(_np.clip(eigenvalues, 0.0, None)).sum())
    fid = (
        mean_term
        + float(_np.trace(cov_real))
        + float(_np.trace(cov_generated))
        - 2.0 * covmean_trace
    )
    return max(0.0, float(fid))


def _symmetric_matrix_sqrt_numpy(matrix: Any) -> Any:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = _np.linalg.eigh(symmetric)
    roots = _np.sqrt(_np.clip(eigenvalues, 0.0, None))
    return (eigenvectors * roots) @ eigenvectors.T


def _mmd_numpy(
    real: Sequence[Sequence[float]],
    generated: Sequence[Sequence[float]],
    *,
    kernel: KernelName,
    unbiased: bool,
    degree: int,
    gamma: float,
    coef0: float,
    block_size: int | None = None,
) -> float:
    """NumPy MMD²/KID via blocked Gram sums; equivalent to the pure-Python path.

    Only the scalar Gram sums (and the diagonal for the unbiased estimator) are
    needed, so the full N×N matrices are never materialized: each ``block``×``block``
    tile is summed and discarded. With ``block >= N`` this is exactly the
    single-Gram computation.
    """

    real_array = _np.asarray(real, dtype=_np.float64)
    generated_array = _np.asarray(generated, dtype=_np.float64)
    n = int(real_array.shape[0])
    m = int(generated_array.shape[0])
    block = _resolve_block_size(block_size)
    kernel_kwargs = {"kernel": kernel, "degree": degree, "gamma": gamma, "coef0": coef0}

    sum_xx, diag_xx = _blocked_gram_sum(
        real_array, real_array, symmetric=True, block=block, **kernel_kwargs
    )
    sum_yy, diag_yy = _blocked_gram_sum(
        generated_array, generated_array, symmetric=True, block=block, **kernel_kwargs
    )
    sum_xy, _ = _blocked_gram_sum(
        real_array, generated_array, symmetric=False, block=block, **kernel_kwargs
    )
    if unbiased:
        xx = (sum_xx - diag_xx) / (n * (n - 1))
        yy = (sum_yy - diag_yy) / (m * (m - 1))
        xy = sum_xy / (n * m)
        return xx + yy - 2.0 * xy
    xx = sum_xx / (n * n)
    yy = sum_yy / (m * m)
    xy = sum_xy / (n * m)
    return xx + yy - 2.0 * xy


def _blocked_gram_sum(
    left: Any,
    right: Any,
    *,
    kernel: KernelName,
    degree: int,
    gamma: float,
    coef0: float,
    block: int,
    symmetric: bool,
) -> tuple[float, float]:
    """Sum a kernel Gram matrix (and its diagonal) without materializing it.

    Returns ``(total, diagonal)`` where ``total`` is the sum over all entries and
    ``diagonal`` is the trace (only meaningful, and only computed, when
    ``symmetric`` is set, i.e. ``left is right``). Symmetric sums evaluate only the
    upper-triangular tiles and double the off-diagonal contribution.
    """

    n = int(left.shape[0])
    m = int(right.shape[0])
    total = 0.0
    diagonal = 0.0
    for i in range(0, n, block):
        left_tile = left[i : i + block]
        col_start = i if symmetric else 0
        for j in range(col_start, m, block):
            right_tile = right[j : j + block]
            tile = _kernel_gram_numpy(
                left_tile, right_tile, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0
            )
            tile_sum = float(tile.sum())
            if symmetric:
                if i == j:
                    total += tile_sum
                    diagonal += float(_np.trace(tile))
                else:
                    total += 2.0 * tile_sum
            else:
                total += tile_sum
    return total, diagonal


def _kernel_gram_numpy(
    left: Any,
    right: Any,
    *,
    kernel: KernelName,
    degree: int,
    gamma: float,
    coef0: float,
) -> Any:
    dot = left @ right.T
    if kernel == "linear":
        return dot
    if kernel == "polynomial":
        return (gamma * dot + coef0) ** degree
    if kernel == "rbf":
        left_sq = _np.sum(left * left, axis=1)[:, None]
        right_sq = _np.sum(right * right, axis=1)[None, :]
        squared_distance = left_sq + right_sq - 2.0 * dot
        _np.maximum(squared_distance, 0.0, out=squared_distance)
        return _np.exp(-gamma * squared_distance)
    raise ValueError(f"unsupported kernel: {kernel}")


def _fid_from_matrices_torch(
    real: Sequence[Sequence[float]],
    generated: Sequence[Sequence[float]],
) -> float:
    torch = _import_torch()
    device = _fid_torch_device()
    real_tensor = torch.tensor(real, dtype=torch.float64, device=device)
    generated_tensor = torch.tensor(generated, dtype=torch.float64, device=device)
    mu_real = real_tensor.mean(dim=0)
    mu_generated = generated_tensor.mean(dim=0)
    cov_real = _covariance_torch(torch, real_tensor, mu_real)
    cov_generated = _covariance_torch(torch, generated_tensor, mu_generated)
    return _fid_from_moments_torch_tensors(torch, mu_real, cov_real, mu_generated, cov_generated)


def _fid_from_moments_torch(
    real_mean: Sequence[float],
    real_covariance: Sequence[Sequence[float]],
    generated_mean: Sequence[float],
    generated_covariance: Sequence[Sequence[float]],
) -> float:
    torch = _import_torch()
    device = _fid_torch_device()
    mu_real = torch.tensor(real_mean, dtype=torch.float64, device=device)
    mu_generated = torch.tensor(generated_mean, dtype=torch.float64, device=device)
    cov_real = torch.tensor(real_covariance, dtype=torch.float64, device=device)
    cov_generated = torch.tensor(generated_covariance, dtype=torch.float64, device=device)
    return _fid_from_moments_torch_tensors(torch, mu_real, cov_real, mu_generated, cov_generated)


def _covariance_torch(torch: Any, tensor: Any, mean: Any) -> Any:
    count = int(tensor.shape[0])
    dim = int(tensor.shape[1])
    if count < 2:
        return torch.zeros((dim, dim), dtype=torch.float64, device=tensor.device)
    centered = tensor - mean
    return (centered.T @ centered) / (count - 1)


def _fid_from_moments_torch_tensors(
    torch: Any,
    mu_real: Any,
    cov_real: Any,
    mu_generated: Any,
    cov_generated: Any,
) -> float:
    mean_term = float(torch.sum((mu_real - mu_generated) ** 2).item())
    symmetric_real = 0.5 * (cov_real + cov_real.T)
    eigenvalues_real, eigenvectors_real = torch.linalg.eigh(symmetric_real)
    roots = torch.sqrt(torch.clamp(eigenvalues_real, min=0.0))
    sqrt_cov_real = (eigenvectors_real * roots) @ eigenvectors_real.T
    sandwich = sqrt_cov_real @ cov_generated @ sqrt_cov_real
    sandwich = 0.5 * (sandwich + sandwich.T)
    eigenvalues = torch.linalg.eigvalsh(sandwich)
    covmean_trace = float(torch.sqrt(torch.clamp(eigenvalues, min=0.0)).sum().item())
    fid = (
        mean_term
        + float(torch.trace(cov_real).item())
        + float(torch.trace(cov_generated).item())
        - 2.0 * covmean_trace
    )
    return max(0.0, float(fid))


def _import_torch() -> Any:
    import torch

    return torch


def _fid_torch_device() -> str:
    return os.environ.get(_FID_DEVICE_ENV, "cpu").strip() or "cpu"


def _kernel(
    left: Vector,
    right: Vector,
    *,
    kernel: KernelName,
    degree: int,
    gamma: float,
    coef0: float,
) -> float:
    dot = _dot(left, right)
    if kernel == "linear":
        return dot
    if kernel == "polynomial":
        return (gamma * dot + coef0) ** degree
    if kernel == "rbf":
        return math.exp(-gamma * _squared_norm(_vector_sub(left, right)))
    raise ValueError(f"unsupported kernel: {kernel}")


def _bootstrap(
    real: Matrix,
    generated: Matrix,
    metric_fn: Any,
    cfg: DistributionMetricConfig,
) -> list[float]:
    return _stats.bootstrap_statistics(
        [real, generated],
        metric_fn,
        repetitions=cfg.bootstrap_repetitions,
        seed=cfg.bootstrap_seed,
    )


def _confidence_intervals(
    value_name: str, values: list[float], cfg: DistributionMetricConfig
) -> dict[str, ConfidenceInterval]:
    interval = _stats.percentile_confidence_interval(values, level=cfg.confidence_level)
    if interval is None:
        return {}
    return {value_name: interval}


def _sample_size_curve(
    real: Matrix,
    generated: Matrix,
    metric_fn: Any,
    cfg: DistributionMetricConfig,
) -> list[dict[str, Any]]:
    """Repeated seeded subset curves with clamped-step deduplication and CIs."""

    grouped_requests: dict[int, list[int]] = {}
    for requested_size in cfg.sample_size_steps:
        size = min(int(requested_size), len(real), len(generated))
        if size <= 0:
            continue
        grouped_requests.setdefault(size, []).append(int(requested_size))

    curve: list[dict[str, Any]] = []
    for size, requested_sizes in grouped_requests.items():
        values: list[float] = []
        failure_reason: str | None = None
        sample_size_repetitions = int(getattr(cfg, "sample_size_repetitions", 20))
        for repetition in range(sample_size_repetitions):
            offset = 1_000_003 * size + repetition
            sampled_real = _stats.seeded_subset(
                real,
                size,
                seed=cfg.bootstrap_seed + 2 * offset,
            )
            sampled_generated = _stats.seeded_subset(
                generated,
                size,
                seed=cfg.bootstrap_seed + 2 * offset + 1,
            )
            try:
                value = float(metric_fn(sampled_real, sampled_generated))
            except ValueError as exc:
                failure_reason = str(exc)
                continue
            if math.isfinite(value):
                values.append(value)
        common: dict[str, Any] = {
            "sample_size": size,
            "effective_sample_size": size,
            "requested_sample_size": requested_sizes[0],
            "requested_sample_sizes": requested_sizes,
            "repetitions_requested": sample_size_repetitions,
            "repetitions_effective": len(values),
            "seed": cfg.bootstrap_seed,
            "selection_policy": "seeded_random_without_replacement_per_cohort",
        }
        if not values:
            curve.append(
                common
                | {
                    "status": "skipped",
                    "reason": failure_reason or "no finite values",
                }
            )
            continue
        interval = _stats.percentile_confidence_interval(
            values,
            level=cfg.confidence_level,
            method=f"repeated_subset_percentile/{len(values)}_repetitions",
        )
        entry = common | {
            "status": "ok",
            "value": sum(values) / len(values),
        }
        if interval is not None:
            entry.update(
                {
                    "confidence_level": interval.level,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                    "ci_method": interval.method,
                }
            )
        curve.append(entry)
    return curve


def _real_real_baseline(
    real: Matrix,
    metric_fn: Any,
    *,
    value_name: str,
    cfg: DistributionMetricConfig,
) -> tuple[dict[str, float | int | bool], dict[str, ConfidenceInterval], dict[str, Any]]:
    """Summarize repeated seeded, size-matched disjoint real-real comparisons."""

    metadata: dict[str, Any] = {
        "split_policy": "repeated_seeded_size_matched_disjoint",
        "metric_value_key": value_name,
        "calibration": "descriptive_only",
        "seed": cfg.bootstrap_seed,
        "repetitions_requested": cfg.real_real_repetitions,
    }
    splits = _real_real_splits(
        real,
        repetitions=cfg.real_real_repetitions,
        seed=cfg.bootstrap_seed,
    )
    if not splits:
        return (
            {},
            {},
            metadata
            | {
                "status": "skipped",
                "reason": "at least two real samples are required for a disjoint real-real split",
            },
        )
    left, right = splits[0]
    metadata.update(
        {
            "left_sample_count": len(left),
            "right_sample_count": len(right),
        }
    )
    values: list[float] = []
    failure_reason: str | None = None
    for split_left, split_right in splits:
        try:
            value = float(metric_fn(split_left, split_right))
        except ValueError as exc:
            failure_reason = str(exc)
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return {}, {}, metadata | {
            "status": "skipped",
            "reason": failure_reason or "no finite real-real comparisons",
        }
    value = sum(values) / len(values)

    scalars: dict[str, float | int | bool] = {
        value_name: value,
        "real_real_left_sample_count": len(left),
        "real_real_right_sample_count": len(right),
    }
    interval = _stats.percentile_confidence_interval(
        values,
        level=cfg.confidence_level,
        method=f"repeated_disjoint_real_real/{len(values)}_repetitions",
    )
    ci = {value_name: interval} if interval is not None else {}
    return scalars, ci, metadata | {
        "status": "ok",
        "value": value,
        "repetitions_effective": len(values),
    }


def _real_real_c2st_baseline(
    real: Matrix,
    *,
    cfg: DistributionMetricConfig,
) -> tuple[dict[str, float | int | bool], dict[str, ConfidenceInterval], dict[str, Any]]:
    def metric(a: Matrix, b: Matrix) -> float:
        return float(
            compute_c2st(a, b, seed=cfg.bootstrap_seed, folds=cfg.c2st_folds)[
                "balanced_accuracy"
            ]
        )

    scalars, ci, metadata = _real_real_baseline(
        real,
        metric,
        value_name="real_real_c2st_balanced_accuracy",
        cfg=cfg,
    )
    if "real_real_c2st_balanced_accuracy" in scalars:
        value = float(scalars["real_real_c2st_balanced_accuracy"])
        scalars["real_real_c2st_accuracy"] = value
        scalars["real_real_accuracy_gap_from_chance"] = abs(value - 0.5)
        metadata = metadata | {
            "balanced_accuracy": value,
            "accuracy_gap_from_chance": abs(value - 0.5),
        }
    return scalars, ci, metadata


def _real_real_splits(
    real: Matrix,
    *,
    repetitions: int,
    seed: int,
) -> list[tuple[Matrix, Matrix]]:
    group_size = len(real) // 2
    if group_size < 1:
        return []
    canonical = sorted(real)
    splits: list[tuple[Matrix, Matrix]] = []
    for repetition in range(repetitions):
        shuffled = list(canonical)
        random.Random(seed + repetition).shuffle(shuffled)
        splits.append(
            (
                shuffled[:group_size],
                shuffled[group_size : 2 * group_size],
            )
        )
    return splits


def _baseline_significance(
    observed: float,
    real: Matrix,
    generated: Matrix,
    metric_fn: Any,
    *,
    value_name: str,
    cfg: DistributionMetricConfig,
    alternative: str = "greater",
) -> dict[str, SignificanceResult]:
    """Pooled-label permutation test preserving the observed class counts."""

    del observed  # permutation_test recomputes it through the identical procedure.
    repetitions = _permutation_repetitions(cfg)
    if repetitions < _required_permutations(cfg):
        return {}
    result = _stats.permutation_test(
        real,
        generated,
        metric_fn,
        repetitions=repetitions,
        seed=cfg.bootstrap_seed,
        null_baseline="pooled_real_generated_label_permutation",
        alternative=alternative,
        method=f"pooled_label_permutation_full_procedure/{repetitions}_repetitions",
    )
    if result is None:
        return {}
    return {value_name: result}


def _permutation_repetitions(cfg: DistributionMetricConfig) -> int:
    if cfg.permutation_repetitions is not None:
        return cfg.permutation_repetitions
    return cfg.bootstrap_repetitions


def _required_permutations(cfg: DistributionMetricConfig) -> int:
    alpha = 1.0 - cfg.confidence_level
    return max(2, math.ceil((1.0 / alpha) - 1.0 - 1e-12))


def _permutation_metadata(
    real: Matrix,
    generated: Matrix,
    *,
    cfg: DistributionMetricConfig,
) -> dict[str, Any]:
    repetitions = _permutation_repetitions(cfg)
    required = _required_permutations(cfg)
    return {
        "status": "ok" if repetitions >= required else "insufficient_resolution",
        "procedure": "pooled_label_permutation_rerun_full_training_evaluation",
        "seed": cfg.bootstrap_seed,
        "repetitions": repetitions,
        "minimum_repetitions_for_confidence_level": required,
        "minimum_attainable_p_value": 1.0 / (repetitions + 1) if repetitions else 1.0,
        "confidence_level": cfg.confidence_level,
        "observed_real_class_count": len(real),
        "observed_generated_class_count": len(generated),
        "permuted_real_class_count": len(real),
        "permuted_generated_class_count": len(generated),
        "group_sizes_matched_between_observed_and_null": True,
    }


def _metric_result(
    *,
    metric_id: str,
    metric_name: str,
    metric_version: str,
    scalar_values: dict[str, float | int | bool],
    confidence_intervals: dict[str, ConfidenceInterval],
    extra_values: dict[str, Any],
    warnings: list[str],
    config: DistributionMetricConfig,
    real_count: int,
    generated_count: int,
    citation_urls: list[str],
    significance_tests: dict[str, SignificanceResult] | None = None,
) -> MetricResult:
    status = "warning" if warnings else "ok"
    provenance = MetricProvenance(
        input_manifest_ids=list(config.input_manifest_ids),
        real_sample_count=real_count,
        generated_sample_count=generated_count,
        excluded_sample_count=0,
        preprocessing_fingerprint=config.preprocessing_fingerprint,
        feature_backbone_id=config.feature_backbone_id,
        feature_layer=config.feature_layer,
        metric_config_hash=config.metric_config_hash or _config_hash(config, extra_values),
        evaluator_git_commit=config.evaluator_git_commit,
        hardware_runtime_class="cpu-light",
        citation_urls=citation_urls,
        implementation_reference_urls=[BLUEPRINT_REF],
    )
    return MetricResult(
        metric_id=metric_id,
        metric_name=metric_name,
        metric_family="distribution",
        metric_version=metric_version,
        status=status,
        status_reason="; ".join(warnings) if warnings else None,
        claim=MetricClaim(
            claim_axis="fidelity",
            claim_kind="distributional_fidelity",
            reference_kind="real_distribution",
            evidence_level="calibrated" if significance_tests else "exploratory",
        ),
        values=MetricValues(
            scalar_values=scalar_values,
            confidence_intervals=confidence_intervals,
            significance_tests=significance_tests or {},
            extra_values=extra_values,
        ),
        provenance=provenance,
        safety=SafetyFlags(
            contains_private_paths=False,
            contains_images=False,
            safe_for_git=True,
            safe_for_public_export=True,
            redaction_mode="aggregate-only",
        ),
        warnings=warnings,
    )


def _embedding_warnings(real: Matrix, generated: Matrix, *, needs_covariance: bool) -> list[str]:
    return _embedding_count_warnings(
        len(real),
        len(generated),
        len(real[0]),
        needs_covariance=needs_covariance,
    )


def _embedding_count_warnings(
    real_count: int,
    generated_count: int,
    embedding_dim: int,
    *,
    needs_covariance: bool,
) -> list[str]:
    warnings: list[str] = []
    if real_count < 2 or generated_count < 2:
        warnings.append("low-N: fewer than two samples in at least one group")
    if min(real_count, generated_count) < 10:
        warnings.append("low-N: toy sample size below 10; uncertainty estimates are underpowered")
    if needs_covariance and (real_count <= embedding_dim or generated_count <= embedding_dim):
        warnings.append(
            "covariance diagnostic: sample count <= embedding dimension; "
            "covariance is rank-deficient"
        )
    return warnings


def _covariance_diagnostics(real: Matrix, generated: Matrix) -> dict[str, int | bool]:
    return _covariance_diagnostics_from_counts(len(real), len(generated), len(real[0]))


def _covariance_diagnostics_from_counts(
    real_count: int,
    generated_count: int,
    embedding_dim: int,
) -> dict[str, int | bool]:
    return {
        "embedding_dim": embedding_dim,
        "real_rank_upper_bound": min(max(real_count - 1, 0), embedding_dim),
        "generated_rank_upper_bound": min(max(generated_count - 1, 0), embedding_dim),
        "real_covariance_rank_deficient": real_count <= embedding_dim,
        "generated_covariance_rank_deficient": generated_count <= embedding_dim,
    }


def _positive_count(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _count_scalars(real: Matrix, generated: Matrix) -> dict[str, int]:
    return {
        "real_sample_count": len(real),
        "generated_sample_count": len(generated),
        "embedding_dim": len(real[0]),
    }


def _config_hash(config: DistributionMetricConfig, extra_values: Mapping[str, Any]) -> str:
    payload = repr((config, sorted((key, repr(value)) for key, value in extra_values.items())))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _quantile(sorted_values: list[float], q: float) -> float:
    return _stats.quantile(sorted_values, q)


def _symmetric_matrix_sqrt(matrix: Matrix) -> Matrix:
    eigenvalues, eigenvectors = _jacobi_eigh(matrix)
    dim = len(matrix)
    result = _zeros(dim, dim)
    for k, eigenvalue in enumerate(eigenvalues):
        root = math.sqrt(max(eigenvalue, 0.0))
        for i in range(dim):
            for j in range(dim):
                result[i][j] += root * eigenvectors[i][k] * eigenvectors[j][k]
    return result


def _symmetric_eigenvalues(matrix: Matrix) -> list[float]:
    return _jacobi_eigh(_symmetrize(matrix))[0]


def _jacobi_eigh(
    matrix: Matrix,
    *,
    max_iter: int = 100,
    tolerance: float = 1e-12,
) -> tuple[list[float], Matrix]:
    n = len(matrix)
    a = [row[:] for row in _symmetrize(matrix)]
    v = _identity(n)
    for _ in range(max_iter * max(1, n * n)):
        p, q, offdiag = _largest_offdiag(a)
        if offdiag < tolerance:
            break
        if abs(a[p][p] - a[q][q]) < tolerance:
            angle = math.pi / 4.0
        else:
            angle = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        c = math.cos(angle)
        s = math.sin(angle)
        app = c * c * a[p][p] - 2.0 * s * c * a[p][q] + s * s * a[q][q]
        aqq = s * s * a[p][p] + 2.0 * s * c * a[p][q] + c * c * a[q][q]
        a[p][q] = 0.0
        a[q][p] = 0.0
        a[p][p] = app
        a[q][q] = aqq
        for r in range(n):
            if r not in (p, q):
                arp = c * a[r][p] - s * a[r][q]
                arq = s * a[r][p] + c * a[r][q]
                a[r][p] = a[p][r] = arp
                a[r][q] = a[q][r] = arq
            vrp = c * v[r][p] - s * v[r][q]
            vrq = s * v[r][p] + c * v[r][q]
            v[r][p] = vrp
            v[r][q] = vrq
    return [a[i][i] for i in range(n)], v


def _largest_offdiag(matrix: Matrix) -> tuple[int, int, float]:
    if len(matrix) < 2:
        return 0, 0, 0.0
    p, q = 0, 1
    largest = abs(matrix[p][q])
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            value = abs(matrix[i][j])
            if value > largest:
                p, q, largest = i, j, value
    return p, q, largest


def _symmetrize(matrix: Matrix) -> Matrix:
    n = len(matrix)
    return [[0.5 * (matrix[i][j] + matrix[j][i]) for j in range(n)] for i in range(n)]


def _zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _identity(n: int) -> Matrix:
    result = _zeros(n, n)
    for i in range(n):
        result[i][i] = 1.0
    return result


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    rows, inner, cols = len(left), len(right), len(right[0])
    return [
        [sum(left[i][k] * right[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]


def _trace(matrix: Matrix) -> float:
    return sum(matrix[i][i] for i in range(len(matrix)))


def _dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _vector_sub(left: Vector, right: Vector) -> Vector:
    return [a - b for a, b in zip(left, right, strict=True)]


def _squared_norm(vector: Iterable[float]) -> float:
    return sum(value * value for value in vector)


def _c2st_prediction(
    distance_to_real: float,
    distance_to_generated: float,
    *,
    true_label: str,
) -> tuple[str, float]:
    if math.isclose(distance_to_real, distance_to_generated, rel_tol=1e-12, abs_tol=1e-12):
        return "tie", 0.5
    prediction = "real" if distance_to_real < distance_to_generated else "generated"
    return prediction, 1.0 if prediction == true_label else 0.0


def _c2st_classify(
    row: Vector,
    real_centroid: Vector,
    generated_centroid: Vector,
    *,
    true_label: str,
) -> tuple[str, float]:
    return _c2st_prediction(
        _squared_norm(_vector_sub(row, real_centroid)),
        _squared_norm(_vector_sub(row, generated_centroid)),
        true_label=true_label,
    )


def _stratified_fold_indices(rows: Matrix, *, folds: int, seed: int) -> list[list[int]]:
    """Assign each class independently to balanced, seeded folds."""

    indices = list(range(len(rows)))
    # ``compute_c2st`` canonicalizes rows first, so this seed-driven shuffle is
    # independent of caller order while still varying fold membership by seed.
    random.Random(seed).shuffle(indices)
    assigned = [[] for _ in range(folds)]
    for position, index in enumerate(indices):
        assigned[position % folds].append(index)
    return assigned
