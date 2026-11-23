"""CPU-safe PRDC/diversity metrics over precomputed embeddings.

This module intentionally works on in-memory embedding arrays only. It does
not extract image features, build nearest-neighbor indexes, read private data, or
write artifacts.  The implementation follows the PRDC definitions from Naeem et
al. 2020 using exact Euclidean distances, which keeps the local smoke path small
and avoids FAISS or GPU dependencies.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Literal

from medgen_eval_lab.plugins.interface import (
    MetricPlugin,
    MetricPluginCapabilities,
    MetricPluginContext,
    MetricPluginError,
    MetricPluginIdentity,
    MetricPluginMetadata,
    parse_metric_plugin_context,
    require_local_toy_context,
)
from medgen_eval_lab.privacy.redaction import contains_private_path, contains_secret
from medgen_eval_lab.schemas.results.metric_result import (
    ExclusionReason,
    MetricProvenance,
    MetricResult,
    MetricValues,
    SafetyFlags,
)
from medgen_eval_lab.schemas.taxonomy import MetricClaim

try:  # NumPy is optional; use it when present but keep a pure-Python fallback.
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in minimal envs without numpy.
    np = None  # type: ignore[assignment]

PRDC_CITATION_URL = "https://arxiv.org/abs/2002.09797"
PRDC_REFERENCE_URL = "src/medgen_eval_lab/metrics/prdc/toy.py"
PRDC_PRECOMPUTED_METRIC_ID = "prdc/precomputed_embeddings/0.1.0"
PRDC_PRECOMPUTED_METRIC_NAME = "prdc/precomputed_embeddings/exact_pairwise_v1"

Role = Literal["real", "generated"]
PRDCK = int | Literal["auto"] | None


@dataclass(frozen=True)
class PRDCComputation:
    """PRDC values plus safe aggregate calibration metadata."""

    values: PRDCValues
    k_calibration: dict[str, Any]
    real_real_values: PRDCValues | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PRDCValues:
    """Raw scalar PRDC outputs before schema normalization."""

    precision: float
    recall: float
    density: float
    coverage: float
    k: int
    real_sample_count: int
    generated_sample_count: int
    embedding_dim: int
    duplicate_real_count: int
    duplicate_generated_count: int
    real_real_mean_distance: float
    generated_generated_mean_distance: float
    generated_real_mean_distance: float
    diversity_distance: float
    real_generated_diversity_gap: float
    mode_collapse_flag: bool
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for label, value in {
            "precision": self.precision,
            "recall": self.recall,
            "coverage": self.coverage,
        }.items():
            _ratio_value(value, label=label)
        _nonnegative_float(self.density, label="density")
        _positive_int(self.k, label="k")
        _positive_int(self.real_sample_count, label="real_sample_count")
        _positive_int(self.generated_sample_count, label="generated_sample_count")
        _positive_int(self.embedding_dim, label="embedding_dim")
        _bounded_count(
            self.duplicate_real_count,
            maximum=self.real_sample_count - 1,
            label="duplicate_real_count",
        )
        _bounded_count(
            self.duplicate_generated_count,
            maximum=self.generated_sample_count - 1,
            label="duplicate_generated_count",
        )
        for label, value in {
            "real_real_mean_distance": self.real_real_mean_distance,
            "generated_generated_mean_distance": self.generated_generated_mean_distance,
            "generated_real_mean_distance": self.generated_real_mean_distance,
            "diversity_distance": self.diversity_distance,
        }.items():
            _nonnegative_float(value, label=label)
        _finite_float(self.real_generated_diversity_gap, label="real_generated_diversity_gap")
        if not isinstance(self.mode_collapse_flag, bool):
            raise ValueError("mode_collapse_flag must be boolean")
        _safe_warning_tuple(self.warnings)


def compute_prdc(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    k: int = 5,
) -> PRDCValues:
    """Compute exact PRDC metrics over precomputed embeddings.

    Args:
        real_embeddings: Reference/real feature vectors, shape ``n_real x dim``.
        generated_embeddings: Generated feature vectors, shape ``n_generated x dim``.
        k: Nearest-neighbor radius rank. For robustness, values larger than
            the feasible rank are clamped to ``min(n_real, n_generated) - 1`` and
            reported via warnings.

    Returns:
        PRDCValues containing precision, recall, density, and coverage.

    Raises:
        ValueError: If embeddings are empty, ragged, non-numeric, non-finite, or
            cannot support at least one non-self nearest-neighbor radius.
    """

    real = _coerce_embeddings(real_embeddings, name="real_embeddings")
    generated = _coerce_embeddings(generated_embeddings, name="generated_embeddings")
    _validate_matching_dims(real, generated)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    real_count = len(real)
    generated_count = len(generated)
    max_k = min(real_count, generated_count) - 1
    if max_k < 1:
        raise ValueError("PRDC requires at least two real and two generated embeddings")

    warnings: list[str] = []
    effective_k = k
    if k > max_k:
        effective_k = max_k
        warnings.append(f"k={k} exceeds available sample support; clamped to k={effective_k}")

    real_radii = _nearest_neighbor_radii(real, effective_k)
    generated_radii = _nearest_neighbor_radii(generated, effective_k)
    real_to_generated = _pairwise_distances(real, generated)

    # Precision: fraction of generated samples inside at least one real manifold ball.
    precision_hits = 0
    density_sum = 0.0
    for generated_index in range(generated_count):
        inside_real = 0
        for real_index in range(real_count):
            if real_to_generated[real_index][generated_index] <= real_radii[real_index]:
                inside_real += 1
        if inside_real > 0:
            precision_hits += 1
        density_sum += inside_real / effective_k

    # Recall and coverage use generated manifold balls and nearest generated sample.
    recall_hits = 0
    coverage_hits = 0
    for real_index in range(real_count):
        nearest_generated = min(real_to_generated[real_index])
        if nearest_generated <= real_radii[real_index]:
            coverage_hits += 1
        for generated_index in range(generated_count):
            if real_to_generated[real_index][generated_index] <= generated_radii[generated_index]:
                recall_hits += 1
                break

    real_real_mean = _mean_nonself_pairwise_distance(real)
    generated_generated_mean = _mean_nonself_pairwise_distance(generated)
    generated_real_mean = _mean_flat(real_to_generated)
    diversity_gap = generated_generated_mean - real_real_mean
    duplicate_generated_count = _duplicate_count(generated)
    mode_collapse_flag = duplicate_generated_count > 0 or (
        real_real_mean > 0.0 and generated_generated_mean < 0.5 * real_real_mean
    )

    return PRDCValues(
        precision=precision_hits / generated_count,
        recall=recall_hits / real_count,
        density=density_sum / generated_count,
        coverage=coverage_hits / real_count,
        k=effective_k,
        real_sample_count=real_count,
        generated_sample_count=generated_count,
        embedding_dim=len(real[0]),
        duplicate_real_count=_duplicate_count(real),
        duplicate_generated_count=duplicate_generated_count,
        real_real_mean_distance=real_real_mean,
        generated_generated_mean_distance=generated_generated_mean,
        generated_real_mean_distance=generated_real_mean,
        diversity_distance=abs(diversity_gap),
        real_generated_diversity_gap=diversity_gap,
        mode_collapse_flag=mode_collapse_flag,
        warnings=tuple(warnings),
    )


def prdc_confidence_intervals(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    k: int,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Percentile bootstrap CIs for the PRDC headline scalars.

    Resamples real and generated rows with replacement and recomputes PRDC at the
    already-selected ``k``, yielding one CI per headline scalar (precision,
    recall, density, coverage) from a single resampling pass. Returns an empty
    mapping when bootstrap is disabled. Keyed by scalar name, matching the
    ``MetricValues.confidence_intervals`` schema.
    """

    from medgen_eval_lab.metrics import statistics as _stats

    if bootstrap_repetitions <= 0:
        return {}
    real = _coerce_embeddings(real_embeddings, name="real_embeddings")
    generated = _coerce_embeddings(generated_embeddings, name="generated_embeddings")

    def headline(a: list[list[float]], b: list[list[float]]) -> dict[str, float]:
        values = compute_prdc(a, b, k=k)
        return {
            "precision": values.precision,
            "recall": values.recall,
            "density": values.density,
            "coverage": values.coverage,
        }

    return _stats.bootstrap_confidence_intervals(
        [real, generated],
        headline,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
        level=confidence_level,
    )


def compute_prdc_with_calibration(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    k: PRDCK = "auto",
    k_auto_threshold: float = 0.95,
) -> PRDCComputation:
    """Compute PRDC with optional real-real calibrated k selection."""

    real = _coerce_embeddings(real_embeddings, name="real_embeddings")
    generated = _coerce_embeddings(generated_embeddings, name="generated_embeddings")
    _validate_matching_dims(real, generated)
    threshold = _validate_prdc_auto_threshold(k_auto_threshold)
    selected_k, calibration, real_real_values, selection_warnings = _resolve_prdc_k(
        real,
        generated,
        k=k,
        k_auto_threshold=threshold,
    )
    values = compute_prdc(real, generated, k=selected_k)
    calibration = dict(calibration)
    calibration["selected_k"] = values.k
    if calibration.get("method") == "manual":
        calibration["effective_k"] = values.k
    return PRDCComputation(
        values=values,
        k_calibration=calibration,
        real_real_values=real_real_values,
        warnings=selection_warnings,
    )


def prdc_metric_result(
    real_embeddings: Sequence[Sequence[float]],
    generated_embeddings: Sequence[Sequence[float]],
    *,
    k: PRDCK = "auto",
    k_auto_threshold: float = 0.95,
    context: MetricPluginContext | Mapping[str, Any] | None = None,
    feature_backbone_id: str = "precomputed-embedding/manual",
    feature_layer: str = "embedding",
    weights_hash: str | None = None,
) -> MetricResult:
    """Compute PRDC and emit a normalized ``MetricResult``."""

    parsed_context = _parse_context(context)
    if parsed_context.execution_mode != "local-toy":
        raise MetricPluginError(
            "PRDC in-memory plugin is only allowed in local-toy mode; "
            "private runs should use run-precomputed-embedding-metrics"
        )
    computation = compute_prdc_with_calibration(
        real_embeddings,
        generated_embeddings,
        k=k,
        k_auto_threshold=k_auto_threshold,
    )
    values = computation.values
    warnings = [*computation.warnings, *values.warnings]
    if values.duplicate_generated_count:
        warnings.append(
            f"generated embeddings contain {values.duplicate_generated_count} duplicate(s); "
            "possible mode collapse"
        )
    elif values.mode_collapse_flag:
        warnings.append(
            "generated diversity is less than half of real diversity; possible mode collapse"
        )
    if values.duplicate_real_count:
        warnings.append(f"real embeddings contain {values.duplicate_real_count} duplicate(s)")

    status = "warning" if warnings else "ok"
    status_reason = "; ".join(warnings) if warnings else None
    metric_values = MetricValues(
        scalar_values={
            "precision": values.precision,
            "recall": values.recall,
            "density": values.density,
            "coverage": values.coverage,
            "k": values.k,
            "real_sample_count": values.real_sample_count,
            "generated_sample_count": values.generated_sample_count,
            "embedding_dim": values.embedding_dim,
            "duplicate_real_count": values.duplicate_real_count,
            "duplicate_generated_count": values.duplicate_generated_count,
            "real_real_mean_distance": values.real_real_mean_distance,
            "generated_generated_mean_distance": values.generated_generated_mean_distance,
            "generated_real_mean_distance": values.generated_real_mean_distance,
            "diversity_distance": values.diversity_distance,
            "real_generated_diversity_gap": values.real_generated_diversity_gap,
            "mode_collapse_flag": values.mode_collapse_flag,
            **_real_real_scalar_values(computation.real_real_values),
        },
        extra_values={
            "distance": "euclidean",
            "input_type": "precomputed embeddings",
            "kernel": "exact pairwise distances; cpu-light",
            "k_calibration": computation.k_calibration,
            "k_sensitivity": _k_sensitivity_table(real_embeddings, generated_embeddings),
            "wad_div_style": {
                "definition": "aggregate complement using mean pairwise embedding distances",
                "diversity_distance": (
                    "abs(generated_generated_mean_distance - real_real_mean_distance)"
                ),
                "real_generated_diversity_gap": (
                    "generated_generated_mean_distance - real_real_mean_distance"
                ),
                "mode_collapse_flag": (
                    "duplicates or generated pairwise mean < half real pairwise mean"
                ),
                "caveat": (
                    "medical diversity complement only; not a replacement for "
                    "PRDC coverage or utility; not the Cardio-AI WAD-DIV metric"
                ),
            },
        },
    )
    provenance = MetricProvenance(
        input_manifest_ids=parsed_context.input_manifest_ids,
        real_sample_count=values.real_sample_count,
        generated_sample_count=values.generated_sample_count,
        excluded_sample_count=0,
        preprocessing_fingerprint=parsed_context.preprocessing_fingerprint,
        feature_backbone_id=feature_backbone_id,
        feature_layer=feature_layer,
        weights_hash=weights_hash,
        cache_keys=[],
        metric_config_hash=parsed_context.metric_config_hash,
        evaluator_git_commit=parsed_context.evaluator_git_commit,
        hardware_runtime_class="cpu-light",
        citation_urls=[PRDC_CITATION_URL],
        implementation_reference_urls=[PRDC_REFERENCE_URL],
    )
    return MetricResult(
        metric_id=PRDC_PRECOMPUTED_METRIC_ID,
        metric_name=PRDC_PRECOMPUTED_METRIC_NAME,
        metric_family="prdc",
        metric_version="0.1.0",
        status=status,
        status_reason=status_reason,
        claim=MetricClaim(
            claim_axis="diversity_coverage",
            claim_kind="coverage_diversity",
            reference_kind="real_distribution",
            evidence_level="exploratory",
        ),
        values=metric_values,
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


class PRDCMetricPlugin(MetricPlugin):
    """Stateful local smoke PRDC plugin accepting real/generated embedding cases.

    Accepted cases are metadata-only mappings:
    ``{"case_id": "r1", "role": "real", "embedding": [0.0, 1.0]}`` or
    ``{"case_id": "g1", "role": "generated", "embedding": [0.1, 1.0]}``.
    Cases may set ``exclude=True`` with an ``exclude_reason``; excluded cases are
    counted in provenance and not used for PRDC.
    """

    metadata = MetricPluginMetadata(
        identity=MetricPluginIdentity(
            metric_name=PRDC_PRECOMPUTED_METRIC_NAME,
            metric_family="prdc",
            metric_version="0.1.0",
            citation_urls=[PRDC_CITATION_URL],
            implementation_reference_urls=[PRDC_REFERENCE_URL],
            owner_or_reviewer="medgen-eval-lab Wave 2 packet 2",
        ),
        capabilities=MetricPluginCapabilities(
            supported_dimensionality=["2d", "3d", "2d+t", "3d+t/4d"],
            supported_modalities=["CMR", "MRI", "CT", "CTA", "X-ray", "other"],
            supports_streaming_update=True,
            supports_distributed=False,
            requires_gpu=False,
            runtime_class="cpu-light",
            paired_or_unpaired="unpaired cohort",
            required_inputs=["feature embeddings"],
        ),
    )

    def __init__(self, *, k: PRDCK = "auto", k_auto_threshold: float = 0.95) -> None:
        self.k: int | Literal["auto"] = _validate_prdc_k_mode(k, error_cls=MetricPluginError)
        self.k_auto_threshold = _validate_prdc_auto_threshold(
            k_auto_threshold,
            error_cls=MetricPluginError,
        )
        self.context: MetricPluginContext | None = None
        self._prepared = False
        self._real_embeddings: list[list[float]] = []
        self._generated_embeddings: list[list[float]] = []
        self._case_ids: set[str] = set()
        self._real_seen = 0
        self._generated_seen = 0
        self._excluded = 0
        self._exclusion_reasons: dict[str, int] = {}

    def prepare(self, context: MetricPluginContext | Mapping[str, Any] | None = None) -> None:
        parsed_context = require_local_toy_context(context, plugin_name=self.__class__.__name__)
        self.context = parsed_context
        self._prepared = True

    def validate_inputs(self) -> None:
        if not self._prepared or self.context is None:
            raise MetricPluginError("plugin must be prepared before validate_inputs/update/compute")
        if self.metadata.capabilities.requires_gpu:
            raise MetricPluginError("PRDC precomputed embedding metric must not require GPU")
        if self.metadata.capabilities.runtime_class != "cpu-light":
            raise MetricPluginError("PRDC precomputed embedding metric must be cpu-light")
        if self.context.execution_mode != "local-toy":
            raise MetricPluginError(
                "PRDC in-memory plugin received unsafe execution mode; "
                "private runs should use run-precomputed-embedding-metrics"
            )

    def update(self, batch_or_case: Mapping[str, Any] | list[Mapping[str, Any]]) -> None:
        self.validate_inputs()
        if isinstance(batch_or_case, Mapping):
            cases = [batch_or_case]
        elif isinstance(batch_or_case, list):
            cases = batch_or_case
        else:
            raise MetricPluginError("update expects a case mapping or list of case mappings")
        for case in cases:
            self._update_one(case)

    def compute(self) -> MetricResult:
        self.validate_inputs()
        if len(self._real_embeddings) < 2 or len(self._generated_embeddings) < 2:
            return self._insufficient_data_result()
        result = prdc_metric_result(
            self._real_embeddings,
            self._generated_embeddings,
            k=self.k,
            k_auto_threshold=self.k_auto_threshold,
            context=self.context,
        )
        scalar_values = {
            **result.values.scalar_values,
            "accepted_real_sample_count": len(self._real_embeddings),
            "accepted_generated_sample_count": len(self._generated_embeddings),
        }
        provenance = result.provenance.model_copy(
            update={
                "real_sample_count": self._real_seen,
                "generated_sample_count": self._generated_seen,
                "excluded_sample_count": self._excluded,
                "exclusion_reasons": [
                    ExclusionReason(reason=reason, count=count)
                    for reason, count in sorted(self._exclusion_reasons.items())
                ],
                "hardware_runtime_class": self.metadata.capabilities.runtime_class,
            }
        )
        values = result.values.model_copy(update={"scalar_values": scalar_values})
        return result.model_copy(update={"values": values, "provenance": provenance})

    def finalize_artifacts(self) -> list[str]:
        self.validate_inputs()
        return []

    def reset(self) -> None:
        self._real_embeddings.clear()
        self._generated_embeddings.clear()
        self._case_ids.clear()
        self._real_seen = 0
        self._generated_seen = 0
        self._excluded = 0
        self._exclusion_reasons.clear()

    def _update_one(self, case: Mapping[str, Any]) -> None:
        if not isinstance(case, Mapping):
            raise MetricPluginError("each PRDC case must be a mapping")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise MetricPluginError("PRDC case requires a non-empty string case_id")
        if contains_private_path(case_id):
            raise MetricPluginError("PRDC case_id must not contain private-looking paths")
        if case_id in self._case_ids:
            raise MetricPluginError(f"duplicate PRDC case_id: {case_id}")

        role = case.get("role")
        if role not in {"real", "generated"}:
            raise MetricPluginError("PRDC case role must be 'real' or 'generated'")

        if case.get("contains_private_data") is True or case.get("contains_images") is True:
            raise MetricPluginError(
                "PRDC in-memory plugin accepts only aggregate embedding rows, "
                "not private-data or image-bearing cases"
            )

        is_exclude = case.get("exclude") is True
        if is_exclude:
            reason = case.get("exclude_reason", "excluded by PRDC case flag")
            if not isinstance(reason, str) or not reason:
                raise MetricPluginError("exclude_reason must be a non-empty string when provided")
        else:
            embedding = _coerce_one_embedding(case.get("embedding"), name="embedding")

        self._case_ids.add(case_id)
        if role == "real":
            self._real_seen += 1
        else:
            self._generated_seen += 1

        if is_exclude:
            self._excluded += 1
            self._exclusion_reasons[reason] = self._exclusion_reasons.get(reason, 0) + 1
            return
        if role == "real":
            self._real_embeddings.append(embedding)
        else:
            self._generated_embeddings.append(embedding)

    def _insufficient_data_result(self) -> MetricResult:
        warning = "PRDC requires at least two accepted real and two accepted generated embeddings"
        provenance = MetricProvenance(
            input_manifest_ids=self.context.input_manifest_ids if self.context else [],
            real_sample_count=self._real_seen,
            generated_sample_count=self._generated_seen,
            excluded_sample_count=self._excluded,
            exclusion_reasons=[
                ExclusionReason(reason=reason, count=count)
                for reason, count in sorted(self._exclusion_reasons.items())
            ],
            preprocessing_fingerprint=(
                self.context.preprocessing_fingerprint if self.context else "none/toy"
            ),
            feature_backbone_id="precomputed-embedding/manual",
            feature_layer="embedding",
            metric_config_hash=self.context.metric_config_hash if self.context else None,
            evaluator_git_commit=self.context.evaluator_git_commit if self.context else None,
            hardware_runtime_class=self.metadata.capabilities.runtime_class,
            citation_urls=[PRDC_CITATION_URL],
            implementation_reference_urls=[PRDC_REFERENCE_URL],
        )
        return MetricResult(
            metric_id=PRDC_PRECOMPUTED_METRIC_ID,
            metric_name=PRDC_PRECOMPUTED_METRIC_NAME,
            metric_family="prdc",
            metric_version="0.1.0",
            status="warning",
            status_reason=warning,
            claim=MetricClaim(
                claim_axis="diversity_coverage",
                claim_kind="coverage_diversity",
                reference_kind="real_distribution",
                evidence_level="exploratory",
            ),
            values=MetricValues(
                scalar_values={
                    "accepted_real_sample_count": len(self._real_embeddings),
                    "accepted_generated_sample_count": len(self._generated_embeddings),
                }
            ),
            provenance=provenance,
            safety=SafetyFlags(
                contains_private_paths=False,
                contains_images=False,
                safe_for_git=True,
                safe_for_public_export=True,
                redaction_mode="aggregate-only",
            ),
            warnings=[warning],
        )


def _validate_prdc_k_mode(
    value: PRDCK | str,
    *,
    error_cls: type[Exception] = ValueError,
) -> int | Literal["auto"]:
    if value is None:
        return "auto"
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw == "auto":
            return "auto"
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise error_cls("k must be a positive integer or 'auto'") from exc
        value = parsed
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise error_cls("k must be a positive integer or 'auto'")
    return int(value)


def _validate_prdc_auto_threshold(
    value: object,
    *,
    error_cls: type[Exception] = ValueError,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise error_cls("k_auto_threshold must be a finite number between 0 and 1")
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0 or parsed > 1.0:
        raise error_cls("k_auto_threshold must be > 0 and <= 1")
    return parsed


def _resolve_prdc_k(
    real: list[list[float]],
    generated: list[list[float]],
    *,
    k: PRDCK,
    k_auto_threshold: float,
) -> tuple[int, dict[str, Any], PRDCValues | None, tuple[str, ...]]:
    mode = _validate_prdc_k_mode(k)
    if mode != "auto":
        real_real_values = _real_real_prdc_values(real, k=mode)
        calibration = {
            "method": "manual",
            "status": "manual-exploratory",
            "manual_k": mode,
            "selected_k": mode,
            "threshold": k_auto_threshold,
            "real_real_baseline": _real_real_calibration_summary(real_real_values),
        }
        return mode, calibration, real_real_values, ()

    split = _real_real_calibration_split(real)
    if split is None:
        fallback_k = _fallback_prdc_k(real, generated)
        warning = "PRDC auto-k calibration skipped: insufficient real-real split support"
        return (
            fallback_k,
            {
                "method": "auto-real-real",
                "status": "fallback-manual-support",
                "threshold": k_auto_threshold,
                "selected_k": fallback_k,
                "reason": warning,
                "candidate_table": [],
            },
            None,
            (warning,),
        )

    real_left, real_right = split
    candidate_max = min(len(real_left), len(real_right), len(real), len(generated)) - 1
    if candidate_max < 1:
        fallback_k = _fallback_prdc_k(real, generated)
        warning = "PRDC auto-k calibration skipped: no feasible k for real-real split"
        return (
            fallback_k,
            {
                "method": "auto-real-real",
                "status": "fallback-manual-support",
                "threshold": k_auto_threshold,
                "selected_k": fallback_k,
                "reason": warning,
                "candidate_table": [],
                "split_policy": "alternating-even-odd",
            },
            None,
            (warning,),
        )

    sweep_cap = _resolve_prdc_auto_candidate_cap(candidate_max)
    candidate_table: list[dict[str, float | int]] = []
    selected_k: int | None = None
    selected_values: PRDCValues | None = None
    scanned_max = 0
    for candidate_k in range(1, sweep_cap + 1):
        candidate_values = compute_prdc(real_left, real_right, k=candidate_k)
        row = _prdc_values_row(candidate_values)
        candidate_table.append(row)
        scanned_max = candidate_k
        if (
            candidate_values.precision >= k_auto_threshold
            and candidate_values.recall >= k_auto_threshold
        ):
            selected_k = candidate_k
            selected_values = candidate_values
            break

    warnings: tuple[str, ...] = ()
    status = "ok"
    if selected_k is None:
        selected_k = scanned_max if scanned_max > 0 else 1
        selected_values = compute_prdc(real_left, real_right, k=selected_k)
        if sweep_cap < candidate_max:
            status = "threshold-not-crossed-within-cap"
            warnings = (
                "PRDC auto-k real-real precision/recall did not cross threshold within "
                f"sweep cap k={sweep_cap} (feasible max k={candidate_max}); "
                f"using k={selected_k}",
            )
        else:
            status = "threshold-not-crossed"
            warnings = (
                "PRDC auto-k real-real precision/recall did not both cross threshold; "
                f"using max feasible k={selected_k}",
            )
    if selected_values is None:
        selected_values = compute_prdc(real_left, real_right, k=selected_k)

    calibration = {
        "method": "auto-real-real",
        "status": status,
        "threshold": k_auto_threshold,
        "selected_k": selected_k,
        "candidate_max_k": candidate_max,
        "candidate_sweep_cap": sweep_cap,
        "split_policy": "alternating-even-odd",
        "left_sample_count": len(real_left),
        "right_sample_count": len(real_right),
        "candidate_table": candidate_table,
        "real_real_precision": selected_values.precision,
        "real_real_recall": selected_values.recall,
        "real_real_density": selected_values.density,
        "real_real_coverage": selected_values.coverage,
    }
    return selected_k, calibration, selected_values, warnings


def _resolve_prdc_auto_candidate_cap(candidate_max: int) -> int:
    """Cap PRDC auto-k linear sweep so large-N runs stay bounded.

    The auto-k loop recomputes a full real-real PRDC per candidate k, so an
    unbounded sweep is O(N^2 * d * candidate_max). Cap candidate_max at
    ``MEDGEN_PRDC_AUTO_CANDIDATE_MAX`` (default 32) so N >= 256 runs finish in
    bounded time. The fallback sweep at k=1..cap still records a sensitivity
    table; the selected_k breaks out early once both thresholds cross.
    """

    raw = os.environ.get("MEDGEN_PRDC_AUTO_CANDIDATE_MAX")
    default_cap = 32
    if raw is None:
        cap = default_cap
    else:
        try:
            cap = int(raw)
        except ValueError:
            cap = default_cap
        if cap <= 0:
            cap = default_cap
    return max(1, min(candidate_max, cap))


def _fallback_prdc_k(real: list[list[float]], generated: list[list[float]]) -> int:
    return max(1, min(len(real), len(generated)) - 1)


def _real_real_calibration_split(
    real: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]] | None:
    left = [row for index, row in enumerate(real) if index % 2 == 0]
    right = [row for index, row in enumerate(real) if index % 2 == 1]
    if len(left) < 2 or len(right) < 2:
        return None
    return left, right


def _real_real_prdc_values(real: list[list[float]], *, k: int) -> PRDCValues | None:
    split = _real_real_calibration_split(real)
    if split is None:
        return None
    left, right = split
    try:
        return compute_prdc(left, right, k=k)
    except ValueError:
        return None


def _real_real_scalar_values(values: PRDCValues | None) -> dict[str, float | int | bool]:
    if values is None:
        return {}
    return {
        "real_real_precision": values.precision,
        "real_real_recall": values.recall,
        "real_real_density": values.density,
        "real_real_coverage": values.coverage,
        "real_real_k": values.k,
        "real_real_left_sample_count": values.real_sample_count,
        "real_real_right_sample_count": values.generated_sample_count,
    }


def _real_real_calibration_summary(values: PRDCValues | None) -> dict[str, float | int | str]:
    if values is None:
        return {"status": "skipped", "reason": "insufficient real-real split support"}
    return {"status": "ok", **_prdc_values_row(values)}


def _prdc_values_row(values: PRDCValues) -> dict[str, float | int]:
    return {
        "k": values.k,
        "precision": values.precision,
        "recall": values.recall,
        "density": values.density,
        "coverage": values.coverage,
    }


def _parse_context(context: MetricPluginContext | Mapping[str, Any] | None) -> MetricPluginContext:
    return parse_metric_plugin_context(context)


def _coerce_embeddings(values: Sequence[Sequence[float]], *, name: str) -> list[list[float]]:
    if np is not None:
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric 2D array") from exc
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"{name} must be non-empty")
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"{name} must contain only finite values")
        return [[float(x) for x in row] for row in array.tolist()]

    if not isinstance(values, Sequence) or isinstance(values, str | bytes) or not values:
        raise ValueError(f"{name} must be a non-empty 2D sequence")
    return [_coerce_one_embedding(row, name=name) for row in values]


def _coerce_one_embedding(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        raise ValueError(f"{name} must be a non-empty numeric vector")
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{name} must contain only numeric values")
        coerced = float(item)
        if not isfinite(coerced):
            raise ValueError(f"{name} must contain only finite values")
        vector.append(coerced)
    return vector


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _nonnegative_float(value: object, *, label: str) -> float:
    parsed = _finite_float(value, label=label)
    if parsed < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _ratio_value(value: object, *, label: str) -> None:
    parsed = _nonnegative_float(value, label=label)
    if parsed > 1.0:
        raise ValueError(f"{label} must be between 0 and 1")


def _positive_int(value: object, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _bounded_count(value: object, *, maximum: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    if value > maximum:
        raise ValueError(f"{label} cannot exceed sample_count - 1")


def _safe_warning_tuple(values: object) -> None:
    if not isinstance(values, tuple):
        raise ValueError("warnings must be a tuple of strings")
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise ValueError(f"warnings[{index}] must be a non-empty string")
        if contains_private_path(value):
            raise ValueError(f"warnings[{index}] contains a private-looking path")
        if contains_secret(value):
            raise ValueError(f"warnings[{index}] contains secret-looking text")


def _validate_matching_dims(real: list[list[float]], generated: list[list[float]]) -> None:
    real_dim = len(real[0])
    generated_dim = len(generated[0])
    if real_dim != generated_dim:
        raise ValueError("real and generated embeddings must have the same dimension")
    if any(len(row) != real_dim for row in real):
        raise ValueError("real_embeddings must not be ragged")
    if any(len(row) != generated_dim for row in generated):
        raise ValueError("generated_embeddings must not be ragged")


def _pairwise_distances(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    if np is not None:
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b — uses O(N*M) memory instead of
        # materializing an (N, M, d) broadcast tensor (blows past RAM at d=2048).
        a2 = np.einsum("ij,ij->i", left_array, left_array)
        b2 = np.einsum("ij,ij->i", right_array, right_array)
        cross = left_array @ right_array.T
        squared = a2[:, None] + b2[None, :] - 2.0 * cross
        return np.sqrt(np.maximum(squared, 0.0)).tolist()

    distances: list[list[float]] = []
    for left_vector in left:
        row: list[float] = []
        for right_vector in right:
            squared = sum((a - b) * (a - b) for a, b in zip(left_vector, right_vector, strict=True))
            row.append(sqrt(squared))
        distances.append(row)
    return distances


def _nearest_neighbor_radii(embeddings: list[list[float]], k: int) -> list[float]:
    distances = _pairwise_distances(embeddings, embeddings)
    radii: list[float] = []
    for index, row in enumerate(distances):
        non_self = [distance for other_index, distance in enumerate(row) if other_index != index]
        non_self.sort()
        radii.append(non_self[k - 1])
    return radii


def _mean_nonself_pairwise_distance(embeddings: list[list[float]]) -> float:
    distances = _pairwise_distances(embeddings, embeddings)
    values = [
        distance
        for row_index, row in enumerate(distances)
        for col_index, distance in enumerate(row)
        if row_index != col_index
    ]
    return sum(values) / len(values) if values else 0.0


def _mean_flat(distances: list[list[float]]) -> float:
    values = [distance for row in distances for distance in row]
    return sum(values) / len(values) if values else 0.0


def _k_sensitivity_table(
    real_embeddings: Sequence[Sequence[float]], generated_embeddings: Sequence[Sequence[float]]
) -> list[dict[str, float | int]]:
    real_count = len(real_embeddings)
    generated_count = len(generated_embeddings)
    max_k = max(0, min(real_count, generated_count) - 1)
    rows: list[dict[str, float | int]] = []
    for k_value in range(1, min(max_k, 3) + 1):
        values = compute_prdc(real_embeddings, generated_embeddings, k=k_value)
        rows.append(
            {
                "k": values.k,
                "precision": values.precision,
                "recall": values.recall,
                "density": values.density,
                "coverage": values.coverage,
            }
        )
    return rows


def _duplicate_count(embeddings: list[list[float]]) -> int:
    seen: set[tuple[float, ...]] = set()
    duplicates = 0
    for row in embeddings:
        key = tuple(row)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates
