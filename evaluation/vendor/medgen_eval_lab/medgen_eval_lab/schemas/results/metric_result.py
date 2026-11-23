"""Normalized metric result and provenance schemas."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from medgen_eval_lab.privacy.redaction import contains_private_path, contains_secret
from medgen_eval_lab.schemas._base import SCHEMA_VERSION, MedGenBaseModel
from medgen_eval_lab.schemas.taxonomy import MetricClaim

MetricStatus = Literal["ok", "warning", "failed", "skipped"]
RedactionMode = Literal["none", "metadata-only", "aggregate-only", "redacted", "private-local-only"]
RuntimeClass = Literal["cpu-light", "gpu-feature", "gpu-heavy", "downstream-training"]
Dimensionality = Literal["2d", "3d", "2d+t", "3d+t/4d"]
PrivacyClassification = Literal["toy", "public", "internal", "private", "restricted"]


def _looks_like_bare_local_path(value: str) -> bool:
    return value.startswith(("/", "~/")) or "\\" in value


class ArtifactReference(MedGenBaseModel):
    """Reference to an external table/figure/report artifact, not the artifact contents."""

    uri: str = Field(..., description="URI such as toy://..., file://..., s3://..., or wandb://...")
    artifact_type: str = Field(..., description="Logical kind, e.g. table, figure, curve, report")
    media_type: str | None = Field(default=None, description="MIME type when known")
    description: str | None = None
    hash: str | None = Field(default=None, description="Optional content hash/fingerprint")
    contains_private_data: bool = False

    @field_validator("uri")
    @classmethod
    def uri_must_be_reference(cls, value: str) -> str:
        if _looks_like_bare_local_path(value):
            raise ValueError("artifact references must be URIs, not bare local paths")
        if "://" not in value:
            raise ValueError("artifact references must include a URI scheme")
        if contains_secret(value):
            raise ValueError("artifact reference uri contains secret-looking text")
        return value

    @model_validator(mode="after")
    def text_metadata_is_safe(self) -> ArtifactReference:
        _validate_required_safe_text(self.artifact_type, label="artifact_type")
        _validate_optional_safe_text(self.media_type, label="media_type")
        _validate_optional_safe_text(self.description, label="description")
        _validate_optional_safe_text(self.hash, label="hash")
        return self


class ConfidenceInterval(MedGenBaseModel):
    """Confidence interval for a scalar metric value."""

    low: float
    high: float
    level: float = Field(default=0.95, gt=0.0, lt=1.0)
    method: str | None = None

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> ConfidenceInterval:
        for label, value in {
            "low": self.low,
            "high": self.high,
            "level": self.level,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"confidence interval {label} must be finite")
        if self.low > self.high:
            raise ValueError("confidence interval low must be <= high")
        _validate_optional_safe_text(self.method, label="method")
        return self


class SignificanceResult(MedGenBaseModel):
    """Null-baseline significance test for a scalar metric value.

    Reports an honest bootstrap/permutation p-value against a stated null baseline
    (e.g. a real-vs-real or train-vs-val split) plus a standardized effect size
    (how many null standard deviations the observed value deviates from the null
    mean). Metrics without a meaningful null must not populate this — never
    fabricate a p-value.
    """

    p_value: float = Field(ge=0.0, le=1.0)
    null_baseline: str = Field(
        ...,
        description="Identifier of the null this was tested against, e.g. real_vs_real_split.",
    )
    method: str = Field(
        ...,
        description="How the p-value was derived, e.g. bootstrap_null/1000_repetitions.",
    )
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    effect_size: float | None = Field(
        default=None,
        description="Standardized effect size (observed minus null mean, in null SDs).",
    )
    repetitions: int | None = Field(default=None, ge=0)
    observed: float | None = None
    null_mean: float | None = None
    null_std: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def values_are_finite_and_safe(self) -> SignificanceResult:
        for label, value in {
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "observed": self.observed,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
        }.items():
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"significance {label} must be finite")
        _validate_required_safe_text(self.null_baseline, label="null_baseline")
        _validate_required_safe_text(self.method, label="method")
        return self


class FailureCase(MedGenBaseModel):
    """Metadata-only reference to a failed sample/case."""

    case_id: str
    reason: str
    severity: Literal["info", "warning", "error", "critical"] = "warning"
    artifact_refs: list[ArtifactReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def text_metadata_is_safe(self) -> FailureCase:
        _validate_required_safe_text(self.case_id, label="case_id")
        _validate_required_safe_text(self.reason, label="reason")
        return self


class MetricValues(MedGenBaseModel):
    """Metric outputs: scalars plus references to external tables/figures."""

    scalar_values: dict[str, float | int | bool] = Field(default_factory=dict)
    confidence_intervals: dict[str, ConfidenceInterval] = Field(default_factory=dict)
    significance_tests: dict[str, SignificanceResult] = Field(default_factory=dict)
    distributions: list[ArtifactReference] = Field(default_factory=list)
    table_refs: list[ArtifactReference] = Field(default_factory=list)
    per_case_table_ref: ArtifactReference | None = None
    per_structure_table_ref: ArtifactReference | None = None
    subgroup_table_ref: ArtifactReference | None = None
    sample_size_curve_ref: ArtifactReference | None = None
    figure_refs: list[ArtifactReference] = Field(default_factory=list)
    failure_cases: list[FailureCase] = Field(default_factory=list)
    extra_values: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def values_are_safe(self) -> MetricValues:
        for key, value in self.scalar_values.items():
            _validate_required_safe_text(key, label="scalar_values key")
            _validate_scalar_value(value, label=f"scalar_values[{key!r}]")
        for key in self.confidence_intervals:
            _validate_required_safe_text(key, label="confidence_intervals key")
        for key in self.significance_tests:
            _validate_required_safe_text(key, label="significance_tests key")
        _validate_extra_mapping(self.extra_values, label="extra_values")
        return self


class ExclusionReason(MedGenBaseModel):
    """Counted sample exclusions with reason text."""

    reason: str
    count: int = Field(ge=0)

    @field_validator("count", mode="before")
    @classmethod
    def count_must_not_be_bool(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("exclusion count must be a non-negative integer")
        return value

    @model_validator(mode="after")
    def metadata_is_safe(self) -> ExclusionReason:
        _validate_required_safe_text(self.reason, label="exclusion reason")
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise ValueError("exclusion count must be a non-negative integer")
        return self


class MetricProvenance(MedGenBaseModel):
    """Required provenance for interpreting a metric result."""

    input_manifest_ids: list[str] = Field(default_factory=list)
    real_sample_count: int | None = Field(default=None, ge=0)
    generated_sample_count: int | None = Field(default=None, ge=0)
    train_sample_count: int | None = Field(default=None, ge=0)
    validation_sample_count: int | None = Field(default=None, ge=0)
    excluded_sample_count: int = Field(default=0, ge=0)
    exclusion_reasons: list[ExclusionReason] = Field(default_factory=list)
    preprocessing_fingerprint: str | None = None
    feature_backbone_id: str | None = None
    feature_layer: str | None = None
    weights_hash: str | None = None
    real_feature_identity: str | None = None
    generated_feature_identity: str | None = None
    cache_keys: list[str] = Field(default_factory=list)
    metric_config_hash: str | None = None
    evaluator_git_commit: str | None = None
    hardware_runtime_class: RuntimeClass = "cpu-light"
    citation_urls: list[str] = Field(default_factory=list)
    implementation_reference_urls: list[str] = Field(default_factory=list)

    @field_validator(
        "real_sample_count",
        "generated_sample_count",
        "train_sample_count",
        "validation_sample_count",
        "excluded_sample_count",
        mode="before",
    )
    @classmethod
    def sample_counts_must_not_be_bool(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("sample counts must be non-negative integers")
        return value

    @model_validator(mode="after")
    def provenance_metadata_is_safe(self) -> MetricProvenance:
        for field_name in (
            "real_sample_count",
            "generated_sample_count",
            "train_sample_count",
            "validation_sample_count",
            "excluded_sample_count",
        ):
            _validate_optional_nonnegative_int(getattr(self, field_name), label=field_name)
        _validate_safe_text_list(self.input_manifest_ids, label="input_manifest_ids")
        _validate_optional_safe_text(
            self.preprocessing_fingerprint,
            label="preprocessing_fingerprint",
        )
        _validate_optional_safe_text(self.feature_backbone_id, label="feature_backbone_id")
        _validate_optional_safe_text(self.feature_layer, label="feature_layer")
        _validate_optional_safe_text(self.weights_hash, label="weights_hash")
        _validate_optional_safe_text(
            self.real_feature_identity, label="real_feature_identity"
        )
        _validate_optional_safe_text(
            self.generated_feature_identity, label="generated_feature_identity"
        )
        _validate_safe_text_list(self.cache_keys, label="cache_keys")
        _validate_optional_safe_text(self.metric_config_hash, label="metric_config_hash")
        _validate_optional_safe_text(self.evaluator_git_commit, label="evaluator_git_commit")
        _validate_safe_text_list(self.citation_urls, label="citation_urls")
        _validate_safe_text_list(
            self.implementation_reference_urls,
            label="implementation_reference_urls",
        )
        return self


class SafetyFlags(MedGenBaseModel):
    """Safety/export classification for a result or bundle."""

    contains_private_paths: bool = False
    contains_images: bool = False
    safe_for_git: bool = True
    safe_for_public_export: bool = False
    redaction_mode: RedactionMode = "metadata-only"

    @model_validator(mode="after")
    def unsafe_content_cannot_be_marked_safe(self) -> SafetyFlags:
        if self.safe_for_git and (self.contains_private_paths or self.contains_images):
            raise ValueError("results containing private paths or images are not safe for git")
        if self.safe_for_public_export and (
            self.contains_private_paths
            or self.contains_images
            or self.redaction_mode in {"none", "private-local-only"}
        ):
            raise ValueError("public export requires redaction and no private paths/images")
        return self


class MetricResult(MedGenBaseModel):
    """Normalized metric result emitted by metric plugins."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    metric_id: str
    metric_name: str
    metric_family: str
    metric_version: str
    status: MetricStatus
    status_reason: str | None = None
    claim: MetricClaim | None = Field(
        default=None,
        description="Machine-readable evaluation-claim taxonomy for this result.",
    )
    values: MetricValues = Field(default_factory=MetricValues)
    provenance: MetricProvenance
    safety: SafetyFlags = Field(default_factory=SafetyFlags)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_reason_required_for_non_ok(self) -> MetricResult:
        for label, value in {
            "metric_id": self.metric_id,
            "metric_name": self.metric_name,
            "metric_family": self.metric_family,
            "metric_version": self.metric_version,
        }.items():
            _validate_required_safe_text(value, label=label)
        _validate_optional_safe_text(self.status_reason, label="status_reason")
        _validate_safe_text_list(self.warnings, label="warnings")
        if self.status in {"warning", "failed", "skipped"} and not self.status_reason:
            raise ValueError("status_reason is required unless status is ok")
        if (
            self.safety.safe_for_public_export
            or self.safety.redaction_mode == "aggregate-only"
        ) and _contains_row_level_payload(self.values.extra_values):
            raise ValueError(
                "public/aggregate metric results must not contain case IDs or "
                "per_case/rows/records collections; use a private external table reference"
            )
        return self


def _validate_required_safe_text(value: object, *, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    _validate_safe_text(value, label=label)


def _validate_optional_safe_text(value: object, *, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    _validate_safe_text(value, label=label)


def _validate_safe_text(value: str, *, label: str) -> None:
    if contains_private_path(value):
        raise ValueError(f"{label} contains a private-looking path")
    if contains_secret(value):
        raise ValueError(f"{label} contains secret-looking text")


def _validate_safe_text_list(values: list[str], *, label: str) -> None:
    for index, value in enumerate(values):
        _validate_required_safe_text(value, label=f"{label}[{index}]")


def _validate_scalar_value(value: object, *, label: str) -> None:
    if isinstance(value, bool):
        return
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite numeric or boolean")


def _validate_optional_nonnegative_int(value: object, *, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _validate_extra_mapping(values: dict[str, Any], *, label: str) -> None:
    for key, value in values.items():
        _validate_required_safe_text(key, label=f"{label} key")
        _validate_extra_value(value, label=f"{label}[{key!r}]")


def _validate_extra_value(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        _validate_safe_text(value, label=label)
        return
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_required_safe_text(str(key), label=f"{label} key")
            _validate_extra_value(item, label=f"{label}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_extra_value(item, label=f"{label}[{index}]")
        return
    raise ValueError(f"{label} must be JSON-serializable")


def _contains_row_level_payload(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            compact = normalized.replace("_", "")
            if compact.endswith(("caseid", "caseids", "patientid", "patientids")):
                return True
            row_collection = (
                normalized in {"rows", "records", "per_case"}
                or normalized.startswith("per_case_")
            )
            if row_collection and isinstance(item, (list, tuple, dict)):
                return True
            if _contains_row_level_payload(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_row_level_payload(item) for item in value)
    return False
