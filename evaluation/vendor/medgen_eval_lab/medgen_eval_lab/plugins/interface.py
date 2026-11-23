"""Metric plugin lifecycle contracts.

Wave 1 intentionally defines only the light-weight interface and metadata needed by
future metric kernels.  Implementations must stay safe for ``local-toy`` use unless
explicitly documented otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from medgen_eval_lab.schemas._base import MedGenBaseModel
from medgen_eval_lab.schemas.results.metric_result import Dimensionality, MetricResult, RuntimeClass

Modality = Literal["CMR", "MRI", "CT", "CTA", "X-ray", "other"]
PairedOrUnpaired = Literal["paired", "unpaired cohort", "generated-vs-train", "generated-only"]
ExecutionMode = Literal["local-toy", "private-local", "private-wandb"]
RequiredInput = Literal[
    "image",
    "generated image",
    "real/reference image",
    "train reference set",
    "validation/calibration set",
    "mask",
    "segmentation",
    "conditioning",
    "feature embeddings",
    "task predictions",
    "temporal metadata",
    "spacing/orientation",
    "labels/phenotypes",
    "site/scanner strata",
]


class MetricPluginIdentity(MedGenBaseModel):
    """Human- and machine-readable plugin identity."""

    metric_name: str
    metric_family: str
    metric_version: str
    citation_urls: list[str] = Field(default_factory=list)
    implementation_reference_urls: list[str] = Field(default_factory=list)
    owner_or_reviewer: str


class MetricPluginCapabilities(MedGenBaseModel):
    """Static capabilities declared before a plugin is run."""

    supported_dimensionality: list[Dimensionality]
    supported_modalities: list[Modality]
    supports_streaming_update: bool
    supports_distributed: bool
    requires_gpu: bool
    runtime_class: RuntimeClass
    paired_or_unpaired: PairedOrUnpaired
    required_inputs: list[RequiredInput] = Field(default_factory=list)

    @model_validator(mode="after")
    def gpu_requirement_matches_runtime_class(self) -> MetricPluginCapabilities:
        if self.requires_gpu and self.runtime_class == "cpu-light":
            raise ValueError("GPU-requiring plugins cannot declare runtime_class='cpu-light'")
        if not self.supported_dimensionality:
            raise ValueError("at least one supported dimensionality is required")
        if not self.supported_modalities:
            raise ValueError("at least one supported modality is required")
        return self


class MetricPluginMetadata(MedGenBaseModel):
    """Combined plugin metadata following the metric schema blueprint."""

    identity: MetricPluginIdentity
    capabilities: MetricPluginCapabilities


class MetricPluginContext(MedGenBaseModel):
    """Minimal execution context shared by Wave 1 plugins."""

    execution_mode: ExecutionMode = "local-toy"
    input_manifest_ids: list[str] = Field(default_factory=list)
    evaluator_git_commit: str | None = None
    preprocessing_fingerprint: str | None = "none/toy"
    metric_config_hash: str | None = None
    artifact_output_uri: str | None = Field(
        default=None,
        description="Optional external artifact destination URI; plugins may ignore it.",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class MetricPluginError(ValueError):
    """Base exception for metric plugin lifecycle failures."""


def parse_metric_plugin_context(
    context: MetricPluginContext | Mapping[str, Any] | None = None,
) -> MetricPluginContext:
    """Parse a mapping or existing context into ``MetricPluginContext``."""

    if context is None:
        return MetricPluginContext()
    if isinstance(context, MetricPluginContext):
        return context
    return MetricPluginContext.model_validate(context)


def require_local_toy_context(
    context: MetricPluginContext | Mapping[str, Any] | None = None,
    *,
    plugin_name: str,
) -> MetricPluginContext:
    """Parse context and reject non-``local-toy`` execution modes.

    CPU-safe toy plugins share this guard so future GPU plugins can be added
    without copying local-only assumptions into every implementation.
    """

    parsed_context = parse_metric_plugin_context(context)
    if parsed_context.execution_mode != "local-toy":
        raise MetricPluginError(f"{plugin_name} is only allowed in local-toy mode")
    return parsed_context


class MetricPlugin(ABC):
    """Abstract lifecycle for metric plugins.

    Required lifecycle methods are ``prepare(context)``, ``validate_inputs()``,
    ``update(batch_or_case)``, ``compute()``, ``finalize_artifacts()``, and
    ``reset()``.
    """

    metadata: MetricPluginMetadata

    @abstractmethod
    def prepare(self, context: MetricPluginContext | Mapping[str, Any] | None = None) -> None:
        """Validate config and attach an execution context before updates."""

    @abstractmethod
    def validate_inputs(self) -> None:
        """Fail early on missing metadata, unsafe execution mode, or bad state."""

    @abstractmethod
    def update(self, batch_or_case: Mapping[str, Any] | list[Mapping[str, Any]]) -> None:
        """Accumulate one case or a batch of cases."""

    @abstractmethod
    def compute(self) -> MetricResult:
        """Return a normalized metric result."""

    @abstractmethod
    def finalize_artifacts(self) -> list[str]:
        """Finalize optional external artifacts and return safe URI references."""

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state while preserving static plugin configuration."""
