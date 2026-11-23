"""Machine-readable metric taxonomy.

This module defines first-class, structured claim semantics that metric specs
and metric results can carry explicitly. Consumers (study aggregation, report
datasets) should group evidence by these structured fields instead of
token-matching metric-family strings.

The vocabulary is intentionally aligned with existing schemas:

- ``pairing_mode`` values match :class:`schemas.manifests.common.PairingMode`.
- the segmentation reference/claim kinds in :mod:`segmentation.tables` are a
  subset of :data:`ReferenceKind` / :data:`ClaimKind` here.

A token-based inference helper is kept for backward compatibility, but it is a
clearly-labeled *fallback*: explicit claims on specs/results always win.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from pydantic import model_validator

from medgen_eval_lab.privacy.redaction import contains_private_path, contains_secret
from medgen_eval_lab.schemas._base import MedGenBaseModel

# --- Canonical vocabularies -------------------------------------------------

ClaimAxis = Literal[
    "fidelity",
    "diversity_coverage",
    "memorisation_privacy",
    "anatomy_physiology",
    "radiomics_phenotype",
    "downstream_utility",
    "efficiency_yield",
    "reporting_governance",
    "conditional_faithfulness",
]

ClaimKind = Literal[
    "distributional_fidelity",
    "coverage_diversity",
    "mode_collapse_screen",
    "copy_risk_screen",
    "memorisation_protocol",
    "distributional_anatomy_plausibility",
    "pseudo_reference_agreement",
    "manual_reference_segmentation_accuracy",
    "cardiac_function_plausibility",
    "temporal_volumetric_coherence",
    "radiomics_texture_shift",
    "downstream_task_delta",
    "efficiency_yield_summary",
    "export_readiness",
    "dashboard_qa",
    "schema_validation",
    "condition_faithfulness",
]

ReferenceKind = Literal[
    "real_distribution",
    "train_reference",
    "validation_calibration",
    "manual_gt",
    "resampled_manual_gt",
    "silver_model_pseudolabel",
    "conditioning_mask",
    "real_baseline_task",
    "none",
]

# Mirrors schemas.manifests.common.PairingMode string values without importing
# the manifest enum (keeps this module dependency-light for studies/reporting).
PairingMode = Literal[
    "unpaired_distributional",
    "paired_conditioning",
    "paired_real_reference",
    "paired_image_to_image",
]

# Ordered from weakest to strongest evidence strength.
EvidenceLevel = Literal[
    "toy",
    "exploratory",
    "calibrated",
    "reference_validated",
]

CLAIM_AXES: tuple[ClaimAxis, ...] = (
    "fidelity",
    "diversity_coverage",
    "memorisation_privacy",
    "anatomy_physiology",
    "radiomics_phenotype",
    "downstream_utility",
    "efficiency_yield",
    "reporting_governance",
    "conditional_faithfulness",
)

CLAIM_AXIS_TITLES: dict[ClaimAxis, str] = {
    "fidelity": "Fidelity",
    "diversity_coverage": "Diversity / coverage",
    "memorisation_privacy": "Memorisation / privacy",
    "anatomy_physiology": "Anatomy / physiology plausibility",
    "radiomics_phenotype": "Radiomics / phenotype texture",
    "downstream_utility": "Downstream utility",
    "efficiency_yield": "Efficiency / yield",
    "reporting_governance": "Reporting / governance",
    "conditional_faithfulness": "Conditional faithfulness",
}

EVIDENCE_LEVEL_ORDER: dict[EvidenceLevel, int] = {
    "toy": 0,
    "exploratory": 1,
    "calibrated": 2,
    "reference_validated": 3,
}

# Which claim kinds are scientifically valid under each axis.
AXIS_CLAIM_COMPATIBILITY: dict[ClaimAxis, frozenset[ClaimKind]] = {
    "fidelity": frozenset({"distributional_fidelity", "radiomics_texture_shift"}),
    "diversity_coverage": frozenset({"coverage_diversity", "mode_collapse_screen"}),
    "memorisation_privacy": frozenset({"copy_risk_screen", "memorisation_protocol"}),
    "anatomy_physiology": frozenset(
        {
            "distributional_anatomy_plausibility",
            "pseudo_reference_agreement",
            "manual_reference_segmentation_accuracy",
            "cardiac_function_plausibility",
            "temporal_volumetric_coherence",
        }
    ),
    "radiomics_phenotype": frozenset(
        {"radiomics_texture_shift", "distributional_anatomy_plausibility"}
    ),
    "downstream_utility": frozenset({"downstream_task_delta"}),
    "efficiency_yield": frozenset({"efficiency_yield_summary"}),
    "reporting_governance": frozenset(
        {"export_readiness", "dashboard_qa", "schema_validation"}
    ),
    "conditional_faithfulness": frozenset(
        {"condition_faithfulness", "manual_reference_segmentation_accuracy"}
    ),
}

_PAIRED_MODES = frozenset(
    {"paired_conditioning", "paired_real_reference", "paired_image_to_image"}
)


class MetricClaim(MedGenBaseModel):
    """Structured, machine-readable evaluation-claim semantics for a metric.

    Attaching this to a :class:`MetricResult` or declaring it on a
    :class:`MetricSpec` lets aggregation/reporting group by category and claim
    type without parsing free-text metric names.
    """

    claim_axis: ClaimAxis
    claim_kind: ClaimKind | None = None
    reference_kind: ReferenceKind = "none"
    pairing_mode: PairingMode = "unpaired_distributional"
    evidence_level: EvidenceLevel = "exploratory"

    @model_validator(mode="after")
    def claim_is_coherent(self) -> MetricClaim:
        if self.claim_kind is not None:
            allowed = AXIS_CLAIM_COMPATIBILITY[self.claim_axis]
            if self.claim_kind not in allowed:
                raise ValueError(
                    f"claim_kind {self.claim_kind!r} is not valid for claim_axis "
                    f"{self.claim_axis!r}; expected one of: "
                    + ", ".join(sorted(allowed))
                )
        if self.claim_axis == "conditional_faithfulness":
            if self.pairing_mode not in _PAIRED_MODES:
                raise ValueError(
                    "conditional_faithfulness claims require a paired pairing_mode"
                )
        elif self.pairing_mode in _PAIRED_MODES and self.claim_axis not in {
            "fidelity",
            "anatomy_physiology",
        }:
            raise ValueError(
                f"paired pairing_mode is not valid for claim_axis {self.claim_axis!r}"
            )
        if self.reference_kind == "conditioning_mask" and (
            self.claim_axis != "conditional_faithfulness"
        ):
            raise ValueError(
                "reference_kind=conditioning_mask requires claim_axis="
                "conditional_faithfulness"
            )
        return self


# --- Token-based fallback inference (legacy compatibility) ------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Ordered: first matching axis wins. Anatomy strong terms precede radiomics so a
# segmentation/cardiac/temporal metric is anatomy, not radiomics.
_AXIS_TOKEN_RULES: tuple[tuple[ClaimAxis, tuple[str, ...]], ...] = (
    ("conditional_faithfulness", ("conditioning", "condition", "faithfulness")),
    ("memorisation_privacy", (
        "privacy",
        "copy",
        "duplicate",
        "neighbor",
        "neighbour",
        "memorization",
        "memorisation",
        "novelty",
    )),
    ("downstream_utility", ("downstream", "task utility", "task-utility", "task_utility")),
    ("reporting_governance", (
        "dashboard",
        "qa",
        "benchmark",
        "governance",
        "report",
        "readiness",
    )),
    ("efficiency_yield", ("efficiency", "throughput", "yield", "nfe")),
    ("diversity_coverage", ("prdc", "diversity", "coverage", "recall", "density")),
    ("anatomy_physiology", ("segmentation", "anatomy", "cardiac", "temporal", "function")),
    ("radiomics_phenotype", ("radiomics", "frd")),
    ("fidelity", ("distribution", "fid", "kid", "mmd", "c2st", "fidelity")),
)


def infer_claim_axis(metric_id: str, metric_family: str, metric_name: str) -> ClaimAxis | None:
    """Best-effort axis inference from metric strings (fallback only).

    Explicit :class:`MetricClaim` data on specs/results should always be
    preferred over this token-matching heuristic.
    """

    text = _normalize_text((metric_id, metric_family, metric_name))
    tokens = tuple(text.split())
    token_set = set(tokens)
    for axis, terms in _AXIS_TOKEN_RULES:
        if _has_any(tokens, token_set, terms):
            return axis
    return None


def resolve_claim_axis(
    claim: MetricClaim | None,
    *,
    metric_id: str,
    metric_family: str,
    metric_name: str,
) -> tuple[ClaimAxis | None, Literal["explicit", "inferred", "unknown"]]:
    """Return ``(axis, source)`` preferring an explicit claim over tokens."""

    if claim is not None:
        return claim.claim_axis, "explicit"
    inferred = infer_claim_axis(metric_id, metric_family, metric_name)
    if inferred is not None:
        return inferred, "inferred"
    return None, "unknown"


def _normalize_text(values: Iterable[str]) -> str:
    return " ".join(" ".join(_TOKEN_RE.findall(value.lower())) for value in values)


def _has_any(tokens: tuple[str, ...], token_set: set[str], terms: Iterable[str]) -> bool:
    for term in terms:
        term_tokens = tuple(_TOKEN_RE.findall(term.lower()))
        if not term_tokens:
            continue
        if len(term_tokens) == 1:
            if term_tokens[0] in token_set:
                return True
            continue
        if _contains_sequence(tokens, term_tokens):
            return True
    return False


def _contains_sequence(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    if len(phrase) > len(tokens):
        return False
    last_start = len(tokens) - len(phrase) + 1
    return any(tokens[index : index + len(phrase)] == phrase for index in range(last_start))


# --- Spec-construction validation helpers -----------------------------------


def validate_claim_axis(value: object, *, label: str) -> None:
    if value not in CLAIM_AXES:
        raise ValueError(
            f"{label} must be one of: " + ", ".join(CLAIM_AXES)
        )


def validate_safe_taxonomy_text(value: str, *, label: str) -> None:
    if contains_private_path(value) or contains_secret(value):
        raise ValueError(f"{label} contains unsafe text")


__all__ = [
    "AXIS_CLAIM_COMPATIBILITY",
    "CLAIM_AXES",
    "CLAIM_AXIS_TITLES",
    "EVIDENCE_LEVEL_ORDER",
    "ClaimAxis",
    "ClaimKind",
    "EvidenceLevel",
    "MetricClaim",
    "PairingMode",
    "ReferenceKind",
    "infer_claim_axis",
    "resolve_claim_axis",
    "validate_claim_axis",
    "validate_safe_taxonomy_text",
]
