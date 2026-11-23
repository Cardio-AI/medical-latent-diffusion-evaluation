"""Shared schema primitives for medgen-eval-lab."""

from __future__ import annotations

from types import UnionType
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

SCHEMA_VERSION = "0.1.0"


class MedGenBaseModel(BaseModel):
    """Base class for first-wave Pydantic v2 contracts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def numeric_fields_reject_booleans(cls, value: Any, info: ValidationInfo) -> Any:
        """Keep JSON booleans from being coerced into numeric configuration values.

        Python's ``bool`` is a subclass of ``int`` and Pydantic's non-strict numeric
        fields therefore accept ``true`` as ``1`` unless every model remembers a
        bespoke validator.  Apply the public contract once for scalar and container
        annotations while preserving fields that explicitly advertise ``bool``.
        """

        field = cls.model_fields.get(info.field_name)
        if field is not None:
            _reject_bool_for_numeric_annotation(value, field.annotation, path=info.field_name)
        return value

    @field_validator("schema_version", check_fields=False)
    @classmethod
    def schema_version_matches_current(cls, value: str) -> str:
        """Validate versioned contracts against the project schema version."""

        if value != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        return value


def _reject_bool_for_numeric_annotation(value: Any, annotation: Any, *, path: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if _annotation_accepts_numeric(annotation) and not _annotation_accepts_bool(annotation):
            raise ValueError(f"{path} must be numeric, not boolean")
        return

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, set, frozenset, tuple) and isinstance(value, (list, tuple, set)):
        item_annotations = args
        for index, item in enumerate(value):
            item_annotation = (
                item_annotations[index]
                if (
                    origin is tuple
                    and len(item_annotations) > 1
                    and item_annotations[-1] is not Ellipsis
                )
                else item_annotations[0] if item_annotations else Any
            )
            _reject_bool_for_numeric_annotation(item, item_annotation, path=f"{path}[{index}]")
    elif origin is dict and isinstance(value, dict) and len(args) == 2:
        for key, item in value.items():
            _reject_bool_for_numeric_annotation(item, args[1], path=f"{path}[{key!r}]")


def _annotation_accepts_numeric(annotation: Any) -> bool:
    if annotation in (int, float):
        return True
    origin = get_origin(annotation)
    if origin is Literal:
        return any(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in get_args(annotation)
        )
    if origin is UnionType:
        return any(_annotation_accepts_numeric(item) for item in get_args(annotation))
    return False


def _annotation_accepts_bool(annotation: Any) -> bool:
    if annotation is bool:
        return True
    origin = get_origin(annotation)
    if origin is Literal:
        return any(isinstance(item, bool) for item in get_args(annotation))
    if origin is UnionType:
        return any(_annotation_accepts_bool(item) for item in get_args(annotation))
    return False
