# Adapted from the Apache-2.0 checkpoint-key converter. See the vendored evaluation notice.
# Licensed under the Apache License, Version 2.0. See the vendored evaluation LICENSE.
"""Convert a RadImageNet checkpoint to torchvision ResNet50 keys."""

from typing import Any

class FeatureExtractionError(ValueError):
    pass

_RADIMAGENET_BACKBONE_KEY_PREFIXES = (
    ("backbone.0.", "conv1."),
    ("backbone.1.", "bn1."),
    ("backbone.4.", "layer1."),
    ("backbone.5.", "layer2."),
    ("backbone.6.", "layer3."),
    ("backbone.7.", "layer4."),
)

def _radimagenet_resnet50_state_dict(
    checkpoint: Any,
    *,
    tensor_predicate: Any | None = None,
) -> dict[str, Any]:
    """Extract and clean a torchvision-compatible state dict."""

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise FeatureExtractionError("RadImageNet checkpoint must contain a state dict")
    if not state:
        raise FeatureExtractionError("RadImageNet checkpoint state dict must not be empty")
    cleaned: dict[str, Any] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise FeatureExtractionError("RadImageNet state dict keys must be strings")
        if tensor_predicate is not None and not tensor_predicate(value):
            raise FeatureExtractionError(
                f"RadImageNet state dict value for {key!r} must be a tensor"
            )
        name = key
        for prefix in ("_orig_mod.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        name = _radimagenet_torchvision_key(name)
        if name in cleaned:
            raise FeatureExtractionError(
                f"RadImageNet state dict contains duplicate normalized key {name!r}"
            )
        cleaned[name] = value
    return cleaned


def _radimagenet_torchvision_key(name: str) -> str:
    """Map RadImageNet's published backbone-only keys to torchvision ResNet50."""

    for source, target in _RADIMAGENET_BACKBONE_KEY_PREFIXES:
        if name.startswith(source):
            return target + name[len(source) :]
    return name
