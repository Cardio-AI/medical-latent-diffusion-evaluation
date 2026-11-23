"""Redaction helpers for privacy/path-safety findings.

These utilities are intentionally conservative and dependency-free.  They are
meant for gate output, logs, and reports that may include accidentally pasted
secrets or private filesystem paths.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

REDACTION = "<REDACTED>"

SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    # Anthropic keys (``sk-ant-api03-...``) embed hyphens, so the generic ``sk-``
    # pattern below (no hyphens, anchored ``\b``) never matches a bare one.
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[:=]\s*['\"]?[^'\"\s,;]{8,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----"),
)

# Roots that, when a path *starts* with them, mark the whole path private.  Scratch
# dirs (``/tmp``, ``/usr``, …) are intentionally excluded: they are not inherently
# PHI and flagging them yields pervasive false positives on temp-file fixtures.
_PRIVATE_ROOTS = (
    r"home|root|data|datasets|patients?|mnt|media|Volumes|private|protected|var/folders"
)
# High-signal segments treated as private *anywhere* in a path so that nested
# locations such as ``/usr/local/data/patient.nii`` are still flagged.
_PRIVATE_SEGMENTS = r"home|data|datasets|patients?|private|protected"

PRIVATE_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"(?<![\w:])/(?:{_PRIVATE_ROOTS})/[^\s'\"`),;]+"),
    re.compile(rf"/(?:{_PRIVATE_SEGMENTS})/[^\s'\"`),;]+"),
    re.compile(r"(?i)\b[A-Z]:\\(?:Users|Data|datasets?|images?|weights?|checkpoints?)\\[^\s'\"`),;]+"),
    re.compile(rf"file:///(?:{_PRIVATE_ROOTS})/[^\s'\"`),;]+"),
)


def iter_secret_matches(text: str) -> Iterable[re.Match[str]]:
    """Yield regex matches for secret-looking material."""

    for pattern in SECRET_PATTERNS:
        yield from pattern.finditer(text)


def iter_private_path_matches(text: str) -> Iterable[re.Match[str]]:
    """Yield regex matches for private-looking filesystem paths."""

    for pattern in PRIVATE_PATH_PATTERNS:
        yield from pattern.finditer(text)


def contains_secret(text: str) -> bool:
    """Return True when text contains a secret-looking token."""

    return any(iter_secret_matches(text))


def contains_private_path(text: str) -> bool:
    """Return True when text contains a private-looking filesystem path."""

    return any(iter_private_path_matches(text))


def redact_text(text: str, replacement: str = REDACTION) -> str:
    """Replace secret-looking tokens and private-looking paths in text."""

    redacted = text
    for pattern in (*SECRET_PATTERNS, *PRIVATE_PATH_PATTERNS):
        redacted = pattern.sub(replacement, redacted)
    return redacted
