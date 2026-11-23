"""Local path and deterministic seed helpers for benchmark operators."""

from hashlib import sha256
from pathlib import Path


def resolve_data_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def stable_seed(*parts: object, bits: int = 32) -> int:
    raw = sha256("\0".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(raw[: bits // 4], 16)
