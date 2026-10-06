"""Detector weights: downloaded once, checked against a pinned hash, cached locally."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    url: str
    sha256: str
    input_size: int


# YOLOX by Megvii, Apache-2.0: https://github.com/Megvii-BaseDetection/YOLOX
MODELS = {
    "yolox_s": ModelInfo(
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        input_size=640,
    ),
}


def cache_dir() -> Path:
    base = os.environ.get("JUNCTIONIQ_CACHE") or Path.home() / ".cache" / "junctioniq"
    return Path(base) / "models"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(name: str = "yolox_s", log=print) -> Path:
    """Return the local path of a model, downloading and verifying it if needed."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Available: {', '.join(MODELS)}")
    info = MODELS[name]
    path = cache_dir() / f"{name}.onnx"
    if path.is_file() and sha256_of(path) == info.sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    log(f"Downloading {name} (about 35 MB) to {path.parent} ...")
    urllib.request.urlretrieve(info.url, tmp)
    digest = sha256_of(tmp)
    if digest != info.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {name}: got {digest}")
    tmp.replace(path)
    return path
