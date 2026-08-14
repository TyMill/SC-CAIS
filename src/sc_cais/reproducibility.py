"""Reproducibility metadata utilities."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def environment_manifest() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "pandas", "networkx", "scipy", "pydantic", "typer", "PyYAML"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output_dir: str | Path, files: list[str | Path] | None = None) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = environment_manifest()
    if files:
        payload["artifacts"] = {str(Path(f).name): sha256(f) for f in files}
    path = output / "reproducibility_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
