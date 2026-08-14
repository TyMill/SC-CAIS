"""Lightweight verification script that requires no external linters."""

from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if not compileall.compile_dir(ROOT / "src", quiet=1):
    raise SystemExit("compileall failed")

subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)
subprocess.run(
    [sys.executable, "-m", "sc_cais.cli", "smoke", "--horizon", "8", "--seed", "0"],
    cwd=ROOT,
    env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    check=True,
)
print("SC-CAIS verification passed")
