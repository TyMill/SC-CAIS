"""YAML configuration models."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    horizon: int = Field(default=24, ge=1, le=10000)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    agents: list[str] = Field(default_factory=lambda: ["rule_based", "agentic_heuristic"])
    assurance: list[str] = Field(default_factory=lambda: ["none", "reject_only", "sc_cais"])
    scenarios: list[str] = Field(default_factory=lambda: ["normal", "compound", "cyber_physical"])


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return BenchmarkConfig.model_validate(data)
