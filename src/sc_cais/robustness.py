"""Severity x uncertainty experiment design helpers."""

from __future__ import annotations

from dataclasses import dataclass

from sc_cais.scenarios import Scenario
from sc_cais.severity import severity_grid


@dataclass(frozen=True, slots=True)
class UncertaintyLevel:
    name: str
    observation_noise: float
    demand_noise: float


def uncertainty_grid() -> tuple[UncertaintyLevel, ...]:
    return (
        UncertaintyLevel("none", 0.00, 0.04),
        UncertaintyLevel("low", 0.05, 0.08),
        UncertaintyLevel("medium", 0.12, 0.12),
        UncertaintyLevel("high", 0.20, 0.18),
    )


def severity_uncertainty_scenarios() -> list[Scenario]:
    """Canonical severity scenarios used for robustness experiments."""
    return severity_grid()
