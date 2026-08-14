"""Parameterized severity grids for robustness experiments."""

from __future__ import annotations

from sc_cais.disruptions import DemandSurge, FalseInventorySignal, HubClosure, SupplierFailure
from sc_cais.scenarios import Scenario


def severity_grid() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for label, multiplier in (("low", 1.25), ("medium", 1.75), ("high", 2.5), ("extreme", 3.5)):
        scenarios.append(
            Scenario(
                f"demand_surge_{label}",
                (DemandSurge("C1", 6, 16, multiplier),),
                "operational",
                label,
            )
        )
    for label, duration in (("low", 3), ("medium", 6), ("high", 10), ("extreme", 14)):
        scenarios.append(
            Scenario(
                f"supplier_failure_{label}",
                (SupplierFailure("S1", 6, 6 + duration),),
                "operational",
                label,
            )
        )
    for label, trust in (("low", 0.65), ("medium", 0.45), ("high", 0.25), ("extreme", 0.05)):
        scenarios.append(
            Scenario(
                f"false_inventory_{label}",
                (FalseInventorySignal("H1", 6, 16, 4.0, trust),),
                "cyber",
                label,
            )
        )
    for label, duration in (("low", 3), ("medium", 6), ("high", 10), ("extreme", 14)):
        scenarios.append(
            Scenario(
                f"hub_closure_{label}",
                (HubClosure("H1", 6, 6 + duration),),
                "operational",
                label,
            )
        )
    return scenarios
