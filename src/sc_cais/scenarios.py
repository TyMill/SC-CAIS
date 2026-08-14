"""Named benchmark scenarios for article-scale experiments."""

from __future__ import annotations

from dataclasses import dataclass

from sc_cais.disruptions import (
    CompoundDisruption,
    DemandSurge,
    Disruption,
    FalseInventorySignal,
    HubClosure,
    RouteSpoofing,
    SupplierFailure,
    TelemetryLoss,
    TransportDelay,
)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    disruptions: tuple[Disruption, ...]
    category: str
    severity: str


def default_scenarios() -> list[Scenario]:
    return [
        Scenario("normal", (), "baseline", "none"),
        Scenario("supplier_failure", (SupplierFailure("S1", 6, 14),), "operational", "medium"),
        Scenario("hub_closure", (HubClosure("H1", 6, 14),), "operational", "medium"),
        Scenario("demand_surge", (DemandSurge("C1", 6, 14, 2.2),), "operational", "high"),
        Scenario(
            "transport_delay",
            (TransportDelay("W1", "H1", 5, 15, 3),),
            "operational",
            "medium",
        ),
        Scenario(
            "false_inventory",
            (FalseInventorySignal("H1", 5, 15, 4.0, 0.3),),
            "cyber",
            "high",
        ),
        Scenario("telemetry_loss", (TelemetryLoss("H2", 5, 15, 0.1),), "cyber", "high"),
        Scenario(
            "route_spoofing",
            (RouteSpoofing("H1", "C1", 5, 15, False, 0.2),),
            "cyber",
            "high",
        ),
        Scenario(
            "compound",
            (
                CompoundDisruption(
                    [
                        SupplierFailure("S1", 5, 15),
                        DemandSurge("C1", 6, 14, 2.2),
                        FalseInventorySignal("H2", 6, 14, 3.5, 0.25),
                    ]
                ),
            ),
            "compound",
            "very_high",
        ),
        Scenario(
            "cyber_physical",
            (
                HubClosure("H1", 5, 15),
                RouteSpoofing("H1", "C1", 5, 15, True, 0.2),
                DemandSurge("C1", 7, 13, 1.8),
            ),
            "compound",
            "very_high",
        ),
    ]


def disruption_window(scenario: Scenario) -> tuple[int | None, int | None]:
    """Return the earliest start and latest end exposed by scenario disruptions."""
    starts: list[int] = []
    ends: list[int] = []

    def collect(item: object) -> None:
        start = getattr(item, "start", None)
        end = getattr(item, "end", None)
        if isinstance(start, int):
            starts.append(start)
        if isinstance(end, int):
            ends.append(end)
        nested = getattr(item, "disruptions", None)
        if nested:
            for child in nested:
                collect(child)

    for disruption in scenario.disruptions:
        collect(disruption)
    return (min(starts) if starts else None, max(ends) if ends else None)
