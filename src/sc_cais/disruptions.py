"""Operational and cyber-physical disruption primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sc_cais.environment import SupplyChainEnv


class Disruption(Protocol):
    name: str

    def apply(self, env: SupplyChainEnv, step: int) -> None: ...


@dataclass(slots=True)
class NodeOutage:
    node_id: str
    start: int
    end: int
    name: str = "node_outage"

    def apply(self, env: SupplyChainEnv, step: int) -> None:
        env.network.node(self.node_id).operating = not (self.start <= step < self.end)


@dataclass(slots=True)
class SupplierFailure(NodeOutage):
    name: str = "supplier_failure"


@dataclass(slots=True)
class HubClosure(NodeOutage):
    name: str = "hub_closure"


@dataclass(slots=True)
class EdgeOutage:
    source: str
    target: str
    start: int
    end: int
    name: str = "edge_outage"

    def apply(self, env: SupplyChainEnv, step: int) -> None:
        env.network.edge(self.source, self.target).active = not (self.start <= step < self.end)


@dataclass(slots=True)
class TransportDelay:
    source: str
    target: str
    start: int
    end: int
    extra_lead_time: int = 2
    name: str = "transport_delay"
    _base: int | None = field(default=None, init=False)

    def apply(self, env: SupplyChainEnv, step: int) -> None:
        edge = env.network.edge(self.source, self.target)
        if self._base is None:
            self._base = edge.lead_time
        if self.start <= step < self.end:
            edge.lead_time = self._base + self.extra_lead_time
        else:
            edge.lead_time = self._base


@dataclass(slots=True)
class DemandSurge:
    node_id: str
    start: int
    end: int
    multiplier: float = 2.0
    name: str = "demand_surge"

    def apply(self, env: SupplyChainEnv, step: int) -> None:
        if self.start <= step < self.end:
            env.demand_multipliers[self.node_id] = self.multiplier
        else:
            env.demand_multipliers.pop(self.node_id, None)


@dataclass(slots=True)
class FalseInventorySignal:
    node_id: str
    start: int
    end: int
    multiplier: float = 3.0
    trust: float = 0.35
    name: str = "false_inventory_signal"

    def apply(self, env: SupplyChainEnv, step: int) -> None:
        if self.start <= step < self.end:
            env.observation_inventory_multiplier[self.node_id] = self.multiplier
            env.observation_trust_override[self.node_id] = self.trust
        else:
            env.observation_inventory_multiplier.pop(self.node_id, None)
            env.observation_trust_override.pop(self.node_id, None)


@dataclass(slots=True)
class TelemetryLoss:
    node_id: str
    start: int
    end: int
    trust: float = 0.1
    name: str = "telemetry_loss"

    def apply(self, env: SupplyChainEnv, step: int) -> None:
        if self.start <= step < self.end:
            env.observation_trust_override[self.node_id] = self.trust
            env.observation_missing_nodes.add(self.node_id)
        else:
            env.observation_trust_override.pop(self.node_id, None)
            env.observation_missing_nodes.discard(self.node_id)


@dataclass(slots=True)
class RouteSpoofing:
    source: str
    target: str
    start: int
    end: int
    reported_active: bool = True
    trust: float = 0.25
    name: str = "route_spoofing"

    def apply(self, env: SupplyChainEnv, step: int) -> None:
        key = (self.source, self.target)
        if self.start <= step < self.end:
            env.observation_edge_active_override[key] = self.reported_active
            env.observation_edge_trust_override[key] = self.trust
        else:
            env.observation_edge_active_override.pop(key, None)
            env.observation_edge_trust_override.pop(key, None)


@dataclass(slots=True)
class CompoundDisruption:
    disruptions: list[Disruption]
    name: str = "compound_disruption"

    def apply(self, env: SupplyChainEnv, step: int) -> None:
        for disruption in self.disruptions:
            disruption.apply(env, step)
