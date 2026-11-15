"""Discrete-time cyber-physical supply-chain environment."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np

from sc_cais.assurance import AssurancePolicy, RuntimeAssurance
from sc_cais.disruptions import Disruption
from sc_cais.models import (
    Action,
    ActionPlan,
    ActionType,
    AssuranceOutcome,
    Observation,
    Shipment,
    StepResult,
)
from sc_cais.network import SupplyChainNetwork

if TYPE_CHECKING:
    from sc_cais.agents import Agent


class SupplyChainEnv:
    def __init__(
        self,
        network: SupplyChainNetwork,
        *,
        seed: int = 0,
        disruptions: Iterable[Disruption] = (),
        demand_noise: float = 0.08,
        observation_noise: float = 0.0,
    ) -> None:
        self.network = network.clone()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.disruptions = list(disruptions)
        self.demand_noise = demand_noise
        self.observation_noise = max(0.0, observation_noise)
        self.step_index = 0
        self.shipments: list[Shipment] = []
        self.demand_multipliers: dict[str, float] = {}
        self.observation_inventory_multiplier: dict[str, float] = {}
        self.observation_trust_override: dict[str, float] = {}
        self.observation_missing_nodes: set[str] = set()
        self.observation_edge_active_override: dict[tuple[str, str], bool] = {}
        self.observation_edge_trust_override: dict[tuple[str, str], float] = {}
        self.cumulative_cost = 0.0
        self.cumulative_demand = 0.0
        self.cumulative_fulfilled = 0.0
        self.history: list[StepResult] = []
        self._disruptions_applied_for_step: int | None = None

    def _apply_disruptions(self) -> None:
        for disruption in self.disruptions:
            disruption.apply(self, self.step_index)

    def begin_step(self) -> None:
        """Apply step disruptions once before the decision policy observes the state."""
        if self._disruptions_applied_for_step != self.step_index:
            self._apply_disruptions()
            self._disruptions_applied_for_step = self.step_index

    def observe(self) -> Observation:
        nodes: dict[str, dict[str, float | bool | str]] = {}
        trust_values: list[float] = []
        for node in self.network.nodes():
            trust = self.observation_trust_override.get(node.node_id, node.trust)
            inventory = node.inventory * self.observation_inventory_multiplier.get(
                node.node_id, 1.0
            )
            if self.observation_noise > 0.0:
                node_index = sorted(n.node_id for n in self.network.nodes()).index(node.node_id)
                noise_rng = np.random.default_rng(
                    np.random.SeedSequence([self.seed, self.step_index, node_index, 9173])
                )
                inventory *= max(0.0, 1.0 + float(noise_rng.normal(0.0, self.observation_noise)))
            if node.node_id in self.observation_missing_nodes:
                inventory = 0.0
            trust_values.append(trust)
            nodes[node.node_id] = {
                "type": node.node_type.value,
                "inventory": inventory,
                "capacity": node.capacity,
                "backlog": node.backlog,
                "operating": node.operating,
                "trust": trust,
                "baseline_demand": node.baseline_demand,
            }
        edges: dict[str, dict[str, float | bool | int | str]] = {}
        for edge in self.network.edges():
            key = (edge.source, edge.target)
            trust = self.observation_edge_trust_override.get(key, edge.trust)
            trust_values.append(trust)
            edges[f"{edge.source}->{edge.target}"] = {
                "source": edge.source,
                "target": edge.target,
                "capacity": edge.capacity,
                "lead_time": edge.lead_time,
                "cost": edge.cost,
                "active": self.observation_edge_active_override.get(key, edge.active),
                "trust": trust,
            }
        return Observation(
            step=self.step_index,
            nodes=nodes,
            edges=edges,
            total_backlog=sum(n.backlog for n in self.network.nodes()),
            service_level=self.service_level,
            data_quality=(float(np.mean(trust_values)) if trust_values else 1.0)
            / (1.0 + self.observation_noise),
        )

    @property
    def service_level(self) -> float:
        if self.cumulative_demand <= 1e-12:
            return 1.0
        return min(1.0, self.cumulative_fulfilled / self.cumulative_demand)

    def _execute(self, action: Action) -> float:
        if action.action_type == ActionType.HOLD:
            return 0.0
        if action.source is None or action.target is None:
            return 0.0
        if not self.network.graph.has_edge(action.source, action.target):
            return 0.0
        edge = self.network.edge(action.source, action.target)
        source = self.network.node(action.source)
        target = self.network.node(action.target)
        if not edge.active or not source.operating or not target.operating:
            return 0.0
        # Physical saturation is independent from the control layer. An unassured
        # command may violate constraints, but the plant cannot exceed real capacity.
        quantity = max(0.0, min(action.quantity, source.inventory, edge.capacity, target.capacity))
        if quantity <= 0:
            return 0.0
        source.inventory -= quantity
        self.shipments.append(
            Shipment(
                action.source,
                action.target,
                quantity,
                edge.lead_time,
                edge.cost,
            )
        )
        return quantity * edge.cost

    def _advance_shipments(self) -> None:
        pending: list[Shipment] = []
        for shipment in self.shipments:
            shipment.eta -= 1
            if shipment.eta <= 0:
                target = self.network.node(shipment.target)
                if target.operating:
                    target.inventory += shipment.quantity
                else:
                    pending.append(
                        Shipment(
                            shipment.source,
                            shipment.target,
                            shipment.quantity,
                            1,
                            shipment.unit_cost,
                        )
                    )
            else:
                pending.append(shipment)
        self.shipments = pending

    def _demand_and_fulfilment(self) -> tuple[float, float, float]:
        total_demand = 0.0
        total_fulfilled = 0.0
        backlog_cost = 0.0
        for node in self.network.nodes():
            if node.baseline_demand <= 0:
                continue
            multiplier = self.demand_multipliers.get(node.node_id, 1.0)
            noise = max(0.0, 1.0 + float(self.rng.normal(0, self.demand_noise)))
            demand = node.baseline_demand * multiplier * noise
            requested = demand + node.backlog
            fulfilled = min(node.inventory, requested)
            node.inventory -= fulfilled
            node.backlog = requested - fulfilled
            total_demand += demand
            total_fulfilled += fulfilled
            backlog_cost += node.backlog * node.backlog_cost
        return total_demand, total_fulfilled, backlog_cost

    def _holding_cost(self) -> float:
        return sum(n.inventory * n.holding_cost for n in self.network.nodes())

    @staticmethod
    def _normalize_plan(proposed: Action | ActionPlan) -> ActionPlan:
        return proposed if isinstance(proposed, ActionPlan) else ActionPlan([proposed])

    def step(
        self,
        proposed: Action | ActionPlan,
        assurance: AssurancePolicy | None = None,
    ) -> StepResult:
        assurance = assurance or RuntimeAssurance()
        self.begin_step()
        plan = self._normalize_plan(proposed)
        outcomes: list[AssuranceOutcome] = []
        transport_cost = 0.0
        for action in plan.actions:
            outcome = assurance.decide(self, action)
            outcomes.append(outcome)
            transport_cost += self._execute(outcome.executed_action)
        self._advance_shipments()
        demand, fulfilled, backlog_cost = self._demand_and_fulfilment()
        step_cost = transport_cost + backlog_cost + self._holding_cost()
        self.cumulative_cost += step_cost
        self.cumulative_demand += demand
        self.cumulative_fulfilled += fulfilled
        result = StepResult(
            step=self.step_index,
            proposed_plan=plan,
            outcomes=outcomes,
            demand=demand,
            fulfilled=fulfilled,
            service_level=self.service_level,
            instant_service_level=min(1.0, fulfilled / max(demand, 1e-12)),
            total_backlog=sum(n.backlog for n in self.network.nodes()),
            step_cost=step_cost,
            cumulative_cost=self.cumulative_cost,
            data_quality=self.observe().data_quality,
        )
        self.history.append(result)
        self.step_index += 1
        return result

    def run(
        self,
        agent: Agent,
        assurance: AssurancePolicy,
        horizon: int,
    ) -> list[StepResult]:
        for _ in range(horizon):
            self.begin_step()
            plan = agent.act(self.observe(), self)
            self.step(plan, assurance)
        return self.history

    def snapshot(self) -> dict[str, object]:
        return {
            "step": self.step_index,
            "service_level": self.service_level,
            "cumulative_cost": self.cumulative_cost,
            "nodes": [asdict(n) for n in self.network.nodes()],
            "shipments": [asdict(s) for s in self.shipments],
        }
