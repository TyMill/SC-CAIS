"""Baseline and agentic decision policies for SC-CAIS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import networkx as nx
import numpy as np

from sc_cais.models import Action, ActionPlan, ActionType, NodeType, Observation

if TYPE_CHECKING:
    from sc_cais.environment import SupplyChainEnv


class Agent(Protocol):
    name: str

    def act(self, observation: Observation, env: SupplyChainEnv) -> ActionPlan: ...


def _downstream_demand(env: SupplyChainEnv, node_id: str) -> float:
    descendants = nx.descendants(env.network.graph, node_id)
    return sum(
        env.network.node(n).baseline_demand
        for n in descendants
        if env.network.node(n).node_type == NodeType.CUSTOMER
    )


def _deficit(observation: Observation, env: SupplyChainEnv, node_id: str, cover: float) -> float:
    node = env.network.node(node_id)
    inv = float(observation.nodes[node_id]["inventory"])
    if node.node_type == NodeType.CUSTOMER:
        need = float(observation.nodes[node_id]["backlog"]) + cover * node.baseline_demand
    else:
        need = cover * _downstream_demand(env, node_id)
    return max(0.0, need - inv)


def _perceived_predecessors(
    observation: Observation,
    env: SupplyChainEnv,
    target: str,
) -> list[str]:
    result = []
    for source in env.network.graph.predecessors(target):
        edge = observation.edges[f"{source}->{target}"]
        if (
            bool(edge["active"])
            and bool(observation.nodes[source]["operating"])
            and bool(observation.nodes[target]["operating"])
        ):
            result.append(source)
    return result


def _targets_downstream_first(env: SupplyChainEnv) -> list[str]:
    order = {
        NodeType.CUSTOMER: 0,
        NodeType.HUB: 1,
        NodeType.WAREHOUSE: 2,
        NodeType.PLANT: 3,
        NodeType.SUPPLIER: 4,
    }
    return [
        n.node_id
        for n in sorted(env.network.nodes(), key=lambda n: (order[n.node_type], n.node_id))
        if n.node_type != NodeType.SUPPLIER
    ]


@dataclass(slots=True)
class RuleBasedAgent:
    """Transparent reorder-point baseline producing a multi-echelon action plan."""

    name: str = "rule_based"
    cover: float = 1.5
    shipment_quantity: float = 24.0

    def act(self, observation: Observation, env: SupplyChainEnv) -> ActionPlan:
        actions: list[Action] = []
        for target in _targets_downstream_first(env):
            deficit = _deficit(observation, env, target, self.cover)
            if deficit <= 0:
                continue
            predecessors = _perceived_predecessors(observation, env, target)
            if not predecessors:
                continue
            source = max(predecessors, key=lambda p: float(observation.nodes[p]["inventory"]))
            quantity = min(self.shipment_quantity, deficit)
            actions.append(
                Action(
                    ActionType.SHIP,
                    source,
                    target,
                    quantity,
                    {"final_target": target, "policy": self.name},
                )
            )
        return ActionPlan(actions or [Action.hold("no reorder action")], {"agent": self.name})


@dataclass(slots=True)
class AgenticHeuristicAgent:
    """Adaptive multi-echelon policy that may propose aggressive autonomous actions."""

    name: str = "agentic_heuristic"
    cover: float = 1.8
    aggressiveness: float = 1.35
    uncertainty_gain: float = 0.5

    def act(self, observation: Observation, env: SupplyChainEnv) -> ActionPlan:
        actions: list[Action] = []
        uncertainty = 1.0 - observation.data_quality
        for target in _targets_downstream_first(env):
            deficit = _deficit(observation, env, target, self.cover)
            if deficit <= 0:
                continue
            predecessors = _perceived_predecessors(observation, env, target)
            if not predecessors:
                continue
            ranked: list[tuple[float, str]] = []
            for source in predecessors:
                edge = observation.edges[f"{source}->{target}"]
                trust = min(
                    float(observation.nodes[source]["trust"]),
                    float(observation.nodes[target]["trust"]),
                    float(edge["trust"]),
                )
                score = (
                    float(observation.nodes[source]["inventory"])
                    - 8.0 * float(edge["cost"])
                    + 12.0 * trust
                )
                ranked.append((score, source))
            _, source = max(ranked)
            quantity = max(
                4.0,
                deficit * self.aggressiveness * (1 + self.uncertainty_gain * uncertainty),
            )
            actions.append(
                Action(
                    (
                        ActionType.REALLOCATE
                        if env.network.node(target).node_type != NodeType.CUSTOMER
                        else ActionType.SHIP
                    ),
                    source,
                    target,
                    float(quantity),
                    {
                        "final_target": target,
                        "agentic": True,
                        "perceived_data_quality": observation.data_quality,
                    },
                )
            )
        return ActionPlan(actions or [Action.hold("no urgent action")], {"agent": self.name})


@dataclass(slots=True)
class CostAwareOptimizerAgent:
    """Deterministic local min-cost flow-style baseline for each replenishment target."""

    name: str = "cost_aware_optimizer"
    cover: float = 1.7
    max_quantity: float = 40.0

    def act(self, observation: Observation, env: SupplyChainEnv) -> ActionPlan:
        actions: list[Action] = []
        for target in _targets_downstream_first(env):
            deficit = _deficit(observation, env, target, self.cover)
            if deficit <= 0:
                continue
            predecessors = _perceived_predecessors(observation, env, target)
            if not predecessors:
                continue
            source = min(
                predecessors,
                key=lambda p: (
                    float(observation.edges[f"{p}->{target}"]["cost"]),
                    -float(observation.nodes[p]["inventory"]),
                ),
            )
            actions.append(
                Action(
                    ActionType.SHIP,
                    source,
                    target,
                    min(self.max_quantity, max(4.0, deficit)),
                    {"final_target": target, "optimizer": True},
                )
            )
        return ActionPlan(
            actions or [Action.hold("optimizer target cover satisfied")],
            {"agent": self.name},
        )


@dataclass(slots=True)
class StochasticAgent:
    """Stress-test policy useful for assurance ablations."""

    name: str = "stochastic"
    seed: int = 0
    plan_size: int = 3

    def act(self, observation: Observation, env: SupplyChainEnv) -> ActionPlan:
        del observation
        rng = np.random.default_rng(self.seed + env.step_index)
        edges = env.network.edges()
        actions = []
        for _ in range(self.plan_size):
            edge = edges[int(rng.integers(0, len(edges)))]
            actions.append(
                Action(
                    ActionType.SHIP,
                    edge.source,
                    edge.target,
                    float(rng.uniform(5, 90)),
                )
            )
        return ActionPlan(actions, {"agent": self.name})


@dataclass(slots=True)
class RollingHorizonMILPAgent:
    """Integer rolling-horizon replenishment baseline solved with SciPy MILP.

    The model optimizes current shipment quantities against projected inventory
    deficits over a configurable planning horizon. It respects perceived route
    availability, edge capacities, source inventory, and target replenishment needs.
    """

    name: str = "rolling_horizon_milp"
    planning_horizon: int = 4
    cover: float = 1.6
    shortage_penalty: float = 25.0
    max_quantity: float = 80.0

    def act(self, observation: Observation, env: SupplyChainEnv) -> ActionPlan:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix

        candidates: list[tuple[str, str]] = []
        target_need: dict[str, float] = {}
        for target in _targets_downstream_first(env):
            projected_cover = self.cover + max(0, self.planning_horizon - 1) * 0.25
            deficit = _deficit(observation, env, target, projected_cover)
            if deficit <= 1e-9:
                continue
            predecessors = _perceived_predecessors(observation, env, target)
            if not predecessors:
                continue
            target_need[target] = min(self.max_quantity, deficit)
            candidates.extend((source, target) for source in predecessors)

        if not candidates:
            return ActionPlan.hold("milp target cover satisfied")

        targets = sorted(target_need)
        n_ship = len(candidates)
        n_short = len(targets)
        n_vars = n_ship + n_short
        target_pos = {target: i for i, target in enumerate(targets)}

        c = np.zeros(n_vars, dtype=float)
        upper = np.zeros(n_vars, dtype=float)
        for idx, (source, target) in enumerate(candidates):
            edge = observation.edges[f"{source}->{target}"]
            lead = float(edge["lead_time"])
            trust = min(
                float(observation.nodes[source]["trust"]),
                float(observation.nodes[target]["trust"]),
                float(edge["trust"]),
            )
            c[idx] = float(edge["cost"]) + 0.1 * lead + (1.0 - trust) * 2.0
            upper[idx] = min(float(edge["capacity"]), self.max_quantity)
        for target, pos in target_pos.items():
            j = n_ship + pos
            c[j] = self.shortage_penalty
            upper[j] = target_need[target]

        rows: list[tuple[dict[int, float], float, float]] = []
        # Target balance: inbound + shortage equals target need.
        for target, pos in target_pos.items():
            coeff: dict[int, float] = {n_ship + pos: 1.0}
            for idx, (_, candidate_target) in enumerate(candidates):
                if candidate_target == target:
                    coeff[idx] = 1.0
            rows.append((coeff, target_need[target], target_need[target]))

        # Source inventory across all simultaneous decisions.
        sources = sorted({source for source, _ in candidates})
        for source in sources:
            coeff = {
                idx: 1.0
                for idx, (candidate_source, _) in enumerate(candidates)
                if candidate_source == source
            }
            available = max(0.0, float(observation.nodes[source]["inventory"]))
            rows.append((coeff, -np.inf, available))

        matrix = lil_matrix((len(rows), n_vars), dtype=float)
        lb = np.empty(len(rows), dtype=float)
        ub = np.empty(len(rows), dtype=float)
        for row_idx, (coeff, row_lb, row_ub) in enumerate(rows):
            for col_idx, value in coeff.items():
                matrix[row_idx, col_idx] = value
            lb[row_idx] = row_lb
            ub[row_idx] = row_ub

        result = milp(
            c=c,
            integrality=np.concatenate([np.ones(n_ship, dtype=int), np.zeros(n_short, dtype=int)]),
            bounds=Bounds(np.zeros(n_vars), upper),
            constraints=LinearConstraint(matrix.tocsr(), lb, ub),
            options={"time_limit": 2.0},
        )
        if result.x is None:
            return ActionPlan.hold("milp infeasible")

        actions: list[Action] = []
        for idx, (source, target) in enumerate(candidates):
            quantity = float(result.x[idx])
            if quantity < 0.5:
                continue
            actions.append(
                Action(
                    ActionType.SHIP,
                    source,
                    target,
                    quantity,
                    {
                        "final_target": target,
                        "optimizer": "rolling_horizon_milp",
                        "planning_horizon": self.planning_horizon,
                    },
                )
            )
        return ActionPlan(actions or [Action.hold("milp chose no shipment")], {"agent": self.name})
