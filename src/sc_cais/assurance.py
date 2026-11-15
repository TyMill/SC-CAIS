"""Runtime assurance with approve, minimal-deviation repair and fallback."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

from sc_cais.constraints import Constraint, default_constraints
from sc_cais.models import (
    Action,
    ActionType,
    AssuranceDecision,
    AssuranceOutcome,
    ConstraintViolation,
)

if TYPE_CHECKING:
    from sc_cais.environment import SupplyChainEnv


class AssurancePolicy(Protocol):
    def decide(self, env: SupplyChainEnv, proposed: Action) -> AssuranceOutcome: ...


class PassthroughAssurance:
    """Monitor-only baseline: record violations but never block the agent."""

    def __init__(self, constraints: list[Constraint] | None = None) -> None:
        self.constraints = constraints or default_constraints()

    def decide(self, env: SupplyChainEnv, proposed: Action) -> AssuranceOutcome:
        start = perf_counter()
        violations = [v for c in self.constraints if (v := c.check(env, proposed)) is not None]
        return AssuranceOutcome(
            proposed,
            proposed,
            AssuranceDecision.APPROVE,
            violations,
            violations,
            (perf_counter() - start) * 1000,
        )


class RuntimeAssurance:
    """Policy-agnostic action shield for controlled agentic autonomy."""

    def __init__(
        self,
        constraints: list[Constraint] | None = None,
        *,
        repair_enabled: bool = True,
    ) -> None:
        self.constraints = constraints or default_constraints()
        self.repair_enabled = repair_enabled

    def _violations(
        self,
        env: SupplyChainEnv,
        action: Action,
    ) -> list[ConstraintViolation]:
        return [
            violation
            for constraint in self.constraints
            if (violation := constraint.check(env, action)) is not None
        ]

    def _repair(self, env: SupplyChainEnv, action: Action) -> Action | None:
        if action.action_type == ActionType.HOLD or action.source is None or action.target is None:
            return None
        if action.source not in env.network.graph or action.target not in env.network.graph:
            return None

        source_id = action.source
        target_id = action.target
        observation = env.observe()
        # A disabled component must also disappear from the repair rules.
        # Otherwise ablation would still enforce its threshold indirectly.
        trust_threshold = float("-inf")
        authority_limit = float("inf")
        for constraint in self.constraints:
            if getattr(constraint, "name", "") == "data_trust":
                trust_threshold = float(getattr(constraint, "threshold", trust_threshold))
            if getattr(constraint, "name", "") == "authority_limit":
                authority_limit = float(getattr(constraint, "max_quantity", authority_limit))

        def trusted_edge(source: str, target: str) -> bool:
            key = f"{source}->{target}"
            if key not in observation.edges:
                return False
            return (
                min(
                    float(observation.nodes[source]["trust"]),
                    float(observation.nodes[target]["trust"]),
                    float(observation.edges[key]["trust"]),
                )
                >= trust_threshold
            )

        route_bad = (
            not env.network.graph.has_edge(source_id, target_id)
            or not env.network.edge(source_id, target_id).active
            or not env.network.node(source_id).operating
            or not env.network.node(target_id).operating
        )
        trust_bad = not route_bad and not trusted_edge(source_id, target_id)

        if trust_bad:
            # Prefer preserving the intended target by switching to a trusted source.
            alt_sources = [
                source
                for source in env.network.active_predecessors(target_id)
                if trusted_edge(source, target_id) and env.network.node(source).inventory > 1e-9
            ]
            if alt_sources:
                source_id = max(alt_sources, key=lambda n: env.network.node(n).inventory)
            else:
                # Otherwise preserve the source and divert to a trusted equivalent echelon.
                target_type = env.network.node(target_id).node_type
                alt_targets = [
                    target
                    for target in env.network.active_successors(source_id)
                    if (
                        env.network.node(target).node_type == target_type
                        and trusted_edge(source_id, target)
                    )
                ]
                if alt_targets:
                    target_id = min(alt_targets, key=lambda n: env.network.edge(source_id, n).cost)
                else:
                    return None

        route_bad = (
            not env.network.graph.has_edge(source_id, target_id)
            or not env.network.edge(source_id, target_id).active
            or not env.network.node(source_id).operating
            or not env.network.node(target_id).operating
        )
        if route_bad:
            final_target = str(action.metadata.get("final_target", target_id))
            candidates = []
            for neighbor in env.network.active_successors(source_id):
                if neighbor == final_target or env.network.has_active_path(neighbor, final_target):
                    candidates.append(neighbor)
            if not candidates:
                return None
            target_id = min(candidates, key=lambda n: env.network.edge(source_id, n).cost)

        if not trusted_edge(source_id, target_id):
            return None
        source = env.network.node(source_id)
        edge = env.network.edge(source_id, target_id)
        target_node = env.network.node(target_id)
        safe_quantity = min(
            action.quantity,
            source.inventory,
            edge.capacity,
            target_node.capacity,
            authority_limit,
        )
        if safe_quantity <= 1e-9:
            return None
        return replace(
            action,
            source=source_id,
            target=target_id,
            quantity=safe_quantity,
            metadata={
                **action.metadata,
                "repaired": True,
                "original_source": action.source,
                "original_target": action.target,
            },
        )

    def decide(self, env: SupplyChainEnv, proposed: Action) -> AssuranceOutcome:
        start = perf_counter()
        violations = self._violations(env, proposed)
        if not violations:
            return AssuranceOutcome(
                proposed,
                proposed,
                AssuranceDecision.APPROVE,
                [],
                [],
                (perf_counter() - start) * 1000,
            )

        if self.repair_enabled and all(v.repairable for v in violations):
            repaired = self._repair(env, proposed)
            if repaired is not None and not self._violations(env, repaired):
                return AssuranceOutcome(
                    proposed,
                    repaired,
                    AssuranceDecision.REPAIR,
                    violations,
                    [],
                    (perf_counter() - start) * 1000,
                )

        fallback = Action.hold("runtime assurance fallback")
        return AssuranceOutcome(
            proposed,
            fallback,
            AssuranceDecision.FALLBACK,
            violations,
            [],
            (perf_counter() - start) * 1000,
        )
