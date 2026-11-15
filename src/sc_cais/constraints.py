"""Runtime constraints used by the SC-CAIS assurance kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sc_cais.models import Action, ActionType, ConstraintSeverity, ConstraintViolation

if TYPE_CHECKING:
    from sc_cais.environment import SupplyChainEnv


class Constraint(Protocol):
    @property
    def name(self) -> str: ...

    def check(
        self,
        env: SupplyChainEnv,
        action: Action,
    ) -> ConstraintViolation | None: ...


@dataclass(frozen=True, slots=True)
class ValidActionConstraint:
    name: str = "valid_action"

    def check(self, env: SupplyChainEnv, action: Action) -> ConstraintViolation | None:
        if action.action_type == ActionType.HOLD:
            return None
        if action.source is None or action.target is None:
            return ConstraintViolation(
                self.name,
                ConstraintSeverity.HARD,
                "source and target required",
                False,
            )
        if action.source not in env.network.graph or action.target not in env.network.graph:
            return ConstraintViolation(
                self.name,
                ConstraintSeverity.HARD,
                "unknown source or target",
                False,
            )
        if action.quantity <= 0:
            return ConstraintViolation(
                self.name,
                ConstraintSeverity.HARD,
                "quantity must be positive",
                False,
            )
        return None


@dataclass(frozen=True, slots=True)
class RouteAvailabilityConstraint:
    name: str = "route_availability"

    def check(self, env: SupplyChainEnv, action: Action) -> ConstraintViolation | None:
        if action.action_type == ActionType.HOLD or action.source is None or action.target is None:
            return None
        if not env.network.graph.has_edge(action.source, action.target):
            return ConstraintViolation(
                self.name,
                ConstraintSeverity.HARD,
                "requested edge does not exist",
                True,
            )
        edge = env.network.edge(action.source, action.target)
        endpoints_operating = (
            env.network.node(action.source).operating and env.network.node(action.target).operating
        )
        if not edge.active or not endpoints_operating:
            return ConstraintViolation(
                self.name,
                ConstraintSeverity.HARD,
                "route or endpoint unavailable",
                True,
            )
        return None


@dataclass(frozen=True, slots=True)
class InventoryConstraint:
    name: str = "inventory_nonnegative"

    def check(self, env: SupplyChainEnv, action: Action) -> ConstraintViolation | None:
        if action.action_type == ActionType.HOLD or action.source is None:
            return None
        source = env.network.node(action.source)
        if action.quantity > source.inventory + 1e-9:
            return ConstraintViolation(
                self.name,
                ConstraintSeverity.HARD,
                f"requested {action.quantity:.3f} exceeds source inventory {source.inventory:.3f}",
                True,
            )
        return None


@dataclass(frozen=True, slots=True)
class CapacityConstraint:
    name: str = "transport_capacity"

    def check(self, env: SupplyChainEnv, action: Action) -> ConstraintViolation | None:
        if action.action_type == ActionType.HOLD or action.source is None or action.target is None:
            return None
        if not env.network.graph.has_edge(action.source, action.target):
            return None
        edge = env.network.edge(action.source, action.target)
        target = env.network.node(action.target)
        max_q = min(edge.capacity, target.capacity)
        if action.quantity > max_q + 1e-9:
            return ConstraintViolation(
                self.name,
                ConstraintSeverity.HARD,
                f"requested {action.quantity:.3f} exceeds safe transfer capacity {max_q:.3f}",
                True,
            )
        return None


@dataclass(frozen=True, slots=True)
class DataTrustConstraint:
    threshold: float = 0.5
    name: str = "data_trust"

    def check(self, env: SupplyChainEnv, action: Action) -> ConstraintViolation | None:
        if action.action_type == ActionType.HOLD or action.source is None or action.target is None:
            return None
        obs = env.observe()
        s = float(obs.nodes[action.source]["trust"])
        t = float(obs.nodes[action.target]["trust"])
        edge_key = f"{action.source}->{action.target}"
        e = float(obs.edges.get(edge_key, {}).get("trust", 1.0))
        if min(s, t, e) < self.threshold:
            return ConstraintViolation(
                self.name,
                ConstraintSeverity.CYBER,
                f"decision evidence trust {min(s, t, e):.3f} is below {self.threshold:.3f}",
                True,
            )
        return None


@dataclass(frozen=True, slots=True)
class AuthorityConstraint:
    max_quantity: float = 45.0
    name: str = "authority_limit"

    def check(self, env: SupplyChainEnv, action: Action) -> ConstraintViolation | None:
        del env
        if action.action_type != ActionType.HOLD and action.quantity > self.max_quantity + 1e-9:
            return ConstraintViolation(
                self.name,
                ConstraintSeverity.GOVERNANCE,
                (
                    f"autonomous quantity {action.quantity:.3f} exceeds "
                    f"authority {self.max_quantity:.3f}"
                ),
                True,
            )
        return None


def default_constraints() -> list[Constraint]:
    return [
        ValidActionConstraint(),
        RouteAvailabilityConstraint(),
        InventoryConstraint(),
        CapacityConstraint(),
        DataTrustConstraint(),
        AuthorityConstraint(),
    ]
