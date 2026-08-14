"""Core typed models used across SC-CAIS."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    SUPPLIER = "supplier"
    PLANT = "plant"
    WAREHOUSE = "warehouse"
    HUB = "hub"
    CUSTOMER = "customer"


class ActionType(StrEnum):
    HOLD = "hold"
    SHIP = "ship"
    REROUTE = "reroute"
    REALLOCATE = "reallocate_inventory"
    EXPEDITE = "expedite_shipment"
    CHANGE_SUPPLIER = "change_supplier"
    SET_PRODUCTION = "set_production"
    PRIORITIZE = "prioritize_order"


class AssuranceDecision(StrEnum):
    APPROVE = "approve"
    REPAIR = "repair"
    FALLBACK = "fallback"


class ConstraintSeverity(StrEnum):
    HARD = "hard"
    RESILIENCE = "resilience"
    CYBER = "cyber"
    GOVERNANCE = "governance"
    SOFT = "soft"


@dataclass(slots=True)
class NodeState:
    node_id: str
    node_type: NodeType
    inventory: float = 0.0
    capacity: float = 100.0
    backlog: float = 0.0
    operating: bool = True
    trust: float = 1.0
    baseline_demand: float = 0.0
    holding_cost: float = 0.05
    backlog_cost: float = 2.0


@dataclass(slots=True)
class EdgeState:
    source: str
    target: str
    capacity: float = 50.0
    lead_time: int = 1
    cost: float = 1.0
    active: bool = True
    trust: float = 1.0


@dataclass(slots=True)
class Shipment:
    source: str
    target: str
    quantity: float
    eta: int
    unit_cost: float


@dataclass(slots=True)
class Action:
    action_type: ActionType
    source: str | None = None
    target: str | None = None
    quantity: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def hold(cls, reason: str = "") -> Action:
        return cls(ActionType.HOLD, metadata={"reason": reason})


@dataclass(slots=True)
class ActionPlan:
    actions: list[Action] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def hold(cls, reason: str = "") -> ActionPlan:
        return cls([Action.hold(reason)])


@dataclass(slots=True)
class Observation:
    step: int
    nodes: dict[str, dict[str, float | bool | str]]
    edges: dict[str, dict[str, float | bool | int | str]]
    total_backlog: float
    service_level: float
    data_quality: float


@dataclass(slots=True)
class ConstraintViolation:
    constraint: str
    severity: ConstraintSeverity
    message: str
    repairable: bool


@dataclass(slots=True)
class AssuranceOutcome:
    proposed_action: Action
    executed_action: Action
    decision: AssuranceDecision
    violations: list[ConstraintViolation] = field(default_factory=list)
    residual_violations: list[ConstraintViolation] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass(slots=True)
class StepResult:
    step: int
    proposed_plan: ActionPlan
    outcomes: list[AssuranceOutcome]
    demand: float
    fulfilled: float
    service_level: float
    instant_service_level: float
    total_backlog: float
    step_cost: float
    cumulative_cost: float
    data_quality: float

    @property
    def proposed_action(self) -> Action:
        if self.proposed_plan.actions:
            return self.proposed_plan.actions[0]
        return Action.hold("empty plan")

    @property
    def outcome(self) -> AssuranceOutcome:
        if self.outcomes:
            return self.outcomes[0]
        hold = Action.hold("empty plan")
        return AssuranceOutcome(hold, hold, AssuranceDecision.APPROVE)
