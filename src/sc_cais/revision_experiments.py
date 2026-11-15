"""Editor-requested evidence-limited SC-CAIS evaluation (separate from frozen v0.3).

The physical environment is accessible to the *offline auditor*, never to the
observed-only kernel. The audit measures invalid commands before the simulator's
physical saturation, not physical inventory deficits after saturation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import pandas as pd

from sc_cais.agents import AgenticHeuristicAgent, RollingHorizonMILPAgent, RuleBasedAgent
from sc_cais.assurance import RuntimeAssurance
from sc_cais.constraints import (
    AuthorityConstraint,
    DataTrustConstraint,
    CapacityConstraint,
    InventoryConstraint,
    RouteAvailabilityConstraint,
    ValidActionConstraint,
)
from sc_cais.disruptions import FalseInventorySignal
from sc_cais.environment import SupplyChainEnv
from sc_cais.evaluation import summarize
from sc_cais.models import Action, ActionType, AssuranceOutcome
from sc_cais.network import create_reference_network
from sc_cais.scenarios import Scenario, default_scenarios, disruption_window
from sc_cais.visibility import from_observation

KernelView = Literal["oracle", "observed"]
RepairMode = Literal[
    "rule_repair",
    "reject_only",
    "quantity_clip",
    "no_trust",
    "no_authority",
    "no_route_repair",
    "no_quantity_clipping",
]

# This independent evaluator has privileged state access *only for evaluation*.
# The kernel's trust check is not counted as a physical violation: reported
# trust scores are themselves attacker-controlled in the covert-attack test.
PHYSICAL_CONSTRAINTS = (
    ValidActionConstraint(),
    RouteAvailabilityConstraint(),
    InventoryConstraint(),
    CapacityConstraint(),
    AuthorityConstraint(),
)


@dataclass(slots=True)
class EvidenceLimitedAssurance:
    view: KernelView = "observed"
    repair_mode: RepairMode = "rule_repair"
    audit_records: list[dict[str, object]] = field(default_factory=list)
    _kernel: RuntimeAssurance = field(init=False)

    def __post_init__(self) -> None:
        from sc_cais.constraints import default_constraints

        constraints = default_constraints()
        if self.repair_mode == "no_trust":
            constraints = [c for c in constraints if not isinstance(c, DataTrustConstraint)]
        if self.repair_mode == "no_authority":
            constraints = [c for c in constraints if not isinstance(c, AuthorityConstraint)]
        self._kernel = RuntimeAssurance(
            constraints=constraints, repair_enabled=self.repair_mode != "reject_only"
        )

    def _quantity_clip(self, state: object, proposed: Action) -> Action | None:
        """Simple feasible-clipping comparator; no route/source substitution."""
        if (
            proposed.action_type == ActionType.HOLD
            or proposed.source is None
            or proposed.target is None
        ):
            return None
        network = state.network  # type: ignore[attr-defined]
        if (
            proposed.source not in network.graph
            or proposed.target not in network.graph
            or not network.graph.has_edge(proposed.source, proposed.target)
        ):
            return None
        source = network.node(proposed.source)
        target = network.node(proposed.target)
        edge = network.edge(proposed.source, proposed.target)
        if not (source.operating and target.operating and edge.active):
            return None
        cap = min(proposed.quantity, source.inventory, target.capacity, edge.capacity, 45.0)
        if cap <= 1e-9:
            return None
        return replace(proposed, quantity=cap)

    def decide(self, env: SupplyChainEnv, proposed: Action) -> AssuranceOutcome:
        # The restricted kernel is passed only a reconstructed state made from
        # the snapshot. It cannot reach env.network or the true shipment ledger.
        state = env if self.view == "oracle" else from_observation(env.observe())
        if self.repair_mode == "quantity_clip":
            from sc_cais.models import AssuranceDecision

            original = self._kernel._violations(state, proposed)
            if not original:
                decision = AssuranceOutcome(proposed, proposed, AssuranceDecision.APPROVE, [], [])
            else:
                clipped = self._quantity_clip(state, proposed)
                if clipped is not None and not self._kernel._violations(state, clipped):
                    decision = AssuranceOutcome(
                        proposed, clipped, AssuranceDecision.REPAIR, original, []
                    )
                else:
                    decision = AssuranceOutcome(
                        proposed,
                        Action.hold("quantity clipping failed"),
                        AssuranceDecision.FALLBACK,
                        original,
                        [],
                    )
        else:
            # Component ablations preserve the complete independent physical auditor.
            # The restricted strategies must not silently fall back to unrestricted repair.
            if self.repair_mode in {"no_route_repair", "no_quantity_clipping"}:
                from sc_cais.models import AssuranceDecision

                original = self._kernel._violations(state, proposed)
                if not original:
                    decision = AssuranceOutcome(
                        proposed, proposed, AssuranceDecision.APPROVE, [], []
                    )
                else:
                    candidate = self._kernel._repair(state, proposed)
                    if candidate is not None:
                        if self.repair_mode == "no_route_repair" and (
                            candidate.source != proposed.source
                            or candidate.target != proposed.target
                        ):
                            candidate = None
                        if self.repair_mode == "no_quantity_clipping" and (
                            candidate is not None and candidate.quantity < proposed.quantity - 1e-9
                        ):
                            candidate = None
                    if candidate is not None and not self._kernel._violations(state, candidate):
                        decision = AssuranceOutcome(
                            proposed, candidate, AssuranceDecision.REPAIR, original, []
                        )
                    else:
                        decision = AssuranceOutcome(
                            proposed,
                            Action.hold("component ablation fallback"),
                            AssuranceDecision.FALLBACK,
                            original,
                            [],
                        )
            else:
                decision = self._kernel.decide(state, proposed)

        def physical_violations(action: Action) -> list[str]:
            # Invalid identifiers must not be forwarded to state-dependent checks.
            invalid = PHYSICAL_CONSTRAINTS[0].check(env, action)
            if invalid is not None:
                return [invalid.constraint]
            return [
                v.constraint
                for constraint in PHYSICAL_CONSTRAINTS[1:]
                if (v := constraint.check(env, action)) is not None
            ]

        proposed_true = physical_violations(proposed)
        executed_true = physical_violations(decision.executed_action)
        final_action = decision.executed_action
        self.audit_records.append(
            {
                "step": env.step_index,
                "decision": str(decision.decision),
                "proposed_true_violation": bool(proposed_true),
                "executed_true_violation": bool(executed_true),
                "proposed_true_constraints": ";".join(proposed_true),
                "executed_true_constraints": ";".join(executed_true),
                "proposed_quantity": proposed.quantity,
                "executed_quantity": final_action.quantity,
                "quantity_deviation": abs(final_action.quantity - proposed.quantity),
                "source_changed": proposed.source != final_action.source,
                "target_changed": proposed.target != final_action.target,
                "fallback": final_action.action_type == ActionType.HOLD,
                "proposed_route": f"{proposed.source}->{proposed.target}",
                "executed_route": f"{final_action.source}->{final_action.target}",
            }
        )
        return decision


def revision_scenarios() -> list[Scenario]:
    names = {
        "normal",
        "false_inventory",
        "telemetry_loss",
        "route_spoofing",
        "compound",
        "cyber_physical",
    }
    selected = [scenario for scenario in default_scenarios() if scenario.name in names]
    selected.append(
        Scenario(
            "covert_false_inventory_trust_080",
            (FalseInventorySignal("H1", 5, 15, multiplier=4.0, trust=0.8),),
            "cyber",
            "high",
        )
    )
    return selected


def run_campaign(
    *,
    seeds: tuple[int, ...] = (0, 1),
    horizon: int = 20,
    scenarios: list[Scenario] | None = None,
    output_dir: str | Path | None = None,
    include_ablations: bool = False,
    action_order: Literal["native", "reverse"] = "native",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = (RuleBasedAgent, RollingHorizonMILPAgent, AgenticHeuristicAgent)
    modes: tuple[tuple[KernelView, RepairMode], ...] = (
        ("oracle", "rule_repair"),
        ("observed", "rule_repair"),
        ("observed", "reject_only"),
        ("observed", "quantity_clip"),
    )
    if include_ablations:
        modes += (
            ("observed", "no_trust"),
            ("observed", "no_authority"),
            ("observed", "no_route_repair"),
            ("observed", "no_quantity_clipping"),
        )
    summaries: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    for scenario in scenarios if scenarios is not None else revision_scenarios():
        _, disruption_end = disruption_window(scenario)
        for seed in seeds:
            for policy_type in policies:
                for view, repair_mode in modes:
                    env = SupplyChainEnv(
                        create_reference_network(),
                        seed=seed,
                        disruptions=scenario.disruptions,
                        demand_noise=0.08,
                        observation_noise=0.0,
                        service_accounting="fifo_on_time",
                    )
                    policy = policy_type()
                    assurance = EvidenceLimitedAssurance(view=view, repair_mode=repair_mode)
                    if action_order == "native":
                        results = env.run(policy, assurance, horizon)
                    else:
                        results = []
                        for _ in range(horizon):
                            env.begin_step()
                            plan = policy.act(env.observe(), env)
                            plan.actions.reverse()
                            results.append(env.step(plan, assurance))
                    records = assurance.audit_records
                    n = len(records)
                    metrics = summarize(results, disruption_end=disruption_end)
                    identity = {
                        "scenario": scenario.name,
                        "seed": seed,
                        "policy": policy.name,
                        "kernel_view": view,
                        "repair_mode": repair_mode,
                        "horizon": horizon,
                        "action_order": action_order,
                        "service_accounting": "fifo_on_time",
                        "n_actions": n,
                    }
                    summaries.append(
                        {
                            **identity,
                            **metrics,
                            "physical_command_violation_rate": (
                                sum(bool(a["executed_true_violation"]) for a in records) / n
                                if n
                                else 0.0
                            ),
                            "proposed_physical_violation_rate": (
                                sum(bool(a["proposed_true_violation"]) for a in records) / n
                                if n
                                else 0.0
                            ),
                            "mean_quantity_deviation": (
                                sum(float(a["quantity_deviation"]) for a in records) / n
                                if n
                                else 0.0
                            ),
                            "route_change_rate": (
                                sum(
                                    bool(a["source_changed"]) or bool(a["target_changed"])
                                    for a in records
                                )
                                / n
                                if n
                                else 0.0
                            ),
                        }
                    )
                    actions.extend({**identity, **record} for record in records)
    summary_frame = pd.DataFrame(summaries)
    action_frame = pd.DataFrame(actions)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_frame.to_csv(out / "revision_summary.csv", index=False)
        action_frame.to_csv(out / "revision_actions.csv", index=False)
    return summary_frame, action_frame
