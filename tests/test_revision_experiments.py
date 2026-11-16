"""Regression tests for deployment-limited runtime assurance experiments."""

from __future__ import annotations

from sc_cais.disruptions import FalseInventorySignal
from sc_cais.environment import SupplyChainEnv
from sc_cais.models import Action, ActionPlan, ActionType
from sc_cais.network import create_reference_network
from sc_cais.revision_experiments import EvidenceLimitedAssurance, run_campaign
from sc_cais.scenarios import Scenario
from sc_cais.visibility import from_observation


def test_observation_view_does_not_alias_true_network() -> None:
    env = SupplyChainEnv(create_reference_network())
    observed = from_observation(env.observe())
    original = observed.network.node("H1").inventory
    env.network.node("H1").inventory = 1.0
    assert observed.network.node("H1").inventory == original


def test_disruption_visible_to_policy_before_its_first_proposal() -> None:
    env = SupplyChainEnv(
        create_reference_network(),
        disruptions=(FalseInventorySignal("H1", 0, 1, 4.0, 0.8),),
        demand_noise=0.0,
    )

    class RecordingAgent:
        name = "recording"

        def __init__(self) -> None:
            self.inventories: list[float] = []

        def act(self, observation, environment) -> ActionPlan:
            del environment
            self.inventories.append(float(observation.nodes["H1"]["inventory"]))
            return ActionPlan.hold("observation test")

    agent = RecordingAgent()
    env.run(agent, EvidenceLimitedAssurance(), 1)
    assert agent.inventories == [260.0]
    assert env.step_index == 1


def test_campaign_returns_audited_modes_without_modifying_frozen_outputs() -> None:
    summary, actions = run_campaign(
        seeds=(0,),
        horizon=3,
        scenarios=[Scenario("normal", (), "baseline", "none")],
    )
    assert len(summary) == 12  # three proposer policies x four assurance configurations
    assert len(actions) > 0
    assert set(summary["kernel_view"]) == {"oracle", "observed"}
    assert set(summary["repair_mode"]) == {"rule_repair", "reject_only", "quantity_clip"}
    assert summary["physical_command_violation_rate"].between(0.0, 1.0).all()
    assert actions["executed_true_violation"].isin([True, False]).all()


def test_covert_high_trust_attack_exposes_observed_kernel_limit() -> None:
    env = SupplyChainEnv(
        create_reference_network(),
        disruptions=(FalseInventorySignal("H1", 0, 1, 4.0, 0.8),),
        demand_noise=0.0,
    )
    env.network.node("H1").inventory = 1.0
    env.begin_step()
    proposal = Action(ActionType.SHIP, "H1", "C1", 3.0)

    observed = EvidenceLimitedAssurance(view="observed")
    result = observed.decide(env, proposal)
    assert result.executed_action.quantity == 3.0
    assert observed.audit_records[0]["executed_true_violation"] is True
    assert observed.audit_records[0]["executed_true_constraints"] == "inventory_nonnegative"

    oracle = EvidenceLimitedAssurance(view="oracle")
    result_oracle = oracle.decide(env, proposal)
    assert result_oracle.executed_action.quantity <= 1.0
    assert oracle.audit_records[0]["executed_true_violation"] is False


def test_on_time_service_distinguishes_backlog_clearance() -> None:
    env = SupplyChainEnv(
        create_reference_network(), demand_noise=0.0, service_accounting="fifo_on_time"
    )
    customer = env.network.node("C1")
    customer.backlog = 30.0
    customer.inventory = 40.0
    demand, fulfilled, _ = env._demand_and_fulfilment()
    assert demand == 47.0
    assert fulfilled >= 40.0
    assert env._current_demand_fulfilled == 10.0 + 22.0
    assert customer.backlog == 15.0


def test_component_ablations_and_action_order_campaign() -> None:
    scenario = Scenario("normal", (), "baseline", "none")
    normal, _ = run_campaign(seeds=(0,), horizon=3, scenarios=[scenario], include_ablations=True)
    reversed_plan, _ = run_campaign(
        seeds=(0,), horizon=3, scenarios=[scenario], action_order="reverse"
    )
    assert len(normal) == 24
    assert len(reversed_plan) == 12
    assert {"no_trust", "no_authority", "no_route_repair", "no_quantity_clipping"}.issubset(
        set(normal["repair_mode"])
    )
    assert (normal["service_accounting"] == "fifo_on_time").all()
    assert (reversed_plan["action_order"] == "reverse").all()


def test_component_ablation_removes_threshold_from_repair_and_check() -> None:
    env = SupplyChainEnv(create_reference_network(), demand_noise=0.0)
    env.observation_trust_override["H1"] = 0.1
    proposal = Action(ActionType.SHIP, "H1", "C1", 5.0)
    full = EvidenceLimitedAssurance(view="observed", repair_mode="rule_repair")
    no_trust = EvidenceLimitedAssurance(view="observed", repair_mode="no_trust")
    full_decision = full.decide(env, proposal)
    no_trust_decision = no_trust.decide(env, proposal)
    assert full_decision.decision != no_trust_decision.decision
    assert no_trust_decision.executed_action.source == "H1"
    assert no_trust_decision.executed_action.target == "C1"

    authority_only = Action(ActionType.SHIP, "S1", "P1", 46.0)
    full2 = EvidenceLimitedAssurance(view="observed", repair_mode="rule_repair")
    no_authority = EvidenceLimitedAssurance(view="observed", repair_mode="no_authority")
    assert full2.decide(env, authority_only).executed_action.quantity <= 45.0
    assert no_authority.decide(env, authority_only).executed_action.quantity == 46.0


def test_all_assurance_modes_record_positive_wall_clock_latency() -> None:
    from sc_cais.models import Action, ActionType

    modes = (
        "rule_repair", "monitor_only", "reject_only", "quantity_clip",
        "no_trust", "no_authority", "no_route_repair", "no_quantity_clipping",
    )
    env = SupplyChainEnv(create_reference_network(), demand_noise=0.0)
    proposal = Action(ActionType.SHIP, "H1", "C1", 8.0)
    for mode in modes:
        checker = EvidenceLimitedAssurance(view="observed", repair_mode=mode)
        outcome = checker.decide(env, proposal)
        assert outcome.latency_ms > 0.0, mode
        assert len(checker.audit_records) == 1
        assert checker.audit_records[0]["assurance_latency_ms"] == outcome.latency_ms
