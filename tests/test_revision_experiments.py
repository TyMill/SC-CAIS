"""Regression tests for deployment-limited runtime assurance experiments."""

from __future__ import annotations

from sc_cais.agents import AgenticHeuristicAgent
from sc_cais.disruptions import FalseInventorySignal
from sc_cais.environment import SupplyChainEnv
from sc_cais.models import ActionPlan
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
