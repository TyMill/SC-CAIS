from sc_cais.assurance import RuntimeAssurance
from sc_cais.environment import SupplyChainEnv
from sc_cais.models import Action, ActionType, AssuranceDecision
from sc_cais.network import create_reference_network


def test_excess_quantity_is_repaired():
    env = SupplyChainEnv(create_reference_network())
    action = Action(ActionType.SHIP, "H1", "C1", 120.0)
    out = RuntimeAssurance().decide(env, action)
    assert out.decision == AssuranceDecision.REPAIR
    assert out.executed_action.quantity <= 32.0


def test_untrusted_source_is_repaired_to_trusted_alternative():
    env = SupplyChainEnv(create_reference_network())
    env.observation_trust_override["H1"] = 0.1
    action = Action(ActionType.SHIP, "H1", "C1", 20.0)
    out = RuntimeAssurance().decide(env, action)
    assert out.decision == AssuranceDecision.REPAIR
    assert out.executed_action.source == "H2"


def test_untrusted_data_without_alternative_forces_fallback():
    env = SupplyChainEnv(create_reference_network())
    env.observation_trust_override["H1"] = 0.1
    env.observation_trust_override["H2"] = 0.1
    action = Action(ActionType.SHIP, "H1", "C1", 20.0)
    out = RuntimeAssurance().decide(env, action)
    assert out.decision == AssuranceDecision.FALLBACK


def test_closed_route_can_be_repaired_to_alternative():
    env = SupplyChainEnv(create_reference_network())
    env.network.edge("H1", "C1").active = False
    action = Action(ActionType.SHIP, "H1", "C1", 20.0, {"final_target": "C2"})
    out = RuntimeAssurance().decide(env, action)
    assert out.decision == AssuranceDecision.REPAIR
    assert out.executed_action.target == "C2"
