from sc_cais.assurance import PassthroughAssurance
from sc_cais.environment import SupplyChainEnv
from sc_cais.models import Action, ActionType, AssuranceDecision
from sc_cais.network import create_reference_network


def test_passthrough_records_but_does_not_block_violation():
    env = SupplyChainEnv(create_reference_network())
    proposed = Action(ActionType.SHIP, "H1", "C1", 120.0)
    out = PassthroughAssurance().decide(env, proposed)
    assert out.decision == AssuranceDecision.APPROVE
    assert out.executed_action.quantity == 120.0
    assert out.violations
