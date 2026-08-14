from sc_cais.agents import CostAwareOptimizerAgent
from sc_cais.environment import SupplyChainEnv
from sc_cais.models import ActionType
from sc_cais.network import create_reference_network


def test_cost_aware_optimizer_returns_valid_plan():
    env = SupplyChainEnv(create_reference_network(), seed=0)
    plan = CostAwareOptimizerAgent().act(env.observe(), env)
    assert plan.actions
    assert all(a.action_type in {ActionType.SHIP, ActionType.HOLD} for a in plan.actions)
