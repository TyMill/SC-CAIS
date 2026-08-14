from sc_cais.agents import AgenticHeuristicAgent
from sc_cais.assurance import RuntimeAssurance
from sc_cais.environment import SupplyChainEnv
from sc_cais.network import create_reference_network


def test_environment_runs_deterministically():
    env1 = SupplyChainEnv(create_reference_network(), seed=7)
    env2 = SupplyChainEnv(create_reference_network(), seed=7)
    r1 = env1.run(AgenticHeuristicAgent(), RuntimeAssurance(), 8)
    r2 = env2.run(AgenticHeuristicAgent(), RuntimeAssurance(), 8)
    assert [x.demand for x in r1] == [x.demand for x in r2]
    assert env1.cumulative_cost == env2.cumulative_cost


def test_inventory_never_negative_with_assurance():
    env = SupplyChainEnv(create_reference_network(), seed=2)
    env.run(AgenticHeuristicAgent(aggressiveness=4), RuntimeAssurance(), 20)
    assert all(node.inventory >= -1e-9 for node in env.network.nodes())
