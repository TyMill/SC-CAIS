from sc_cais.disruptions import FalseInventorySignal, HubClosure
from sc_cais.environment import SupplyChainEnv
from sc_cais.network import create_reference_network


def test_false_inventory_changes_observation_not_truth():
    env = SupplyChainEnv(
        create_reference_network(),
        disruptions=[FalseInventorySignal("H1", 0, 2, 4.0)],
    )
    true_inventory = env.network.node("H1").inventory
    env._apply_disruptions()
    obs = env.observe()
    assert obs.nodes["H1"]["inventory"] == true_inventory * 4
    assert env.network.node("H1").inventory == true_inventory


def test_hub_closure_changes_physical_state():
    env = SupplyChainEnv(create_reference_network(), disruptions=[HubClosure("H1", 0, 2)])
    env._apply_disruptions()
    assert env.network.node("H1").operating is False
