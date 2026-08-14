from sc_cais.network import create_reference_network


def test_reference_network_valid():
    net = create_reference_network()
    assert net.validate() == []
    assert net.has_active_path("S1", "C1")
    assert net.shortest_active_path("S1", "C1")[0] == "S1"
