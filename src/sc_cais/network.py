"""Supply-chain network representation and reference topology."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import networkx as nx

from sc_cais.models import EdgeState, NodeState, NodeType


class SupplyChainNetwork:
    """Directed supply-chain graph with mutable node and edge states."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_node(self, state: NodeState) -> None:
        self.graph.add_node(state.node_id, state=state)

    def add_edge(self, state: EdgeState) -> None:
        if state.source not in self.graph or state.target not in self.graph:
            raise ValueError("Both edge endpoints must exist before adding an edge")
        self.graph.add_edge(state.source, state.target, state=state)

    def node(self, node_id: str) -> NodeState:
        return cast(NodeState, self.graph.nodes[node_id]["state"])

    def edge(self, source: str, target: str) -> EdgeState:
        return cast(EdgeState, self.graph.edges[source, target]["state"])

    def nodes(self) -> list[NodeState]:
        return [self.graph.nodes[n]["state"] for n in self.graph.nodes]

    def edges(self) -> list[EdgeState]:
        return [self.graph.edges[u, v]["state"] for u, v in self.graph.edges]

    def active_successors(self, node_id: str) -> list[str]:
        return [
            target
            for target in self.graph.successors(node_id)
            if self.edge(node_id, target).active and self.node(target).operating
        ]

    def active_predecessors(self, node_id: str) -> list[str]:
        return [
            source
            for source in self.graph.predecessors(node_id)
            if self.edge(source, node_id).active and self.node(source).operating
        ]

    def has_active_path(self, source: str, target: str) -> bool:
        active = nx.DiGraph()
        active.add_nodes_from(n.node_id for n in self.nodes() if n.operating)
        active.add_edges_from(
            (e.source, e.target)
            for e in self.edges()
            if e.active and self.node(e.source).operating and self.node(e.target).operating
        )
        return source in active and target in active and nx.has_path(active, source, target)

    def shortest_active_path(self, source: str, target: str) -> list[str] | None:
        active = nx.DiGraph()
        for n in self.nodes():
            if n.operating:
                active.add_node(n.node_id)
        for e in self.edges():
            if e.active and e.source in active and e.target in active:
                active.add_edge(e.source, e.target, weight=e.cost)
        try:
            return cast(list[str], nx.shortest_path(active, source, target, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def clone(self) -> SupplyChainNetwork:
        cloned = SupplyChainNetwork()
        for node in self.nodes():
            cloned.add_node(replace(node))
        for edge in self.edges():
            cloned.add_edge(replace(edge))
        return cloned

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not nx.is_directed_acyclic_graph(self.graph):
            errors.append("Reference supply-chain topology must be a directed acyclic graph")
        if not any(n.node_type == NodeType.SUPPLIER for n in self.nodes()):
            errors.append("At least one supplier is required")
        if not any(n.node_type == NodeType.CUSTOMER for n in self.nodes()):
            errors.append("At least one customer is required")
        for n in self.nodes():
            if n.capacity < 0 or n.inventory < 0:
                errors.append(f"Node {n.node_id} has negative capacity or inventory")
        for e in self.edges():
            if e.capacity <= 0 or e.lead_time < 1 or e.cost < 0:
                errors.append(f"Edge {e.source}->{e.target} has invalid parameters")
        return errors


def create_reference_network() -> SupplyChainNetwork:
    """Create a deterministic multi-echelon network used by examples and tests."""
    net = SupplyChainNetwork()
    nodes = [
        NodeState("S1", NodeType.SUPPLIER, inventory=900, capacity=80),
        NodeState("S2", NodeType.SUPPLIER, inventory=220, capacity=45),
        NodeState("P1", NodeType.PLANT, inventory=110, capacity=85),
        NodeState("W1", NodeType.WAREHOUSE, inventory=95, capacity=65),
        NodeState("W2", NodeType.WAREHOUSE, inventory=85, capacity=65),
        NodeState("H1", NodeType.HUB, inventory=65, capacity=55),
        NodeState("H2", NodeType.HUB, inventory=65, capacity=55),
        NodeState("C1", NodeType.CUSTOMER, inventory=35, capacity=45, baseline_demand=25),
        NodeState("C2", NodeType.CUSTOMER, inventory=32, capacity=45, baseline_demand=22),
    ]
    for node in nodes:
        net.add_node(node)
    edges = [
        EdgeState("S1", "P1", 55, 1, 0.8),
        EdgeState("S2", "P1", 35, 1, 0.9),
        EdgeState("P1", "W1", 42, 1, 1.0),
        EdgeState("P1", "W2", 42, 1, 1.1),
        EdgeState("W1", "H1", 36, 1, 1.0),
        EdgeState("W1", "H2", 28, 2, 1.3),
        EdgeState("W2", "H1", 28, 2, 1.3),
        EdgeState("W2", "H2", 36, 1, 1.0),
        EdgeState("H1", "C1", 32, 1, 0.7),
        EdgeState("H1", "C2", 24, 1, 0.9),
        EdgeState("H2", "C1", 24, 1, 0.9),
        EdgeState("H2", "C2", 32, 1, 0.7),
    ]
    for edge in edges:
        net.add_edge(edge)
    errors = net.validate()
    if errors:
        raise RuntimeError("Invalid reference network: " + "; ".join(errors))
    return net
