"""Evidence-limited view for deployment-realistic runtime constraint checks.

The observation-only kernel receives *only* the observation snapshot supplied at
this decision boundary. The physical simulator and its true network are not
reachable through this view. Static topology/parameter values are reconstructed
from the same observation, not copied from the hidden physical network.
"""

from __future__ import annotations

from dataclasses import dataclass

from sc_cais.models import EdgeState, NodeState, NodeType, Observation
from sc_cais.network import SupplyChainNetwork


@dataclass(frozen=True, slots=True)
class ObservedStateView:
    network: SupplyChainNetwork
    snapshot: Observation

    def observe(self) -> Observation:
        return self.snapshot


def from_observation(observation: Observation) -> ObservedStateView:
    """Reconstruct the kernel's evidence state with no oracle-state access."""
    network = SupplyChainNetwork()
    for node_id, values in observation.nodes.items():
        network.add_node(
            NodeState(
                node_id=node_id,
                node_type=NodeType(str(values["type"])),
                inventory=float(values["inventory"]),
                capacity=float(values["capacity"]),
                backlog=float(values["backlog"]),
                operating=bool(values["operating"]),
                trust=float(values["trust"]),
                baseline_demand=float(values["baseline_demand"]),
            )
        )
    for edge_values in observation.edges.values():
        network.add_edge(
            EdgeState(
                source=str(edge_values["source"]),
                target=str(edge_values["target"]),
                capacity=float(edge_values["capacity"]),
                lead_time=int(edge_values["lead_time"]),
                cost=float(edge_values["cost"]),
                active=bool(edge_values["active"]),
                trust=float(edge_values["trust"]),
            )
        )
    return ObservedStateView(network=network, snapshot=observation)
