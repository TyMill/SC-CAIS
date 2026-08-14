# SC-CAIS method specification

## Decision pipeline

At time step `t`, an agent receives a potentially corrupted observation `o_t` and proposes `a_t^p`.
The assurance kernel checks the proposal against hard feasibility, cyber-data trust and governance constraints.

- **APPROVE**: execute the proposal unchanged.
- **REPAIR**: compute a minimal-deviation feasible action, preserving intent where possible.
- **FALLBACK**: execute a safe hold action when the proposal cannot be repaired or evidence is untrusted.

## State / observation separation

The simulator maintains true node/edge state separately from observations. Cyber disruptions alter the observation layer without necessarily modifying physical state. This enables false-inventory, telemetry-loss and route-spoofing experiments.

## Main experimental comparison

1. Rule-based decision policy.
2. Agentic heuristic policy without assurance.
3. Agentic heuristic with reject-only assurance.
4. Agentic heuristic with full SC-CAIS repair assurance.

## Primary outcome groups

- performance: cost, service level, backlog;
- resilience: mean/min service level and recovery behaviour;
- assurance: violation attempts, prevention, repair and fallback rates;
- overhead: assurance latency.
