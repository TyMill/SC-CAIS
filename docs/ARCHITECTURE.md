# Architecture

SC-CAIS separates five concerns:

1. **Physical environment** — true node, edge, shipment, inventory and backlog state.
2. **Observation layer** — what the decision policy perceives, including corrupted or missing telemetry.
3. **Decision policy** — produces a multi-action replenishment/reconfiguration plan.
4. **Runtime assurance** — monitors every proposed action sequentially and chooses approve, repair or fallback.
5. **Evaluation layer** — records operational performance, resilience, attempted violations, residual violations and assurance overhead.

## Why sequential plan assurance?

A plan can contain multiple actions from the same source. Each action is checked after earlier approved actions have reserved/removed inventory, preventing aggregate plan quantities from bypassing per-action feasibility checks.

## Monitor-only baseline

`PassthroughAssurance` is intentionally a monitor, not a no-op. It records the same constraint violations as SC-CAIS but does not modify the proposed action. The physical simulator still saturates impossible commands at true plant/edge capacities. This prevents an unassured policy from receiving physically impossible throughput while preserving the distinction between attempted/residual policy violations and runtime-enforced actions.

## Cyber repair

When evidence trust is below threshold, SC-CAIS first attempts a trusted equivalent source for the intended target. If that is unavailable, it tries a trusted equivalent target at the same echelon. Only then does it fall back.
