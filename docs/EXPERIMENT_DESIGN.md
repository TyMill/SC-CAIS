# Experimental design

## Factorial core

The primary campaign crosses:

- scenario / disruption class;
- random seed;
- decision policy;
- assurance mode.

Assurance modes are deliberately distinct:

- `none`: monitor-only; violations are recorded and proposed actions are not corrected;
- `reject_only`: any violating proposal is replaced by the fallback action;
- `sc_cais`: repair is attempted first, then fallback is used only when repair is impossible or evidence is insufficiently trusted.

## Primary causal contrast

For each seed and scenario, compare the same agent under `none` and `sc_cais`. This paired design isolates the effect of runtime assurance from random demand variation.

## Safety / governance outcomes

Always report both:

- `attempted_violation_rate`: fraction of proposed actions violating at least one runtime constraint;
- `residual_violation_rate`: fraction of executed actions that still violate a constraint after the assurance decision.

The distinction is essential: an intelligent but aggressive agent may attempt more invalid actions under one state trajectory, while a successful assurance layer should reduce residual violations.

## Performance and resilience outcomes

Core outputs are service level, backlog, cumulative cost, resilience index, repair rate, fallback rate, and assurance latency. The article layer provides paired effects and bootstrap confidence intervals.

## Severity study

`sc_cais.severity.severity_grid()` generates low/medium/high/extreme operational and cyber scenarios for testing whether the relative value of assurance increases with disruption severity and data uncertainty.

## Severity × uncertainty robustness campaign

The robustness campaign crosses four disruption severity levels (`low`, `medium`, `high`, `extreme`) with four ambient uncertainty levels (`none`, `low`, `medium`, `high`). Ambient uncertainty is represented by deterministic multiplicative observation noise plus stochastic demand uncertainty. Cyber disruptions remain explicit perturbations and are therefore distinguishable from ordinary measurement uncertainty.

The primary robustness response variables are residual violation rate, prevented violation rate, normalized resilience loss, recovery time, cumulative cost, and resilience index.

## Rolling-horizon MILP baseline

The `rolling_horizon_milp` baseline solves an integer shipment-allocation problem at every decision step. Shipment quantities are integer variables while shortage slack is continuous. The model minimizes transportation, lead-time, trust-risk, and shortage penalties subject to perceived route availability, source-inventory limits, route capacities, and target replenishment needs. It is intentionally policy-only: it receives no privileged access to the hidden true state.
