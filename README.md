[![DOI](https://zenodo.org/badge/1334411735.svg)](https://doi.org/10.5281/zenodo.21937793)


# SC-CAIS

**Controlled Agentic AI for Resilient Supply Chains** — a research-oriented Python library for runtime-assured autonomous supply-chain reconfiguration under operational and cyber-physical disruptions.

SC-CAIS separates the **decision policy** from the **assurance kernel**. An AI agent may propose an action, but the proposal is passed through runtime constraints before execution:

```text
Environment -> Observation -> Agent -> Runtime Assurance -> Execution -> Audit
                                   |         |
                                   |         +-- APPROVE
                                   |         +-- REPAIR
                                   |         +-- FALLBACK
                                   +-- proposed action
```

## Research purpose

The library is designed to evaluate the performance–resilience–assurance trade-off of controlled agentic autonomy. It intentionally maintains separate **true physical state** and **agent observation state**, allowing cyber-physical experiments in which an apparently rational agent acts on stale, missing or manipulated data.

## Included in v0.1.0

- directed multi-echelon supply-chain network model;
- discrete-time inventory, backlog, shipment and demand simulation;
- typed agent interface and three baseline policies;
- policy-agnostic runtime assurance kernel;
- hard feasibility, transport capacity, inventory, cyber trust and governance constraints;
- minimal-deviation quantity/route repair;
- safe fallback action;
- operational disruptions: supplier failure, hub closure, demand surge, transport delay;
- cyber disruptions: false inventory, telemetry loss, route spoofing;
- compound cyber-physical scenarios;
- reproducible multi-seed benchmark runner;
- CSV summaries and environment/hash manifest;
- CLI, examples and pytest suite.

## Installation

```bash
python -m pip install -e .
```

Development tooling:

```bash
python -m pip install -e '.[dev]'
```

Requires Python 3.13 or newer.

## Quick start

```python
from sc_cais.agents import AgenticHeuristicAgent
from sc_cais.assurance import RuntimeAssurance
from sc_cais.environment import SupplyChainEnv
from sc_cais.evaluation import summarize
from sc_cais.network import create_reference_network
from sc_cais.scenarios import default_scenarios

scenario = next(s for s in default_scenarios() if s.name == "cyber_physical")
env = SupplyChainEnv(create_reference_network(), seed=42, disruptions=scenario.disruptions)
results = env.run(AgenticHeuristicAgent(), RuntimeAssurance(), horizon=24)
print(summarize(results))
```

Or:

```bash
PYTHONPATH=src python -m sc_cais.cli smoke
PYTHONPATH=src python -m sc_cais.cli benchmark --config configs/benchmark.yaml --output outputs
```

## Assurance semantics

For proposed action `a_p`, the kernel checks a constraint family `C`.

1. If `a_p` is feasible, it is approved.
2. If every violation is repairable, SC-CAIS searches for a nearby feasible action by changing only the dimensions required to restore feasibility (currently target route and/or quantity).
3. If trust is insufficient or no feasible repair exists, the kernel falls back to `HOLD`.

This makes the assurance layer independent from the agent implementation.

## Benchmark scenarios

| Scenario | Type | Core stressor |
|---|---|---|
| `normal` | baseline | no disruption |
| `supplier_failure` | operational | supplier outage |
| `hub_closure` | operational | hub unavailable |
| `demand_surge` | operational | demand shock |
| `transport_delay` | operational | increased lead time |
| `false_inventory` | cyber | manipulated inventory observation |
| `telemetry_loss` | cyber | unavailable / low-trust telemetry |
| `route_spoofing` | cyber | manipulated route status |
| `compound` | compound | supplier + demand + false inventory |
| `cyber_physical` | compound | physical hub closure + route spoof + surge |

## Reproducibility

Run the full article-style campaign:

```bash
PYTHONPATH=src python examples/article_run.py
```

Outputs include a benchmark CSV and SHA-256/environment manifest.

## Verification

```bash
PYTHONPATH=src python scripts/verify.py
```

The repository also contains configuration for Ruff and strict mypy. These optional developer tools can be installed with the `dev` extra.

## Repository map

```text
src/sc_cais/
  agents.py           decision policies
  assurance.py        runtime assurance kernel
  benchmark.py        multi-seed experiments
  cli.py              command-line interface
  config.py           validated YAML configuration
  constraints.py      assurance constraints
  disruptions.py      operational + cyber perturbations
  environment.py      simulation engine
  evaluation.py       metrics
  io.py               serialization
  models.py           typed domain models
  network.py          graph and reference network
  reproducibility.py  hashes/environment manifest
  scenarios.py        named article scenarios
```

## Scientific extension points

The next research-grade extensions are deliberately supported by the package boundaries:

- MILP / rolling-horizon optimizer baseline;
- RL agent adapter;
- LLM tool-using agent adapter;
- temporal-logic constraints and runtime monitor;
- richer production/BOM dynamics;
- Monte Carlo disruption severity grids;
- recovery-time and resilience-area calculations against a scenario-specific baseline;
- article figure/table generator.

## License

MIT.

## Article-grade v0.2 additions

SC-CAIS v0.2 extends the benchmark from a runtime-assurance demonstrator to a publication-oriented experimental framework. It adds a rolling-horizon MILP replenishment baseline, deterministic observation uncertainty, severity × uncertainty sweeps, and formal resilience recovery metrics.

Key resilience outputs now include instantaneous service level, resilience loss area, normalized resilience loss, maximum performance degradation, recovery time, recovery-within-horizon, and post-disruption service. The benchmark continues to report proposed and residual constraint violations separately, which makes the effect of runtime assurance directly measurable.

Run a fast deterministic reference pack with:

```bash
PYTHONPATH=src python examples/article_run.py
```

Run the publication-scale campaign with 10 seeds and a 36-step horizon with:

```bash
PYTHONPATH=src python examples/article_full_run.py
```

The full campaign compares classical baselines with unassured agentic AI, reject-only guardrails, and SC-CAIS repair-enabled runtime assurance. Robustness experiments sweep four disruption-severity levels and four uncertainty levels.


## Final publication experiment (v0.3)

Run the frozen 10-seed, 36-step article campaign with:

```bash
PYTHONPATH=src python examples/article_final_experiment.py --output outputs/article_final
```

For constrained environments, the same design can be executed with the resumable
`run_main_chunk.py` and `run_robustness_chunk.py` helpers and merged afterward.
The final campaign includes seed-paired Wilcoxon tests, deterministic bootstrap
confidence intervals, Cohen's dz, repair-vs-fallback ablation, step trajectories,
and severity-by-uncertainty robustness experiments.
