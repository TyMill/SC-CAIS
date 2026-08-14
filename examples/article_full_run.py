"""Publication-scale SC-CAIS campaign (10 seeds, 36-step horizon)."""

from pathlib import Path

import pandas as pd

from sc_cais.article import write_all_article_outputs
from sc_cais.benchmark import BenchmarkRunner
from sc_cais.reproducibility import write_manifest
from sc_cais.robustness import severity_uncertainty_scenarios, uncertainty_grid

OUTPUT = Path("outputs/article_full")
OUTPUT.mkdir(parents=True, exist_ok=True)
runner = BenchmarkRunner(horizon=36, seeds=tuple(range(10)))

baseline = runner.run(
    agent_names={"rule_based", "cost_aware_optimizer", "rolling_horizon_milp"},
    assurance_names={"none"},
)
agentic = runner.run(
    agent_names={"agentic_heuristic"},
    assurance_names={"none", "reject_only", "sc_cais"},
)
main = pd.concat([baseline, agentic], ignore_index=True)
main_path = OUTPUT / "benchmark_main.csv"
main.to_csv(main_path, index=False)
main_outputs = write_all_article_outputs(main, OUTPUT)

robustness = runner.run(
    scenarios=severity_uncertainty_scenarios(),
    agent_names={"agentic_heuristic"},
    assurance_names={"none", "reject_only", "sc_cais"},
    uncertainty_levels=uncertainty_grid(),
)
robustness_path = OUTPUT / "benchmark_severity_uncertainty.csv"
robustness.to_csv(robustness_path, index=False)
robustness_outputs = write_all_article_outputs(robustness, OUTPUT / "robustness")

manifest = write_manifest(
    OUTPUT,
    [main_path, robustness_path, *main_outputs, *robustness_outputs],
)
print(f"main_rows={len(main)}")
print(f"robustness_rows={len(robustness)}")
print(f"manifest={manifest}")
