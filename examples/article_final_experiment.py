"""Final publication-scale SC-CAIS experimental campaign.

Runs the frozen v0.3 design and writes all manuscript-ready artifacts.
Use --quick only to verify the pipeline before launching the full campaign.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sc_cais.article import (
    write_ablation_table,
    write_all_article_outputs,
    write_trajectory_figure,
)
from sc_cais.benchmark import BenchmarkRunner
from sc_cais.reproducibility import write_manifest
from sc_cais.robustness import severity_uncertainty_scenarios, uncertainty_grid
from sc_cais.statistics import inferential_effect_summary, paired_assurance_effects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/article_final"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="2 seeds / 20 steps pipeline verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    seeds = (0, 1) if args.quick else tuple(range(10))
    horizon = 20 if args.quick else 36
    runner = BenchmarkRunner(horizon=horizon, seeds=seeds)

    classical = runner.run(
        agent_names={"rule_based", "cost_aware_optimizer", "rolling_horizon_milp"},
        assurance_names={"none"},
    )
    agentic, trajectories = runner.run_with_trajectories(
        agent_names={"agentic_heuristic"},
        assurance_names={"none", "reject_only", "sc_cais"},
    )
    main_frame = pd.concat([classical, agentic], ignore_index=True)
    main_path = output / "benchmark_main.csv"
    trajectory_path = output / "trajectories_main.csv"
    main_frame.to_csv(main_path, index=False)
    trajectories.to_csv(trajectory_path, index=False)

    article_outputs = write_all_article_outputs(main_frame, output)
    ablation_path = write_ablation_table(main_frame, output)
    trajectory_paths = [
        path
        for scenario in ("compound", "cyber_physical")
        if (path := write_trajectory_figure(trajectories, output, scenario=scenario)) is not None
    ]

    inference_paths: list[Path] = []
    for baseline, treatment, label in (
        ("none", "sc_cais", "sc_cais_vs_none"),
        ("reject_only", "sc_cais", "repair_vs_fallback"),
    ):
        paired = paired_assurance_effects(agentic, baseline=baseline, treatment=treatment)
        paired_path = output / f"paired_{label}.csv"
        inference_path = output / f"inference_{label}.csv"
        paired.to_csv(paired_path, index=False)
        inferential_effect_summary(paired).to_csv(inference_path, index=False)
        inference_paths.extend([paired_path, inference_path])

    robustness = runner.run(
        scenarios=severity_uncertainty_scenarios(),
        agent_names={"agentic_heuristic"},
        assurance_names={"none", "reject_only", "sc_cais"},
        uncertainty_levels=uncertainty_grid(),
    )
    robustness_path = output / "benchmark_severity_uncertainty.csv"
    robustness.to_csv(robustness_path, index=False)
    robustness_outputs = write_all_article_outputs(robustness, output / "robustness")

    manifest_files = [
        main_path,
        trajectory_path,
        ablation_path,
        robustness_path,
        *article_outputs,
        *trajectory_paths,
        *inference_paths,
        *robustness_outputs,
    ]
    manifest = write_manifest(output, manifest_files)
    print(f"mode={'quick' if args.quick else 'full'}")
    print(f"horizon={horizon}")
    print(f"seeds={len(seeds)}")
    print(f"main_rows={len(main_frame)}")
    print(f"trajectory_rows={len(trajectories)}")
    print(f"robustness_rows={len(robustness)}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
