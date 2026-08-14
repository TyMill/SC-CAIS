"""Run one final-campaign scenario as a resumable chunk."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sc_cais.benchmark import BenchmarkRunner
from sc_cais.scenarios import default_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/article_final/parts"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = next(s for s in default_scenarios() if s.name == args.scenario)
    args.output.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(horizon=36, seeds=tuple(range(10)))
    classical = runner.run(
        scenarios=[scenario],
        agent_names={"rule_based", "cost_aware_optimizer", "rolling_horizon_milp"},
        assurance_names={"none"},
    )
    agentic, trajectories = runner.run_with_trajectories(
        scenarios=[scenario],
        agent_names={"agentic_heuristic"},
        assurance_names={"none", "reject_only", "sc_cais"},
    )
    main_frame = pd.concat([classical, agentic], ignore_index=True)
    main_frame.to_csv(args.output / f"main_{args.scenario}.csv", index=False)
    trajectories.to_csv(args.output / f"traj_{args.scenario}.csv", index=False)
    print(args.scenario, len(main_frame), len(trajectories))


if __name__ == "__main__":
    main()
