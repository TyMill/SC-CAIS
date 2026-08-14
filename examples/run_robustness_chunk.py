"""Run one robustness scenario as a resumable final-campaign chunk."""

from __future__ import annotations

import argparse
from pathlib import Path

from sc_cais.benchmark import BenchmarkRunner
from sc_cais.robustness import severity_uncertainty_scenarios, uncertainty_grid


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
    scenario = next(
        s for s in severity_uncertainty_scenarios() if s.name == args.scenario
    )
    args.output.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(horizon=36, seeds=tuple(range(10)))
    frame = runner.run(
        scenarios=[scenario],
        agent_names={"agentic_heuristic"},
        assurance_names={"none", "reject_only", "sc_cais"},
        uncertainty_levels=uncertainty_grid(),
    )
    frame.to_csv(args.output / f"robust_{args.scenario}.csv", index=False)
    print(args.scenario, len(frame))


if __name__ == "__main__":
    main()
