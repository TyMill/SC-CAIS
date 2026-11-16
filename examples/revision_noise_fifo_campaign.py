"""Matched observation-noise sensitivity and FIFO on-time trajectory evidence.

Experimental scope:
* 11 scenarios (including trust=0.8 covert spoof), 10 seeds, 36 steps
* three proposal policies, observed-only full repair / clipping / monitor-only
* multiplicative observation noise sigma in {0.00, 0.10, 0.20}
* independent offline command audit, never visible to the assurance kernel
* FIFO on-time demand measure (not the historical backlog-crediting metric)

This script performs new simulations; it never substitutes previous legacy plots.
The policy API still exposes simulator metadata to deterministic proposers.
Therefore this probes robustness of the tested interface, not fully isolated
end-to-end industrial information boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sc_cais.agents import AgenticHeuristicAgent, RollingHorizonMILPAgent, RuleBasedAgent
from sc_cais.environment import SupplyChainEnv
from sc_cais.network import create_reference_network
from sc_cais.revision_experiments import EvidenceLimitedAssurance, revision_scenarios
from sc_cais.scenarios import default_scenarios

MODES = ("rule_repair", "quantity_clip", "monitor_only")
LEVELS = (0.0, 0.10, 0.20)
TRAJECTORY_SCENARIOS = {
    "normal", "compound", "covert_false_inventory_trust_080", "false_inventory",
}


def simulate(*, quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = (
        default_scenarios()[:1]
        if quick
        else [*default_scenarios(), revision_scenarios()[-1]]
    )
    seeds = (0,) if quick else tuple(range(10))
    horizon = 4 if quick else 36
    policies = (AgenticHeuristicAgent, RollingHorizonMILPAgent, RuleBasedAgent)
    rows: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []

    for scenario in scenarios:
        for seed in seeds:
            for policy_type in policies:
                for sigma in LEVELS:
                    for mode in MODES:
                        environment = SupplyChainEnv(
                            create_reference_network(),
                            seed=seed,
                            disruptions=scenario.disruptions,
                            demand_noise=0.08,
                            observation_noise=sigma,
                            service_accounting="fifo_on_time",
                        )
                        policy = policy_type()
                        assurance = EvidenceLimitedAssurance(
                            view="observed", repair_mode=mode
                        )
                        results = environment.run(policy, assurance, horizon)
                        audits = assurance.audit_records
                        n = len(audits)
                        assert n > 0
                        index = np.array(
                            [result.instant_service_level for result in results],
                            dtype=float,
                        )
                        rows.append(
                            {
                                "scenario": scenario.name,
                                "seed": seed,
                                "policy": policy.name,
                                "mode": mode,
                                "observation_noise": sigma,
                                "horizon": horizon,
                                "resilience_fifo": float(index.mean()),
                                "max_backlog": max(
                                    float(result.total_backlog) for result in results
                                ),
                                "final_backlog": float(results[-1].total_backlog),
                                "true_invalid_rate": sum(
                                    bool(a["executed_true_violation"]) for a in audits
                                ) / n,
                                "n_actions": n,
                                "true_invalid_count": sum(
                                    bool(a["executed_true_violation"]) for a in audits
                                ),
                                "repair_rate": sum(
                                    a["decision"] == "repair" for a in audits
                                ) / n,
                                "fallback_rate": sum(
                                    a["decision"] == "fallback" for a in audits
                                ) / n,
                            }
                        )
                        if scenario.name in TRAJECTORY_SCENARIOS or quick:
                            for result in results:
                                trajectories.append(
                                    {
                                        "scenario": scenario.name,
                                        "seed": seed,
                                        "policy": policy.name,
                                        "mode": mode,
                                        "observation_noise": sigma,
                                        "step": result.step,
                                        "fifo_service": result.instant_service_level,
                                        "backlog": result.total_backlog,
                                    }
                                )
    summary = pd.DataFrame(rows)
    trajectory = pd.DataFrame(trajectories)
    if not quick:
        assert len(summary) == 11 * 10 * 3 * 3 * 3
        assert len(trajectory) == 4 * 10 * 3 * 3 * 3 * 36
    return summary, trajectory


def write_results(
    output: Path, runs: pd.DataFrame, trajectories: pd.DataFrame, quick: bool
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    grouped = (
        runs.groupby(["policy", "mode", "observation_noise"], as_index=False)
        .agg(
            n_runs=("resilience_fifo", "size"),
            mean_fifo_resilience=("resilience_fifo", "mean"),
            mean_true_invalid_rate=("true_invalid_rate", "mean"),
            sum_true_invalid_count=("true_invalid_count", "sum"),
            sum_action_count=("n_actions", "sum"),
            mean_max_backlog=("max_backlog", "mean"),
        )
        .sort_values(["policy", "mode", "observation_noise"])
    )
    grouped["pooled_true_invalid_rate"] = (
        grouped["sum_true_invalid_count"] / grouped["sum_action_count"]
    )
    by_scenario = (
        runs.groupby(
            ["scenario", "policy", "mode", "observation_noise"], as_index=False
        )
        .agg(
            n_runs=("resilience_fifo", "size"),
            mean_fifo_resilience=("resilience_fifo", "mean"),
            mean_true_invalid_rate=("true_invalid_rate", "mean"),
            sum_true_invalid_count=("true_invalid_count", "sum"),
            sum_action_count=("n_actions", "sum"),
        )
        .sort_values(["scenario", "policy", "mode", "observation_noise"])
    )
    traj_mean = (
        trajectories.groupby(
            ["scenario", "policy", "mode", "observation_noise", "step"],
            as_index=False,
        )
        .agg(
            n_seeds=("fifo_service", "size"),
            mean_fifo_service=("fifo_service", "mean"),
            sd_fifo_service=("fifo_service", "std"),
            mean_backlog=("backlog", "mean"),
        )
        .sort_values(["scenario", "policy", "mode", "observation_noise", "step"])
    )
    # Matched change in noise under exactly the same scenario, seed and policy.
    base = runs.loc[
        runs.observation_noise == 0.0,
        ["scenario", "seed", "policy", "mode", "resilience_fifo"],
    ].rename(columns={"resilience_fifo": "baseline_fifo_resilience"})
    pairs = runs.merge(base, on=["scenario", "seed", "policy", "mode"])
    pairs = pairs.loc[pairs.observation_noise > 0].copy()
    pairs["paired_delta_vs_zero_noise"] = (
        pairs.resilience_fifo - pairs.baseline_fifo_resilience
    )
    paired = (
        pairs.groupby(["policy", "mode", "observation_noise"], as_index=False)
        .agg(
            n_pairs=("paired_delta_vs_zero_noise", "size"),
            mean_delta=("paired_delta_vs_zero_noise", "mean"),
            median_delta=("paired_delta_vs_zero_noise", "median"),
            negative_pairs=("paired_delta_vs_zero_noise", lambda x: int((x < 0).sum())),
        )
    )
    files = {
        "noise_runs.csv": runs,
        "noise_overall.csv": grouped,
        "noise_by_scenario.csv": by_scenario,
        "noise_paired_changes.csv": paired,
        "fifo_trajectories_per_seed.csv": trajectories,
        "fifo_trajectories_mean.csv": traj_mean,
    }
    for filename, frame in files.items():
        frame.to_csv(output / filename, index=False)
    manifest = {
        "protocol": "observed-only assurance, FIFO service, paired seed and scenario",
        "noise_levels": LEVELS,
        "modes": MODES,
        "trajectory_scenarios": sorted(TRAJECTORY_SCENARIOS),
        "n_runs": len(runs),
        "n_trajectory_rows": len(trajectories),
        "python": sys.version,
        "platform": platform.platform(),
        "quick": quick,
        "sha256": {
            filename: hashlib.sha256((output / filename).read_bytes()).hexdigest()
            for filename in files
        },
    }
    (output / "noise_fifo_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("RUN_COUNT", len(runs))
    print("ACTION_COUNT", int(runs.n_actions.sum()))
    print("OVERALL_RESULTS")
    print(grouped.to_csv(index=False))
    print("MATCHED_SHIFTS")
    print(paired.to_csv(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/revision_noise_fifo"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    runs, trajectories = simulate(quick=args.quick)
    write_results(args.output, runs, trajectories, args.quick)


if __name__ == "__main__":
    main()
