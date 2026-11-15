"""Reproducible benchmark runner for SC-CAIS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from sc_cais.agents import (
    Agent,
    AgenticHeuristicAgent,
    CostAwareOptimizerAgent,
    RollingHorizonMILPAgent,
    RuleBasedAgent,
    StochasticAgent,
)
from sc_cais.assurance import AssurancePolicy, PassthroughAssurance, RuntimeAssurance
from sc_cais.environment import SupplyChainEnv
from sc_cais.evaluation import results_frame, summarize
from sc_cais.network import create_reference_network
from sc_cais.robustness import UncertaintyLevel
from sc_cais.scenarios import Scenario, default_scenarios, disruption_window


@dataclass(slots=True)
class BenchmarkRunner:
    horizon: int = 24
    seeds: tuple[int, ...] = (0, 1, 2)

    def _agents(self) -> list[Agent]:
        return [
            RuleBasedAgent(),
            CostAwareOptimizerAgent(),
            RollingHorizonMILPAgent(),
            AgenticHeuristicAgent(),
            StochasticAgent(),
        ]

    def _assurance_modes(self) -> dict[str, AssurancePolicy]:
        return {
            "none": PassthroughAssurance(),
            "reject_only": RuntimeAssurance(repair_enabled=False),
            "sc_cais": RuntimeAssurance(repair_enabled=True),
        }

    def run(
        self,
        *,
        scenarios: list[Scenario] | None = None,
        agent_names: set[str] | None = None,
        assurance_names: set[str] | None = None,
        uncertainty_levels: tuple[UncertaintyLevel, ...] | None = None,
    ) -> pd.DataFrame:
        scenarios = scenarios or default_scenarios()
        uncertainty_levels = uncertainty_levels or (UncertaintyLevel("default", 0.0, 0.08),)
        rows: list[dict[str, object]] = []
        for scenario in scenarios:
            _, disruption_end = disruption_window(scenario)
            for uncertainty in uncertainty_levels:
                for seed in self.seeds:
                    for agent in self._agents():
                        if agent_names and agent.name not in agent_names:
                            continue
                        for assurance_name, assurance in self._assurance_modes().items():
                            if assurance_names and assurance_name not in assurance_names:
                                continue
                            env = SupplyChainEnv(
                                create_reference_network(),
                                seed=seed,
                                disruptions=scenario.disruptions,
                                demand_noise=uncertainty.demand_noise,
                                observation_noise=uncertainty.observation_noise,
                            )
                            results = env.run(agent, assurance, self.horizon)
                            row: dict[str, object] = {
                                "scenario": scenario.name,
                                "category": scenario.category,
                                "severity": scenario.severity,
                                "uncertainty": uncertainty.name,
                                "observation_noise": uncertainty.observation_noise,
                                "demand_noise": uncertainty.demand_noise,
                                "seed": seed,
                                "agent": agent.name,
                                "assurance": assurance_name,
                            }
                            row.update(summarize(results, disruption_end=disruption_end))
                            rows.append(row)
        return pd.DataFrame(rows)

    def run_with_trajectories(
        self,
        *,
        scenarios: list[Scenario] | None = None,
        agent_names: set[str] | None = None,
        assurance_names: set[str] | None = None,
        uncertainty_levels: tuple[UncertaintyLevel, ...] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run benchmark and return both experiment summaries and step trajectories."""
        scenarios = scenarios or default_scenarios()
        uncertainty_levels = uncertainty_levels or (UncertaintyLevel("default", 0.0, 0.08),)
        summary_rows: list[dict[str, object]] = []
        trajectory_frames: list[pd.DataFrame] = []
        for scenario in scenarios:
            disruption_start, disruption_end = disruption_window(scenario)
            for uncertainty in uncertainty_levels:
                for seed in self.seeds:
                    for agent in self._agents():
                        if agent_names and agent.name not in agent_names:
                            continue
                        for assurance_name, assurance in self._assurance_modes().items():
                            if assurance_names and assurance_name not in assurance_names:
                                continue
                            env = SupplyChainEnv(
                                create_reference_network(),
                                seed=seed,
                                disruptions=scenario.disruptions,
                                demand_noise=uncertainty.demand_noise,
                                observation_noise=uncertainty.observation_noise,
                            )
                            results = env.run(agent, assurance, self.horizon)
                            identity: dict[str, object] = {
                                "scenario": scenario.name,
                                "category": scenario.category,
                                "severity": scenario.severity,
                                "uncertainty": uncertainty.name,
                                "observation_noise": uncertainty.observation_noise,
                                "demand_noise": uncertainty.demand_noise,
                                "seed": seed,
                                "agent": agent.name,
                                "assurance": assurance_name,
                            }
                            summary = dict(identity)
                            summary.update(summarize(results, disruption_end=disruption_end))
                            summary_rows.append(summary)
                            trajectory = results_frame(results)
                            for key, value in reversed(list(identity.items())):
                                trajectory.insert(0, key, cast(Any, value))
                            trajectory["disruption_start"] = disruption_start
                            trajectory["disruption_end"] = disruption_end
                            trajectory["phase"] = "pre"
                            if disruption_start is not None:
                                mask = trajectory["step"] >= disruption_start
                                trajectory.loc[mask, "phase"] = "disruption"
                            if disruption_end is not None:
                                mask = trajectory["step"] >= disruption_end
                                trajectory.loc[mask, "phase"] = "recovery"
                            trajectory_frames.append(trajectory)
        summary_frame = pd.DataFrame(summary_rows)
        trajectory_frame = (
            pd.concat(trajectory_frames, ignore_index=True) if trajectory_frames else pd.DataFrame()
        )
        return summary_frame, trajectory_frame

    def write(
        self,
        output_dir: str | Path,
        *,
        scenarios: list[Scenario] | None = None,
        agent_names: set[str] | None = None,
        assurance_names: set[str] | None = None,
        uncertainty_levels: tuple[UncertaintyLevel, ...] | None = None,
    ) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        frame = self.run(
            scenarios=scenarios,
            agent_names=agent_names,
            assurance_names=assurance_names,
            uncertainty_levels=uncertainty_levels,
        )
        path = output / "benchmark_summary.csv"
        frame.to_csv(path, index=False)
        return path
