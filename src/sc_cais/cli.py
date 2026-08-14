"""Command-line interface for SC-CAIS."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from sc_cais.agents import AgenticHeuristicAgent
from sc_cais.assurance import RuntimeAssurance
from sc_cais.benchmark import BenchmarkRunner
from sc_cais.config import load_benchmark_config
from sc_cais.environment import SupplyChainEnv
from sc_cais.evaluation import summarize
from sc_cais.network import create_reference_network
from sc_cais.reproducibility import write_manifest
from sc_cais.scenarios import default_scenarios

app = typer.Typer(help="SC-CAIS controlled agentic supply-chain simulator and benchmark.")


@app.command()
def smoke(horizon: int = 12, seed: int = 0) -> None:
    """Run a deterministic compound-disruption smoke test."""
    scenario = next(s for s in default_scenarios() if s.name == "compound")
    env = SupplyChainEnv(
        create_reference_network(),
        seed=seed,
        disruptions=scenario.disruptions,
    )
    results = env.run(AgenticHeuristicAgent(), RuntimeAssurance(), horizon)
    metrics = summarize(results)
    typer.echo("SC-CAIS smoke test complete")
    for key, value in metrics.items():
        typer.echo(f"{key}: {value:.6f}")


@app.command()
def benchmark(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/benchmark.yaml"),
    output: Annotated[Path, typer.Option()] = Path("outputs"),
) -> None:
    """Run configured benchmark campaign and write article-ready CSV output."""
    cfg = load_benchmark_config(config)
    scenario_map = {s.name: s for s in default_scenarios()}
    selected = [scenario_map[name] for name in cfg.scenarios]
    runner = BenchmarkRunner(horizon=cfg.horizon, seeds=tuple(cfg.seeds))
    csv_path = runner.write(
        output,
        scenarios=selected,
        agent_names=set(cfg.agents),
        assurance_names=set(cfg.assurance),
    )
    manifest = write_manifest(output, [csv_path])
    typer.echo(f"benchmark: {csv_path}")
    typer.echo(f"manifest: {manifest}")


@app.command("validate-network")
def validate_network() -> None:
    """Validate the deterministic reference network."""
    errors = create_reference_network().validate()
    if errors:
        for error in errors:
            typer.echo(error)
        raise typer.Exit(code=1)
    typer.echo("Reference network valid")


if __name__ == "__main__":
    app()
