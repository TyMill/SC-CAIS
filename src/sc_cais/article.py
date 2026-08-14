"""Article-ready aggregation, tables, and figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

METRIC_COLS = [
    "final_service_level",
    "mean_service_level",
    "mean_instant_service_level",
    "min_instant_service_level",
    "resilience_index",
    "resilience_loss_area",
    "normalized_resilience_loss",
    "max_performance_degradation",
    "recovery_time",
    "recovered_within_horizon",
    "post_disruption_service",
    "max_backlog",
    "final_backlog",
    "cumulative_cost",
    "attempted_violation_rate",
    "residual_violation_rate",
    "prevented_violation_rate",
    "repair_rate",
    "fallback_rate",
    "mean_assurance_latency_ms",
]


def aggregate_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in METRIC_COLS if c in frame.columns]
    group_cols = ["scenario", "category", "severity"]
    if "uncertainty" in frame.columns:
        group_cols.append("uncertainty")
    group_cols.extend(["agent", "assurance"])
    grouped = frame.groupby(group_cols, dropna=False)
    means = grouped[available].mean().add_suffix("_mean")
    stds = grouped[available].std(ddof=1).fillna(0).add_suffix("_sd")
    return means.join(stds).reset_index()


def write_article_tables(frame: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_benchmark(frame)
    aggregate_path = output / "table_benchmark_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    cols = [
        "scenario",
        "severity",
        "agent",
        "assurance",
        "cumulative_cost_mean",
        "resilience_index_mean",
        "normalized_resilience_loss_mean",
        "recovery_time_mean",
        "attempted_violation_rate_mean",
        "residual_violation_rate_mean",
        "prevented_violation_rate_mean",
        "mean_assurance_latency_ms_mean",
    ]
    if "uncertainty" in aggregate.columns:
        cols.insert(2, "uncertainty")
    frontier = aggregate[[c for c in cols if c in aggregate.columns]].copy()
    frontier_path = output / "table_assurance_performance_frontier.csv"
    frontier.to_csv(frontier_path, index=False)
    return aggregate_path, frontier_path


def write_frontier_figure(frame: pd.DataFrame, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    subset = frame[(frame["agent"] == "agentic_heuristic") & (frame["scenario"] == "compound")]
    summary = subset.groupby("assurance", as_index=False).agg(
        cost=("cumulative_cost", "mean"),
        resilience=("resilience_index", "mean"),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(summary["cost"], summary["resilience"], s=70)
    for row in summary.itertuples(index=False):
        ax.annotate(
            str(row.assurance),
            (cast(float, row.cost), cast(float, row.resilience)),
            xytext=(6, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean cumulative cost")
    ax.set_ylabel("Mean resilience index")
    ax.set_title("SC-CAIS assurance-performance frontier")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output / "fig_assurance_performance_frontier.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_resilience_figure(frame: pd.DataFrame, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    subset = frame[
        (frame["agent"] == "agentic_heuristic")
        & frame["scenario"].isin(["compound", "cyber_physical"])
    ]
    if subset.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No compound scenarios in this campaign", ha="center", va="center")
        ax.set_axis_off()
        path = output / "fig_resilience_loss.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return path
    summary = subset.groupby(["scenario", "assurance"], as_index=False).agg(
        loss=("normalized_resilience_loss", "mean")
    )
    pivot = summary.pivot(index="scenario", columns="assurance", values="loss")
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Normalized resilience loss")
    ax.set_xlabel("Scenario")
    ax.set_title("Resilience loss under compound disruptions")
    ax.grid(axis="y", alpha=0.25)
    bar_fig = cast(Figure, ax.figure)
    bar_fig.tight_layout()
    path = output / "fig_resilience_loss.png"
    bar_fig.savefig(path, dpi=220)
    plt.close(bar_fig)
    return path


def write_severity_uncertainty_heatmap(frame: pd.DataFrame, output_dir: str | Path) -> Path | None:
    if "uncertainty" not in frame.columns:
        return None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    severity_order = ["low", "medium", "high", "extreme"]
    uncertainty_order = ["none", "low", "medium", "high"]
    subset = frame[
        (frame["agent"] == "agentic_heuristic")
        & (frame["assurance"] == "sc_cais")
        & (frame["severity"].isin(severity_order))
        & (frame["uncertainty"].isin(uncertainty_order))
    ]
    if subset.empty:
        return None
    summary = subset.groupby(["severity", "uncertainty"], as_index=False).agg(
        benefit=("prevented_violation_rate", "mean")
    )
    matrix = summary.pivot(index="severity", columns="uncertainty", values="benefit").reindex(
        index=severity_order, columns=uncertainty_order
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Disruption severity")
    ax.set_title("Prevented violation rate: severity x uncertainty")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                numeric_value = cast(float, value)
                ax.text(j, i, f"{numeric_value:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Prevented violation rate")
    fig.tight_layout()
    path = output / "fig_severity_uncertainty_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_all_article_outputs(frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    table1, table2 = write_article_tables(frame, output_dir)
    paths = [
        table1,
        table2,
        write_frontier_figure(frame, output_dir),
        write_resilience_figure(frame, output_dir),
    ]
    heatmap = write_severity_uncertainty_heatmap(frame, output_dir)
    if heatmap is not None:
        paths.append(heatmap)
    return paths


def write_trajectory_figure(
    trajectory: pd.DataFrame,
    output_dir: str | Path,
    *,
    scenario: str = "compound",
    agent: str = "agentic_heuristic",
) -> Path | None:
    """Plot disruption-degradation-recovery trajectories for article use."""
    subset = trajectory[
        (trajectory["scenario"] == scenario) & (trajectory["agent"] == agent)
    ].copy()
    if subset.empty:
        return None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = subset.groupby(["step", "assurance"], as_index=False).agg(
        service=("instant_service_level", "mean"),
        residual=("residual_violations", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for assurance, group in summary.groupby("assurance"):
        ax.plot(
            group["step"],
            group["service"],
            marker="o",
            markersize=2.5,
            label=str(assurance),
        )
    start = subset["disruption_start"].dropna()
    end = subset["disruption_end"].dropna()
    if not start.empty and not end.empty:
        ax.axvspan(float(start.iloc[0]), float(end.iloc[0]), alpha=0.12)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Mean instantaneous service level")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Disruption-degradation-recovery trajectory: {scenario}")
    ax.legend(title="Assurance")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output / f"fig_trajectory_{scenario}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_ablation_table(frame: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write repair-versus-fallback ablation table for the agentic policy."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    subset = frame[
        (frame["agent"] == "agentic_heuristic")
        & (frame["assurance"].isin(["reject_only", "sc_cais"]))
    ]
    cols = [
        "resilience_index",
        "normalized_resilience_loss",
        "recovery_time",
        "cumulative_cost",
        "residual_violation_rate",
        "repair_rate",
        "fallback_rate",
    ]
    available = [c for c in cols if c in subset.columns]
    table = (
        subset.groupby(["scenario", "assurance"], as_index=False)[available]
        .mean()
        .sort_values(["scenario", "assurance"])
    )
    path = output / "table_repair_vs_fallback_ablation.csv"
    table.to_csv(path, index=False)
    return path
