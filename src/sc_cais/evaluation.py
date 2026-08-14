"""Experiment metrics and resilience summaries."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from sc_cais.models import AssuranceDecision, StepResult


def results_frame(results: list[StepResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        proposed_violations = sum(len(o.violations) for o in r.outcomes)
        residual_violations = sum(len(o.residual_violations) for o in r.outcomes)
        decisions = Counter(o.decision for o in r.outcomes)
        rows.append(
            {
                "step": r.step,
                "n_actions": len(r.outcomes),
                "proposed_violations": proposed_violations,
                "residual_violations": residual_violations,
                "repairs": decisions[AssuranceDecision.REPAIR],
                "fallbacks": decisions[AssuranceDecision.FALLBACK],
                "demand": r.demand,
                "fulfilled": r.fulfilled,
                "service_level": r.service_level,
                "instant_service_level": r.instant_service_level,
                "backlog": r.total_backlog,
                "step_cost": r.step_cost,
                "cumulative_cost": r.cumulative_cost,
                "data_quality": r.data_quality,
                "assurance_latency_ms": sum(o.latency_ms for o in r.outcomes),
            }
        )
    return pd.DataFrame(rows)


def _recovery_metrics(
    instant_service: np.ndarray,
    *,
    recovery_threshold: float = 0.95,
    disruption_end: int | None = None,
) -> dict[str, float]:
    """Compute article-grade resilience loss and recovery metrics.

    The resilience loss area is the discrete integral of performance loss relative
    to perfect service. Recovery time is measured from ``disruption_end`` when it
    is known, otherwise from the first step below the recovery threshold.
    """
    if instant_service.size == 0:
        return {
            "resilience_loss_area": 0.0,
            "normalized_resilience_loss": 0.0,
            "max_performance_degradation": 0.0,
            "recovery_time": 0.0,
            "recovered_within_horizon": 1.0,
            "post_disruption_service": 1.0,
        }
    loss = np.maximum(0.0, 1.0 - instant_service)
    loss_area = float(np.sum(loss))
    max_degradation = float(np.max(loss))
    first_degraded = np.flatnonzero(instant_service < recovery_threshold)
    anchor = int(disruption_end) if disruption_end is not None else (
        int(first_degraded[0]) if first_degraded.size else 0
    )
    anchor = max(0, min(anchor, instant_service.size - 1))
    recovered_at: int | None = None
    # Require two consecutive satisfactory steps to avoid declaring recovery on a spike.
    for idx in range(anchor, instant_service.size):
        window = instant_service[idx : min(idx + 2, instant_service.size)]
        if window.size > 0 and bool(np.all(window >= recovery_threshold)):
            recovered_at = idx
            break
    recovered = recovered_at is not None
    if recovered_at is None:
        recovery_time = float(instant_service.size - anchor)
    else:
        recovery_time = float(recovered_at - anchor)
    post = instant_service[anchor:]
    return {
        "resilience_loss_area": loss_area,
        "normalized_resilience_loss": loss_area / float(instant_service.size),
        "max_performance_degradation": max_degradation,
        "recovery_time": recovery_time,
        "recovered_within_horizon": float(recovered),
        "post_disruption_service": (
            float(np.mean(post)) if post.size else float(instant_service[-1])
        ),
    }


def summarize(
    results: list[StepResult],
    *,
    disruption_end: int | None = None,
    recovery_threshold: float = 0.95,
) -> dict[str, float]:
    if not results:
        return {}
    outcomes = [outcome for r in results for outcome in r.outcomes]
    decisions = Counter(o.decision for o in outcomes)
    attempted = sum(bool(o.violations) for o in outcomes)
    residual = sum(bool(o.residual_violations) for o in outcomes)
    prevented = sum(bool(o.violations) and not o.residual_violations for o in outcomes)
    service = np.array([r.service_level for r in results], dtype=float)
    instant_service = np.array([r.instant_service_level for r in results], dtype=float)
    backlog = np.array([r.total_backlog for r in results], dtype=float)
    latencies = (
        np.array([o.latency_ms for o in outcomes], dtype=float)
        if outcomes
        else np.array([0.0])
    )
    n_actions = max(len(outcomes), 1)
    metrics = {
        "final_service_level": float(service[-1]),
        "mean_service_level": float(np.mean(service)),
        "mean_instant_service_level": float(np.mean(instant_service)),
        "min_instant_service_level": float(np.min(instant_service)),
        "min_service_level": float(np.min(service)),
        "resilience_index": float(np.mean(instant_service)),
        "max_backlog": float(np.max(backlog)),
        "final_backlog": float(backlog[-1]),
        "cumulative_cost": float(results[-1].cumulative_cost),
        "actions_per_step": len(outcomes) / len(results),
        "attempted_violation_rate": attempted / n_actions,
        "residual_violation_rate": residual / n_actions,
        "prevented_violation_rate": prevented / max(attempted, 1),
        "repair_rate": decisions[AssuranceDecision.REPAIR] / n_actions,
        "fallback_rate": decisions[AssuranceDecision.FALLBACK] / n_actions,
        "approve_rate": decisions[AssuranceDecision.APPROVE] / n_actions,
        "mean_assurance_latency_ms": float(np.mean(latencies)),
        "p95_assurance_latency_ms": float(np.percentile(latencies, 95)),
    }
    metrics.update(
        _recovery_metrics(
            instant_service,
            recovery_threshold=recovery_threshold,
            disruption_end=disruption_end,
        )
    )
    return metrics
