"""Statistical comparison helpers for benchmark campaigns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DEFAULT_METRICS: tuple[str, ...] = (
    "cumulative_cost",
    "resilience_index",
    "normalized_resilience_loss",
    "recovery_time",
    "residual_violation_rate",
    "final_service_level",
)


def paired_assurance_effects(
    frame: pd.DataFrame,
    *,
    baseline: str = "none",
    treatment: str = "sc_cais",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Compute seed-paired treatment minus baseline effects by scenario and agent."""
    keys = ["scenario", "category", "severity", "seed", "agent"]
    if "uncertainty" in frame.columns:
        keys.insert(3, "uncertainty")
    available = tuple(metric for metric in metrics if metric in frame.columns)
    left = frame[frame["assurance"] == baseline][keys + list(available)].copy()
    right = frame[frame["assurance"] == treatment][keys + list(available)].copy()
    paired = right.merge(left, on=keys, suffixes=("_treatment", "_baseline"), validate="one_to_one")
    for metric in available:
        paired[f"delta_{metric}"] = paired[f"{metric}_treatment"] - paired[f"{metric}_baseline"]
    paired["baseline_assurance"] = baseline
    paired["treatment_assurance"] = treatment
    return paired


def bootstrap_mean_ci(
    values: np.ndarray | pd.Series,
    *,
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: int = 12345,
) -> tuple[float, float, float]:
    """Return mean and percentile bootstrap confidence interval."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(x.mean()), float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def paired_dz(values: np.ndarray | pd.Series) -> float:
    """Cohen's dz for paired differences; 0 when variance is numerically zero."""
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd <= 1e-15:
        return 0.0 if abs(float(np.mean(x))) <= 1e-15 else float("inf") * np.sign(float(np.mean(x)))
    return float(np.mean(x) / sd)


def _wilcoxon_safe(values: np.ndarray) -> tuple[float, float]:
    nonzero = values[np.abs(values) > 1e-15]
    if nonzero.size == 0:
        return 0.0, 1.0
    result = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def inferential_effect_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Long-form paired inference table for article reporting.

    Reports mean/median paired difference, deterministic bootstrap CI, Wilcoxon
    signed-rank p-value and Cohen's dz. The unit of pairing is the seed within
    scenario/agent (and uncertainty, when present).
    """
    delta_cols = [column for column in paired.columns if column.startswith("delta_")]
    group_cols = ["scenario", "agent"]
    if "uncertainty" in paired.columns:
        group_cols.insert(1, "uncertainty")
    rows: list[dict[str, object]] = []
    for group_key, group in paired.groupby(group_cols, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        identity = dict(zip(group_cols, key_values, strict=True))
        for col in delta_cols:
            values = group[col].to_numpy(dtype=float)
            mean, lo, hi = bootstrap_mean_ci(values)
            statistic, pvalue = _wilcoxon_safe(values)
            rows.append(
                {
                    **identity,
                    "baseline_assurance": str(group["baseline_assurance"].iloc[0]),
                    "treatment_assurance": str(group["treatment_assurance"].iloc[0]),
                    "metric": col.removeprefix("delta_"),
                    "n_pairs": int(values.size),
                    "mean_delta": mean,
                    "median_delta": float(np.median(values)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "wilcoxon_w": statistic,
                    "p_value": pvalue,
                    "cohen_dz": paired_dz(values),
                }
            )
    return pd.DataFrame(rows)


def effect_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wide paired mean effects with deterministic bootstrap CIs."""
    delta_cols = [c for c in paired.columns if c.startswith("delta_")]
    rows: list[dict[str, object]] = []
    for (scenario, agent), group in paired.groupby(["scenario", "agent"]):
        row: dict[str, object] = {"scenario": scenario, "agent": agent, "n_pairs": len(group)}
        for col in delta_cols:
            mean, lo, hi = bootstrap_mean_ci(group[col].to_numpy())
            row[f"{col}_mean"] = mean
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)
