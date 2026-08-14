import pandas as pd

from sc_cais.statistics import inferential_effect_summary, paired_assurance_effects


def test_inferential_summary_is_seed_paired():
    rows = []
    for seed in range(5):
        rows.append(
            {
                "scenario": "compound",
                "category": "compound",
                "severity": "very_high",
                "seed": seed,
                "agent": "agentic_heuristic",
                "assurance": "none",
                "cumulative_cost": 100 + seed,
                "resilience_index": 0.7,
                "normalized_resilience_loss": 0.3,
                "recovery_time": 8.0,
                "residual_violation_rate": 0.8,
                "final_service_level": 0.8,
            }
        )
        rows.append(
            {
                "scenario": "compound",
                "category": "compound",
                "severity": "very_high",
                "seed": seed,
                "agent": "agentic_heuristic",
                "assurance": "sc_cais",
                "cumulative_cost": 102 + seed,
                "resilience_index": 0.72,
                "normalized_resilience_loss": 0.28,
                "recovery_time": 6.0,
                "residual_violation_rate": 0.0,
                "final_service_level": 0.82,
            }
        )
    paired = paired_assurance_effects(pd.DataFrame(rows))
    summary = inferential_effect_summary(paired)
    residual = summary[summary["metric"] == "residual_violation_rate"].iloc[0]
    assert residual["n_pairs"] == 5
    assert residual["mean_delta"] < 0
    assert 0 <= residual["p_value"] <= 1
