from sc_cais.benchmark import BenchmarkRunner
from sc_cais.scenarios import default_scenarios
from sc_cais.statistics import effect_summary, paired_assurance_effects


def test_paired_effects_are_seed_aligned():
    scenario = [next(s for s in default_scenarios() if s.name == "normal")]
    frame = BenchmarkRunner(horizon=3, seeds=(0, 1)).run(
        scenarios=scenario,
        agent_names={"agentic_heuristic"},
        assurance_names={"none", "sc_cais"},
    )
    paired = paired_assurance_effects(frame)
    assert len(paired) == 2
    summary = effect_summary(paired)
    assert summary.iloc[0]["n_pairs"] == 2
