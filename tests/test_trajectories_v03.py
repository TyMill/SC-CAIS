from sc_cais.benchmark import BenchmarkRunner
from sc_cais.scenarios import default_scenarios


def test_run_with_trajectories_marks_phases():
    scenario = next(item for item in default_scenarios() if item.name == "supplier_failure")
    summary, trajectory = BenchmarkRunner(horizon=18, seeds=(0,)).run_with_trajectories(
        scenarios=[scenario], agent_names={"agentic_heuristic"}, assurance_names={"sc_cais"}
    )
    assert len(summary) == 1
    assert len(trajectory) == 18
    assert {"pre", "disruption", "recovery"}.issubset(set(trajectory["phase"]))
