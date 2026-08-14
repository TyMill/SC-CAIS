from sc_cais.benchmark import BenchmarkRunner
from sc_cais.scenarios import default_scenarios


def test_small_benchmark_returns_expected_design_cells():
    scenario = [next(s for s in default_scenarios() if s.name == "normal")]
    frame = BenchmarkRunner(horizon=3, seeds=(0,)).run(
        scenarios=scenario,
        agent_names={"agentic_heuristic"},
        assurance_names={"none", "sc_cais"},
    )
    assert len(frame) == 2
    assert set(frame["assurance"]) == {"none", "sc_cais"}
