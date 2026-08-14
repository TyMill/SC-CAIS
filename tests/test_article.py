from sc_cais.article import aggregate_benchmark
from sc_cais.benchmark import BenchmarkRunner
from sc_cais.scenarios import default_scenarios


def test_article_aggregation_has_mean_and_sd():
    scenario = [next(s for s in default_scenarios() if s.name == "compound")]
    frame = BenchmarkRunner(horizon=3, seeds=(0, 1)).run(
        scenarios=scenario,
        agent_names={"agentic_heuristic"},
        assurance_names={"none", "sc_cais"},
    )
    agg = aggregate_benchmark(frame)
    assert "cumulative_cost_mean" in agg.columns
    assert "cumulative_cost_sd" in agg.columns
