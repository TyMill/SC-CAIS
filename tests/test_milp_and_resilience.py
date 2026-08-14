from sc_cais.agents import RollingHorizonMILPAgent
from sc_cais.assurance import RuntimeAssurance
from sc_cais.environment import SupplyChainEnv
from sc_cais.evaluation import summarize
from sc_cais.network import create_reference_network
from sc_cais.scenarios import default_scenarios, disruption_window


def test_rolling_horizon_milp_runs() -> None:
    env = SupplyChainEnv(create_reference_network(), seed=3)
    results = env.run(RollingHorizonMILPAgent(), RuntimeAssurance(), 5)
    assert len(results) == 5
    assert results[-1].cumulative_cost >= 0.0


def test_observation_noise_is_deterministic_within_step() -> None:
    env = SupplyChainEnv(create_reference_network(), seed=7, observation_noise=0.2)
    first = env.observe()
    second = env.observe()
    assert first.nodes == second.nodes
    assert first.data_quality == second.data_quality


def test_recovery_metrics_present() -> None:
    scenario = next(s for s in default_scenarios() if s.name == "demand_surge")
    _, end = disruption_window(scenario)
    env = SupplyChainEnv(create_reference_network(), seed=1, disruptions=scenario.disruptions)
    results = env.run(RollingHorizonMILPAgent(), RuntimeAssurance(), 24)
    metrics = summarize(results, disruption_end=end)
    assert "resilience_loss_area" in metrics
    assert "recovery_time" in metrics
    assert 0.0 <= metrics["normalized_resilience_loss"] <= 1.0
