from pathlib import Path

from sc_cais.config import load_benchmark_config


def test_default_config_loads():
    path = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"
    cfg = load_benchmark_config(path)
    assert cfg.horizon == 24
    assert "cyber_physical" in cfg.scenarios
