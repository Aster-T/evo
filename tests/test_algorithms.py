import numpy as np
import pytest

from src.gmpb import GMPBConfig, Benchmark
from src.algorithms import ALGORITHMS


@pytest.fixture
def cfg():
    return GMPBConfig(dim=5, peak_number=5, change_frequency=1000, environment_number=5)


@pytest.mark.parametrize("name", ["GA", "PSO", "DE"])
def test_runs_to_budget(name, cfg):
    bench = Benchmark(cfg, seed=0)
    algo = ALGORITHMS[name](cfg.dim, cfg.min_coordinate, cfg.max_coordinate,
                            np.random.default_rng(0))
    algo.run(bench)
    assert bench.done
    assert bench.fe == bench.max_evals
    assert np.isfinite(bench.offline_error)
    assert bench.offline_error >= 0


@pytest.mark.parametrize("name", ["GA", "PSO", "DE"])
def test_beats_random_search(name, cfg):
    """优化器的 offline error 应明显低于纯随机搜索。"""
    # 随机搜索基线
    rb = Benchmark(cfg, seed=1)
    rng = np.random.default_rng(123)
    while not rb.done:
        rb.evaluate(rng.uniform(-50, 50, size=(50, cfg.dim)))
        if rb.changed:
            rb.acknowledge_change()

    ob = Benchmark(cfg, seed=1)
    algo = ALGORITHMS[name](cfg.dim, cfg.min_coordinate, cfg.max_coordinate,
                            np.random.default_rng(123))
    algo.run(ob)
    assert ob.offline_error < rb.offline_error
