import numpy as np
import pytest

from src.gmpb import GMPBConfig, Benchmark


@pytest.fixture
def small_cfg():
    return GMPBConfig(dim=5, peak_number=5, change_frequency=200, environment_number=4)


def test_peak_center_equals_height(small_cfg):
    b = Benchmark(small_cfg, seed=0)
    p = b.envs[0]
    f = b._fitness(p.positions, p)
    assert np.allclose(f, p.heights, atol=1e-9)


def test_optimum_is_max_height(small_cfg):
    b = Benchmark(small_cfg, seed=0)
    for env, opt in zip(b.envs, b.optima):
        assert np.isclose(opt, env.heights.max())


def test_fitness_never_exceeds_optimum(small_cfg):
    b = Benchmark(small_cfg, seed=0)
    rng = np.random.default_rng(0)
    p = b.envs[0]
    X = rng.uniform(-50, 50, size=(2000, small_cfg.dim))
    f = b._fitness(X, p)
    assert np.all(f <= b.optima[0] + 1e-9)


def test_fe_counting_and_changes(small_cfg):
    b = Benchmark(small_cfg, seed=1)
    rng = np.random.default_rng(0)
    changes = 0
    while not b.done:
        b.evaluate(rng.uniform(-50, 50, size=(25, small_cfg.dim)))
        if b.changed:
            changes += 1
            b.acknowledge_change()
    assert b.fe == b.max_evals
    assert changes == small_cfg.environment_number - 1


def test_error_history_nonneg_and_filled(small_cfg):
    b = Benchmark(small_cfg, seed=1)
    rng = np.random.default_rng(0)
    while not b.done:
        b.evaluate(rng.uniform(-50, 50, size=(25, small_cfg.dim)))
        if b.changed:
            b.acknowledge_change()
    assert np.all(np.isfinite(b.error_history))
    assert np.all(b.error_history >= 0)
    assert b.offline_error >= 0


def test_reproducible_environments():
    cfg = GMPBConfig(dim=4, peak_number=6, change_frequency=100, environment_number=5)
    a = Benchmark(cfg, seed=7)
    b = Benchmark(cfg, seed=7)
    for ea, eb in zip(a.envs, b.envs):
        assert np.allclose(ea.positions, eb.positions)
        assert np.allclose(ea.heights, eb.heights)
        assert np.allclose(ea.rotation, eb.rotation)


def test_running_min_resets_per_environment(small_cfg):
    """同一环境内误差运行最小值单调不增；换环境后允许回升。"""
    b = Benchmark(small_cfg, seed=2)
    rng = np.random.default_rng(0)
    cf = small_cfg.change_frequency
    while not b.done:
        b.evaluate(rng.uniform(-50, 50, size=(20, small_cfg.dim)))
        if b.changed:
            b.acknowledge_change()
    # 每个环境段内 error_history 应单调不增
    for e in range(small_cfg.environment_number):
        seg = b.error_history[e * cf:(e + 1) * cf]
        assert np.all(np.diff(seg) <= 1e-9)
