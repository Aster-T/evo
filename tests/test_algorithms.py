"""测试 6 个算法（连续 ×3，TSP ×3）：能跑满预算并明显优于随机搜索。"""
import numpy as np
import pytest

from src.algorithms.continuous import CONTINUOUS_ALGORITHMS
from src.algorithms.tsp import TSP_ALGORITHMS
from src.benchmarks import FUNCTIONS, ContinuousObjective
from src.tsp import TSPProblem


def _random_search_continuous(name, dim, max_evals, seed):
    obj = ContinuousObjective(FUNCTIONS[name], dim=dim, max_evals=max_evals)
    rng = np.random.default_rng(seed)
    while not obj.done:
        obj.evaluate(rng.uniform(obj.lb, obj.ub, (50, dim)))
    return obj.best_f


@pytest.mark.parametrize("name", list(CONTINUOUS_ALGORITHMS))
def test_continuous_runs_and_beats_random(name):
    rs = _random_search_continuous("Sphere", 10, 5000, 0)
    obj = ContinuousObjective(FUNCTIONS["Sphere"], dim=10, max_evals=5000)
    CONTINUOUS_ALGORITHMS[name](obj, np.random.default_rng(0)).run()
    assert obj.done and obj.fe == obj.max_evals
    assert obj.best_f < rs                      # 优于随机搜索


def test_acor_solves_sphere():
    """ACOR 在单峰 Sphere 上应收敛到接近 0。"""
    obj = ContinuousObjective(FUNCTIONS["Sphere"], dim=10, max_evals=10000)
    CONTINUOUS_ALGORITHMS["ACO"](obj, np.random.default_rng(0)).run()
    assert obj.best_f < 1e-6


def _random_search_tsp(inst, max_evals, seed):
    p = TSPProblem(inst, max_evals=max_evals)
    rng = np.random.default_rng(seed)
    while not p.done:
        p.evaluate(rng.permutation(p.n))
    return p.best_len


@pytest.mark.parametrize("name", list(TSP_ALGORITHMS))
def test_tsp_runs_and_beats_random(name):
    rs = _random_search_tsp("berlin52", 3000, 0)
    p = TSPProblem("berlin52", max_evals=3000)
    TSP_ALGORITHMS[name](p, np.random.default_rng(0)).run()
    assert p.done
    assert p.best_len < rs                      # 优于随机搜索


def test_aco_near_optimal_berlin52():
    """ACO 是 TSP 本职算法，berlin52 上 gap 应较小（<15%）。"""
    p = TSPProblem("berlin52", max_evals=10000)
    TSP_ALGORITHMS["ACO"](p, np.random.default_rng(0)).run()
    assert p.gap() < 15.0
