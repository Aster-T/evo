"""测试 TSPLIB 加载与 TSP 问题封装。"""
import numpy as np
import pytest

from src.tsp import OPTIMA, TSPProblem, distance_matrix, load_coords


def test_load_berlin52():
    coords = load_coords("berlin52")
    assert coords.shape == (52, 2)
    D = distance_matrix(coords)
    assert D.shape == (52, 52)
    assert np.allclose(D, D.T)              # 对称
    assert np.all(np.diag(D) == 0)          # 自距为 0


def test_tour_length_and_fe():
    p = TSPProblem("berlin52", max_evals=10)
    tour = np.arange(p.n)
    L = p.tour_length(tour)
    assert L > 0
    # tour_length 不计 FE；evaluate 计 FE
    assert p.fe == 0
    p.evaluate(tour)
    assert p.fe == 1 and p.best_len == pytest.approx(L)


def test_gap_against_known_optimum():
    """最优巡回长度等于已知最优时 gap 应为 0。"""
    p = TSPProblem("berlin52", max_evals=5)
    assert p.optimum == OPTIMA["berlin52"] == 7542
    g = p.gap(length=float(p.optimum))
    assert g == pytest.approx(0.0)


def test_evaluate_budget_truncation():
    p = TSPProblem("eil51", max_evals=3)
    for _ in range(10):
        p.evaluate(np.random.default_rng(0).permutation(p.n))
    assert p.fe == 3 and p.done
