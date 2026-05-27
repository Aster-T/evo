"""真实数据静态问题适配层的单元测试。

覆盖：StaticObjective（FE 计数、done 触发、最小化负号语义、best 记录、收敛曲线），
投资组合解码器（softmax 归一、基数=K），以及三算法能在静态目标上跑满预算。
"""
import numpy as np
import pytest

from src.algorithms import ALGORITHMS
from src.problems.clustering import ClusteringProblem
from src.problems.portfolio import (PortfolioProblem, apply_cardinality,
                                     softmax_decode)
from src.problems.static import StaticObjective


# ---------------------------------------------------------------- StaticObjective
def test_static_fe_counting_and_done():
    obj = StaticObjective(lambda X: (X ** 2).sum(1), dim=3, lb=-1, ub=1, max_evals=100)
    assert not obj.done
    obj.evaluate(np.zeros((40, 3)))
    assert obj.fe == 40
    obj.evaluate(np.zeros((100, 3)))  # 超过预算
    assert obj.fe == 100 and obj.done


def test_static_minimization_sign_and_best():
    """最小化：func 越小越好；返回给算法的适应度应取负（越大越好）。"""
    obj = StaticObjective(lambda X: (X ** 2).sum(1), dim=2, lb=-5, ub=5,
                          max_evals=50, maximize=False)
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])  # f = 25, 0, 2
    fit = obj.evaluate(X)
    assert np.allclose(fit, [-25.0, 0.0, -2.0])         # 适应度 = -f
    assert obj.best_f == pytest.approx(0.0)              # 原始符号下最小值
    assert np.allclose(obj.best_x, [0.0, 0.0])


def test_static_maximization_sign():
    obj = StaticObjective(lambda X: X.sum(1), dim=2, lb=0, ub=1,
                          max_evals=10, maximize=True)
    fit = obj.evaluate(np.array([[1.0, 1.0], [0.0, 0.0]]))
    assert np.allclose(fit, [2.0, 0.0])
    assert obj.best_f == pytest.approx(2.0)


def test_static_changed_always_false():
    obj = StaticObjective(lambda X: X.sum(1), dim=1, lb=0, ub=1, max_evals=10)
    assert obj.changed is False
    obj.acknowledge_change()  # 不应抛错


def test_static_curve_monotone_for_minimization():
    obj = StaticObjective(lambda X: (X ** 2).sum(1), dim=2, lb=-5, ub=5,
                          max_evals=60, maximize=False)
    rng = np.random.default_rng(0)
    obj.evaluate(rng.uniform(-5, 5, (60, 2)))
    curve = obj.convergence_curve(60)
    assert np.all(np.diff(curve) <= 1e-9)  # best-so-far 单调不增


# ---------------------------------------------------------------- 解码器
def test_softmax_decode_simplex():
    g = np.array([[0.0, 1.0, 2.0, -1.0]])
    w = softmax_decode(g)
    assert w.shape == (1, 4)
    assert np.all(w >= 0) and w.sum() == pytest.approx(1.0)


def test_cardinality_keeps_exactly_K():
    W = np.array([[0.4, 0.3, 0.2, 0.1]])
    out = apply_cardinality(W, K=2)
    assert (out > 0).sum() == 2
    assert out.sum() == pytest.approx(1.0)
    assert out[0, 0] > 0 and out[0, 1] > 0  # 保留最大的两个
    assert out[0, 2] == 0 and out[0, 3] == 0


# ---------------------------------------------------------------- 端到端
@pytest.mark.parametrize("name", ["GA", "PSO", "DE"])
def test_algorithms_run_on_portfolio(name):
    rng = np.random.default_rng(0)
    mu = rng.normal(0.01, 0.005, 8)
    A = rng.normal(size=(8, 8))
    Sigma = A @ A.T / 8 + np.eye(8) * 0.01
    prob = PortfolioProblem(mu, Sigma, K=4, max_evals=600)
    obj = prob.make_objective()
    algo = ALGORITHMS[name](obj.dim, obj.lb, obj.ub, np.random.default_rng(1))
    algo.run(obj)
    assert obj.done and obj.fe == obj.max_evals
    m = prob.metrics_of(obj.best_x)
    assert m["cardinality"] <= 4
    assert np.isfinite(m["sharpe"])


@pytest.mark.parametrize("name", ["GA", "PSO", "DE"])
def test_algorithms_run_on_clustering(name):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, (30, 4)) for c in (-2, 2)])
    prob = ClusteringProblem(X, K=2, max_evals=600)
    obj = prob.make_objective()
    algo = ALGORITHMS[name](obj.dim, obj.lb, obj.ub, np.random.default_rng(1))
    algo.run(obj)
    assert obj.done
    assert prob.metrics_of(obj.best_x)["sse"] >= 0
