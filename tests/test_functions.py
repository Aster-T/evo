"""测试连续测试函数与目标封装。"""
import numpy as np
import pytest

from src.benchmarks import FUNCTIONS, ContinuousObjective


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_optimum_value_at_optimum(name):
    """各函数在全局最优点应取 f*=0。Rosenbrock 最优在 x=1，其余在 x=0。"""
    spec = FUNCTIONS[name]
    d = 8
    x = np.ones((1, d)) if name == "Rosenbrock" else np.zeros((1, d))
    assert spec.func(x)[0] == pytest.approx(spec.optimum, abs=1e-9)


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_batch_shape_and_nonnegative(name):
    spec = FUNCTIONS[name]
    rng = np.random.default_rng(0)
    X = rng.uniform(spec.lb, spec.ub, (16, 10))
    f = spec.func(X)
    assert f.shape == (16,)
    assert np.all(f >= spec.optimum - 1e-6)  # 这些函数最小值即 f*


def test_objective_fe_counting_and_best():
    obj = ContinuousObjective(FUNCTIONS["Sphere"], dim=5, max_evals=100)
    obj.evaluate(np.zeros((1, 5)))         # f=0，全局最优
    assert obj.best_f == pytest.approx(0.0)
    assert obj.fe == 1
    obj.evaluate(np.ones((200, 5)))        # 超预算，FE 截断
    assert obj.fe == 100 and obj.done
    assert obj.best_f == pytest.approx(0.0)


def test_curve_monotone_nonincreasing():
    obj = ContinuousObjective(FUNCTIONS["Rastrigin"], dim=5, max_evals=80)
    rng = np.random.default_rng(1)
    obj.evaluate(rng.uniform(-5.12, 5.12, (80, 5)))
    curve = obj.convergence_curve(80)
    assert np.all(np.diff(curve) <= 1e-9)
