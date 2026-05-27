import numpy as np
import pytest

from src.gmpb.rotation import givens_product, random_orthogonal


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_givens_product_orthogonal(d):
    R = givens_product(0.37, d)
    assert np.allclose(R @ R.T, np.eye(d), atol=1e-10)
    assert np.isclose(abs(np.linalg.det(R)), 1.0)


def test_givens_zero_angle_is_identity():
    assert np.allclose(givens_product(0.0, 6), np.eye(6))


@pytest.mark.parametrize("d", [2, 4, 8])
def test_random_orthogonal(d):
    rng = np.random.default_rng(0)
    Q = random_orthogonal(d, rng)
    assert np.allclose(Q @ Q.T, np.eye(d), atol=1e-10)


def test_givens_matches_naive_matmul():
    """优化后的按列更新应等价于朴素的 Givens 矩阵连乘。"""
    d, theta = 5, 0.6
    c, s = np.cos(theta), np.sin(theta)
    out = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            g = np.eye(d)
            g[i, i] = c; g[j, j] = c; g[i, j] = s; g[j, i] = -s
            out = out @ g
    assert np.allclose(out, givens_product(theta, d))
