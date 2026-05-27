import numpy as np

from src.gmpb.transform import transform


def test_identity_when_no_irregularity():
    z = np.array([[[-3.0, -0.5, 0.0, 0.5, 3.0]]])
    out = transform(z, tau=np.array([0.0]), eta=np.zeros((1, 4)))
    assert np.allclose(out, z)


def test_zero_maps_to_zero():
    z = np.zeros((1, 1, 4))
    out = transform(z, tau=np.array([0.3]), eta=np.array([[15.0, 20.0, 12.0, 18.0]]))
    assert np.allclose(out, 0.0)


def test_sign_preserved():
    z = np.array([[[-4.0, -1.0, 2.0, 7.0]]])
    out = transform(z, tau=np.array([0.3]), eta=np.array([[15.0, 20.0, 12.0, 18.0]]))
    assert np.all(np.sign(out) == np.sign(z))


def test_finite_outputs():
    rng = np.random.default_rng(0)
    z = rng.uniform(-50, 50, size=(7, 3, 5))
    tau = rng.uniform(0, 0.4, size=3)
    eta = rng.uniform(10, 25, size=(3, 4))
    out = transform(z, tau, eta)
    assert np.all(np.isfinite(out))
