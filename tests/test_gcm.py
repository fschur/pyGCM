import numpy as np
import pytest
from pyGCM import GCMTest

def test_continuous_gcm_linear():
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 1)
    E = 0.5 * X + 0.1 * np.random.randn(n, 1)
    Y = 0.8 * X + 0.1 * np.random.randn(n, 1)

    gcm = GCMTest(get_resid='linear')
    stat, p_val = gcm.fit(E, X, Y)
    assert np.isclose(p_val, 0.733715, atol=0.1), "p-value out of range!"


def test_continuous_gcm_gp():
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 1)
    E = 0.5 * X + 0.1 * np.random.randn(n, 1)
    Y = 0.8 * X + 0.1 * np.random.randn(n, 1)

    gcm = GCMTest(get_resid='GP')
    stat, p_val = gcm.fit(E, X, Y)
    assert np.isclose(p_val, 0.75973, atol=0.1), "p-value out of range!"


def test_categorical_gcm():
    np.random.seed(42)
    n = 1000
    E_continuous = 0.5 * np.random.randn(n, 1)  # some continuous E
    X_categorical = np.random.choice([0, 1, 2], size=(n, 1))
    Y_continuous = 0.5 * X_categorical + 0.1 * np.random.randn(n, 1)

    gcm = GCMTest(get_resid='categorical')
    stat, p_val = gcm.fit(E_continuous, X_categorical, Y_continuous)
    assert np.isclose(p_val, 0.81802, atol=0.1), "p-value out of range!"
