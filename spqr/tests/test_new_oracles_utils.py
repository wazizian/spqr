import numpy as np
import pytest
from .. import new_oracles_utils
from .. import oracles_utils

N = 100
MAX_VALUE = 10
P = 0.8
RHO = 10

gen = np.random.default_rng()

@pytest.fixture
def v():
    return gen.integers(0, MAX_VALUE, size=N).astype(np.float64)

@pytest.fixture
def sorted_index(v):
    return np.argsort(v)

@pytest.fixture
def weights(v):
    return np.ones_like(v)/len(v)

@pytest.fixture
def lmbda():
    return gen.chisquare(1)

def test_fast_theta_prime(v, sorted_index, weights, lmbda):
    old_res = oracles_utils.fast_theta_prime(lmbda, v, sorted_index, P, RHO)
    new_res = new_oracles_utils.fast_theta_prime(lmbda, v, weights,
            weights/(1-P), sorted_index, P, RHO)
    assert old_res == pytest.approx(new_res)

def test_fast_find_lmbda(v, sorted_index, weights):
    old_res = oracles_utils.fast_find_lmbda(v, sorted_index, P, RHO)
    new_res = new_oracles_utils.fast_find_lmbda(v, weights, weights/(1-P),  sorted_index, P, RHO)
    assert old_res == pytest.approx(new_res)

def test_fast_projection(v, sorted_index, weights):
    old_res = oracles_utils.fast_projection(v, P, RHO)
    new_res = new_oracles_utils.fast_projection(v, weights, P, RHO)
    assert old_res == pytest.approx(new_res)

