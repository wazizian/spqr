import numpy as np
import pytest

from sklearn.datasets import make_blobs
from functools import partial

from .. import new_oracle
from .. import oracle
from .. import losses

N = 100
D = 10
P = 0.8
KAPPA = 2

gen = np.random.default_rng()

def log_sigmoid(u):
    exp_u_pos_part = np.exp(-np.clip(u, 0, None))
    exp_u_neg_part = np.exp( np.clip(u, None, 0))
    return np.clip(u, None, 0) - np.log(exp_u_neg_part + exp_u_pos_part)

def sigmoid(u):
    exp_u_pos_part = np.exp(-np.clip(u, 0, None))
    exp_u_neg_part = np.exp( np.clip(u, None, 0))
    return exp_u_neg_part/(exp_u_neg_part + exp_u_pos_part)

@pytest.fixture
def loss():
    def loss_func(w, x, y):
        n, d = x.shape
        assert y.shape == (n,)
        assert w.shape == (d,)

        return -log_sigmoid(y * np.dot(x, w))
    return loss_func

@pytest.fixture
def loss_grad():
    def grad_func(w, x, y):
        return - y[:,None] * x * sigmoid(-y * np.dot(x, w))[:, None]
    return grad_func

@pytest.fixture
def w():
    return gen.normal(size=D)

@pytest.fixture
def dataset():
    x, y = make_blobs(n_samples=N, n_features=D, centers=2)
    assert x.shape == (N, D)
    assert y.shape == (N,)
    return x, 2*y - 1

METHODS = ["f", "g"]

@pytest.mark.parametrize("method_name", METHODS) 
def test_oracleSmoothGradient(loss, loss_grad, w, dataset, method_name):
    old_oracle_class = oracle.OracleSmoothGradient(loss, loss_grad, P)
    old_method = getattr(old_oracle_class, method_name)
    old_res = old_method(w, *dataset)

    new_oracle_class = new_oracle.OracleSmoothGradient(loss, loss_grad, P)
    new_method = getattr(new_oracle_class, method_name)
    new_res = new_method(w, *dataset)

    assert old_res == pytest.approx(new_res)

@pytest.mark.parametrize("ambiguity_radius", [1/4, 1/2, 1, 3/2])
@pytest.mark.parametrize("sq_smoothing_parameter", [0.1, 1, 10, 100])
def test_oracleSmoothedDRO(loss, loss_grad, w, dataset, ambiguity_radius,
        sq_smoothing_parameter):

    oracle = new_oracle.OracleSmoothedWDRO(loss, loss_grad, ambiguity_radius,
            KAPPA, superquantile_smoothing_parameter=sq_smoothing_parameter) 
    assert oracle.alpha == ambiguity_radius/KAPPA
    assert oracle.alpha < 1

    x, y = dataset
    n = len(x)
    losses = loss(w, *dataset)
    inv_losses = loss(w, x, 1-y)
    diff_losses = inv_losses - losses
    norm = oracle.norm_oracle.f(w, x, y)
    wdro_loss = oracle.f(w, x, y)
    
    simplex_center_loss = (ambiguity_radius - KAPPA/2) * norm + np.mean(losses) \
            + oracle.alpha * 0.5 * np.mean(diff_losses)
    assert simplex_center_loss <= wdro_loss

    no_label_inv_loss = ambiguity_radius * norm + np.mean(losses) \
            - oracle.alpha * sq_smoothing_parameter * 0.25 * (1/n + 1)
    assert no_label_inv_loss <= wdro_loss

    max_diff_loss = max(diff_losses)
    max_q = min(1, 1/(n * oracle.alpha))
    one_label_inv_loss = (ambiguity_radius - KAPPA * max_q) * norm \
            + np.mean(losses) + oracle.alpha * max_q * max_diff_loss \
            - oracle.alpha*sq_smoothing_parameter * ((1/(2*n)-max_q)**2 + (0.5+max_q)**2)
    assert one_label_inv_loss <= wdro_loss

