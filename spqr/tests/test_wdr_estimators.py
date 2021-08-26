import numpy as np
import pytest

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from ..wdr_estimators import WDRLogisticRegression
from ..estimators     import  DRLogisticRegression

N = 500
D = 10

@pytest.fixture
def dataset():
    x, y, centers = make_blobs(n_samples=N, n_features=D, centers=2, return_centers=True)

    assert x.shape == (N, D)
    assert y.shape == (N,)
    assert centers.shape == (2, D)

    splitted_dataset = train_test_split(x, y)

    return *splitted_dataset, centers

@pytest.fixture
def perturbed_dataset(dataset):
    x_train, x_test, y_train, y_test, centers = dataset

    shift = .2 * (centers[0] - centers[1]) 
    x_test[y_test == 1] += shift[None,:]

    return x_train, x_test, y_train, y_test, centers

@pytest.mark.parametrize("rho,kappa", [
    (0.1, 1),
    (1/4,2),
    (1/2,2),
    (1,2), 
    ])
@pytest.mark.parametrize("mu", [0.1, 1, 10, 100])
@pytest.mark.parametrize("mu_norm", [0.1, 1, 10, 100])
def test_interpolate_WDRLogisticRegression(dataset, rho, kappa, mu, mu_norm):
    x_train, x_test, y_train, y_test, centers = dataset

    estimator = WDRLogisticRegression(rho, kappa, mu, mu_norm)
    estimator.fit(x_train, y_train, verbose_mode=True)

    assert estimator.algorithm.bfgs_result_object.success, \
           estimator.algorithm.bfgs_result_object.message
    
    for x, y, label in [(x_train, y_train, 'train'), (x_test, y_test, 'test')]:
        y_predicted = estimator.predict(x)
        acc = accuracy_score(y, y_predicted, normalize=False)
        print(f"Accuracy {label} = {acc}/{len(x)}")
        assert acc == len(x)

@pytest.mark.parametrize("rho,kappa", [(5,5)])
@pytest.mark.parametrize("mu", [1, 10])
@pytest.mark.parametrize("mu_norm", [1, 10])
def test_shift_WDRLogisticRegression(perturbed_dataset, rho, kappa, mu, mu_norm):
    x_train, x_test, y_train, y_test, centers = perturbed_dataset

    estimator = WDRLogisticRegression(rho, kappa, mu, mu_norm)
    estimator.fit(x_train, y_train)

    plain_estimator = LogisticRegression(penalty='none')
    plain_estimator.fit(x_train, y_train)
    
    y_predicted = estimator.predict(x_test)
    acc = accuracy_score(y_test, y_predicted, normalize=False)

    y_predicted = plain_estimator.predict(x_test)
    acc_plain = accuracy_score(y_test, y_predicted, normalize=False)

    print(f"Accuracy {acc} vs plain accuracy {acc_plain}")
    assert acc > acc_plain
