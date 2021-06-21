import numpy as np
import pytest

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ..wdr_estimators import WDRLogisticRegression
from ..estimators     import  DRLogisticRegression

N = 1000
D = 50

@pytest.fixture
def dataset():
    x, y, centers = make_blobs(n_samples=N, n_features=D, centers=2, return_centers=True)

    assert x.shape == (N, D)
    assert y.shape == (N,)
    assert centers.shape == (2, D)

    splitted_dataset = train_test_split(x, y)

    return *splitted_dataset, centers

@pytest.mark.parametrize("rho,kappa", [
    (0.1, 1),
    (1/4,2),
    (1/2,2),
    (1,2), 
    ])
@pytest.mark.parametrize("mu", [0.1, 1, 10, 100])
@pytest.mark.parametrize("mu_norm", [0.1, 1, 10, 100])
def test_WDRLogisticRegression(dataset, rho, kappa, mu, mu_norm):
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
