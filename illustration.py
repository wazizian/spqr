import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from spqr import DRLogisticRegression, WDRLogisticRegression
from classifier_comparison import plot_classifier_comparison

import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from General import GeneralWDRLogisticRegression

gen = np.random.default_rng()

def illustration():
    lst_rho = [0.001, 0.01, 0.1]
    kappa = 1
    mu = 1
    mu_norm = 1
    names = [f"$\\rho={rho}$" for rho in lst_rho]
    classifiers = [WDRLogisticRegression(rho, kappa, mu, mu_norm) for rho in lst_rho]
    _, _, centers = make_blobs(centers=2, cluster_std=2, return_centers=True)
    datasets = [
            (make_moons(noise=0.1), make_moons(noise=0.1)),
            (make_blobs(centers=centers, cluster_std=2), make_blobs(centers=centers, cluster_std=2)),
            (make_blobs(centers=centers, cluster_std=4), make_blobs(centers=centers, cluster_std=4)),
            ]
    plot_classifier_comparison(names, classifiers, datasets, save='illustration.pdf')

def perturbed_dataset(perturb_prob, dataset):
    x, y = dataset
    
    perturb = gen.binomial(1, perturb_prob, size=y.shape)
    y[perturb*y == 1] = 0

    return x, y

def perturbed_illustration():
    N = 500
    train_split = 0.75
    n_train = int(0.75 * N)
    n_test = N - n_train
    centers = [np.array([-3,-3]), np.array([4,4])]
    dataset = lambda n : make_blobs(n_samples=n, centers=centers, cluster_std=2)
    
    perturb_probs = [0.4, 0.5]
    datasets = [
            (perturbed_dataset(perturb_prob, dataset(n_train)), dataset(n_test))
            for perturb_prob in perturb_probs
            ]
    names = ["Logistic Regression", "WDRO Logistic Regression"]
    rho = 0.02
    kappa = 0.1
    mu = 0.1
    classifiers = [
            LogisticRegression(penalty='none'),
            #DRLogisticRegression(p=1-0.5*rho/kappa, mu=mu),
            WDRLogisticRegression(rho=rho, kappa=kappa, mu=mu, mu_norm=mu)
            ]
    levels = [0., 0.25, 0.45, 0.5, 0.55, 0.75, 1.]
    plot_classifier_comparison(names, classifiers, datasets, levels=levels, save='perturbed.pdf')

def spatially_perturbed_illustration(entropic=False):
    N = 500
    train_split = 0.75
    n_train = int(0.75 * N)
    n_test = N - n_train
    pos = 4
    centers = [np.array([-pos,-pos]), np.array([pos,pos])]
    dataset = lambda n,sdevs : make_blobs(n_samples=n, centers=centers, cluster_std=sdevs)
    sdevs = [(2.5, 5), (1, 5)]#, (0.1, 5)]
    datasets = [
            (dataset(n_train, (sdev_1, sdev_2)), dataset(n_test, (sdev_2, sdev_1)))
            for (sdev_1, sdev_2) in sdevs
            ]
    names = ["Logistic Regression", "WDR0 Logistic Regression"]
    rho = 2*4**2
    kappa = 1000
    mu = 0.1
    classifiers = [
            LogisticRegression(penalty='none'),
            #DRLogisticRegression(p=0.5, mu=mu),
            WDRLogisticRegression(rho=rho, kappa=kappa, mu=mu, mu_norm=mu),
            ]
    filename = 'spatially.pdf'
    if entropic:
        names.append("Entropic WDRO Logistic Regression")
        classifiers.append(GeneralWDRLogisticRegression(rho=rho, m=300, epochs=1000, lr=1e-2, scheduler_verbose=True))
        filename ='entropic_spatially.pdf'

    levels = [0., 0.25, 0.45, 0.5, 0.55, 0.75, 1.]
    plot_classifier_comparison(names, classifiers, datasets, levels=levels, save=filename)

if __name__ == '__main__':
    #perturbed_illustration()
    spatially_perturbed_illustration(entropic=False)
    spatially_perturbed_illustration(entropic=True)







