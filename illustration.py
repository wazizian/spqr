import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from spqr import WDRLogisticRegression
from classifier_comparison import plot_classifier_comparison

def main():
	lst_rho = [0.001, 0.01, 0.1, 0.9]
	kappa = 1
	mu = 1
	mu_norm = 1
	names = [f"WDRLogisticRegression({rho})" for rho in lst_rho]
	classifiers = [WDRLogisticRegression(rho, kappa, mu, mu_norm) for rho in lst_rho]
	datasets = [
			(make_moons(noise=0.1), make_moons(noise=0.1)),
			(make_blobs(centers=2, cluster_std=4), make_blobs(centers=2, cluster_std=4)),
			]
	plot_classifier_comparison(names, classifiers, datasets)

if __name__ == '__main__':
	main()







