"""
.. module:: oracle
   :synopsis: Module with auxiliary numba functions for oracle computations.

.. moduleauthor:: Yassine LAGUEL
"""

from numba import njit
import numpy as np
from .measures import quantile


@njit
def compute_subgradient(sequence_losses, sequence_gradients, p):
    """Computes a subgradient of :math:`w \mapsto Cvar \circ L(w)` for the dataset :math:`(x,y)`
    """

    raise NotImplementedError

    n = len(sequence_losses)

    q_p_u = quantile(p, sequence_losses)

    i_p = np.where(sequence_losses == q_p_u)[0]
    j_p = np.where(sequence_losses > q_p_u)[0]

    alpha = (np.ceil(p*n)/n - p)/len(i_p)

    res = 1.0/(1.0 - p) * ((1.0/n) * np.sum(sequence_gradients[j_p], axis=0)
                           + alpha * np.sum(sequence_gradients[i_p] , axis=0))

    return res


@njit
def fast_projection(u, weights, p, rho, simplex_center):
    if p == 0:
        return weights

    n = len(u)
    c = weights/(1 - p)
    mu = 2 * rho
    v = u + mu * simplex_center

    # sorts the coordinate of v
    sorted_index = np.argsort(v)

    # finds the zero of the function theta prime
    lmbda = fast_find_lmbda(v, weights, sorted_index, p, rho, c=c)

    # instantiate the output
    res = np.zeros(n, dtype=np.float64)

    # fills the coordinate of the output
    counter = n - 1
    while counter >= 0:
        if lmbda > v[sorted_index[counter]]:
            break
        elif lmbda > v[sorted_index[counter]] - mu * c[sorted_index[counter]]:
            res[sorted_index[counter]] = (v[sorted_index[counter]] - lmbda) / mu
        else:
            res[sorted_index[counter]] = c[sorted_index[counter]]
        counter -= 1
    return res


@njit
def fast_find_lmbda(v, weights, sorted_index, p, rho, c=None):
    n = len(v)
    mu = 2*rho
    if c is None:
        c = weights/(1-p)
    set_p = np.sort(np.concatenate((v, v - mu * c)))

    a= 0
    b= 2 * n - 1
    m = (a + b) // 2
    while (b - a) > 1:
        if fast_theta_prime(set_p[m], v, weights, sorted_index, p, rho, c=c) > 0:
            b = m
        elif fast_theta_prime(set_p[m], v, weights, sorted_index, p, rho, c=c) < 0:
            a = m
        else:
            return set_p[m]
        m = (a + b) // 2

    if fast_theta_prime(set_p[a], v, weights, sorted_index, p, rho, c=c) == 0.:
        return set_p[a]
    elif fast_theta_prime(set_p[b], v, weights, sorted_index, p, rho, c=c) == 0.:
        return set_p[b]
    else:

        res = set_p[a] - (fast_theta_prime(set_p[a], v, weights, sorted_index, p, rho, c=c) * (set_p[b] - set_p[a])) / \
              (fast_theta_prime(set_p[b], v, weights, sorted_index, p, rho, c=c) -
               fast_theta_prime(set_p[a], v, weights, sorted_index, p, rho, c=c))
        return res


@njit
def fast_theta_prime(lmbda, v, weights, sorted_index, p, rho, c=None):
    n = len(v)
    mu = 2*rho
    if c is None:
        c = weights/(1-p)
    res = 1.0
    counter = n - 1
    while counter >= 0:
        if lmbda >= v[sorted_index[counter]]:
            break
        elif lmbda >= v[sorted_index[counter]] - mu * c[sorted_index[counter]]:
            res -= (v[sorted_index[counter]] - lmbda) / mu
        else:
            res -= c[sorted_index[counter]]
        counter -= 1

    return res
