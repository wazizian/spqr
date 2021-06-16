import numpy as np
from abc import ABC, abstractmethod

from .new_oracles_utils import *

class AbstractOracle(ABC):
    @abstractmethod
    def f(self, w, x, y) -> np.ndarray:
        pass

    @abstractmethod
    def g(self, w, x, y) -> np.ndarray:
        pass

class OracleSmoothGradient(AbstractOracle):
    def __init__(self, loss, loss_grad, p, smoothing_parameter=1000.0,
            extra_samples=np.array([]), extra_samples_weights=np.array([]), mixing_param=0):
        super().__init__()
        self.L_f = loss
        self.L_f_prime = loss_grad
        self.p = p
        self.smoothing_parameter = smoothing_parameter # corresponds to 2*mu=rho
        self.extra_samples = extra_samples
        self.extra_samples_weights = extra_samples_weights
        self.mixing_param = mixing_param

    def f(self, w, x, y):
        """Computes the value of the smooth approximation :math:`w \mapsto Cvar \circ L(w)`
        """
        f, g = self._smooth_superquantile(w, x, y)
        
        return f

    def g(self, w, x, y):
        """Computes the gradient of the smooth approximation of :math:`w \mapsto Cvar \circ L(w)`
        """
        f, g = self._smooth_superquantile(w, x, y)

        return g

    def seq_loss(self, w, x, y):
        n = len(x)
        return np.concatenate((np.asarray(self.L_f(w, x, y), dtype=np.float64), self.extra_samples)), \
               np.concatenate(((1-self.mixing_param)*np.ones(n)/n, self.mixing_param*self.extra_samples_weights))

    def seq_grad(self, w, x, y):
        return np.asarray(self.L_f_prime(w, x, y), dtype=np.float64)

    def _smooth_superquantile(self, w, x, y):

        sequence_losses, sequence_weights = self.seq_loss(w, x, y)
        simplex_center = sequence_weights
        q_mu = self._projection(sequence_losses, sequence_weights, simplex_center)

        f = np.dot(sequence_losses, q_mu) - self.smoothing_parameter * np.linalg.norm(q_mu - simplex_center) ** 2

        jacobian_l = self.seq_grad(w, x, y)
        g = np.transpose(jacobian_l).dot(q_mu[:jacobian_l.shape[0]])

        return f, g

    def _projection(self, u, weights, simplex_center):
        return fast_projection(u, weights, self.p, self.smoothing_parameter,
                simplex_center)

    def _theta_prime(self, lmbda, v, weights, sorted_index):
        return fast_theta_prime(lmbda, v, weights, sorted_index, self.p, self.smoothing_parameter)

    def _find_lmbda(self, v, weights, sorted_index):
        return fast_find_lmbda(v, weights, sorted_index, self.p, self.smoothing_parameter)

class OracleSmoothedNorm(AbstractOracle):
    def __init__(self, smoothing_parameter=0.01):
        self.smoothing_parameter=0.01
    
    def f(self, w, x, y):
        norm = np.linalg.norm(w)
        mu = 2*self.smoothing_parameter
        if norm >= mu:
            return norm - mu/2
        else:
            return norm**2/(2*mu)
    
    def g(self, w, x, y):
        norm = np.linalg.norm(w)
        mu = 2*self.smoothing_parameter
        if norm >= mu:
            return w/norm
        else:
            return w/mu

class OracleSmoothedWDRO(AbstractOracle):
    """
    Asssume the loss is of the form phi(y * x^Tw) with pĥi convex smooth 1-Lipschitz
    """
    def __init__(self, loss, loss_grad, ambiguity_radius, ambiguity_pen_labels, superquantile_smoothing_parameter=1000.0, norm_smoothing_parameter=0.01):
        self.loss = loss
        self.loss_grad = loss_grad
        self.ambiguity_radius = ambiguity_radius
        self.alpha = ambiguity_radius/ambiguity_pen_labels

        self.norm_oracle = OracleSmoothedNorm(smoothing_parameter=norm_smoothing_parameter)

        # TODO: improve efficiency by passing sequence loss vectors only to the SQ oracle
        sq_loss = lambda w, x, y : loss(w, x, -y) - loss(w, x, y) - ambiguity_pen_labels * self.norm_oracle.f(w, x, y)
        sq_loss_grad = lambda w, x, y : loss_grad(w, x, -y) - loss_grad(w, x, y) - ambiguity_pen_labels * self.norm_oracle.g(w, x, y)

        extra_samples = np.zeros(1, dtype=np.float64)
        extra_samples_weights = np.ones(1, dtype=np.float64)
        mixing_param = 0.5
        self.superquantile_oracle = OracleSmoothGradient(sq_loss, sq_loss_grad, 1 - self.alpha/2, smoothing_parameter=superquantile_smoothing_parameter,
                extra_samples=extra_samples, extra_samples_weights=extra_samples_weights, mixing_param=mixing_param)

    def f(self, w, x, y):
        return self.ambiguity_radius * self.norm_oracle.f(w, x, y) \
                + np.mean(self.loss(w, x, y)) \
                + self.alpha * self.superquantile_oracle.f(w, x, y)

    def g(self, w, x, y):
        return self.ambiguity_radius * self.norm_oracle.g(w, x, y) \
                + np.mean(self.loss_grad(w, x, y), axis=0) \
                + self.alpha * self.superquantile_oracle.g(w, x, y)
