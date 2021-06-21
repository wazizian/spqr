from .new_oracle import OracleSmoothedWDRO
from .risk_optimization import RiskOptimizer
from .algorithms.gradient_method import GradientMethod
from .algorithms.quasi_newton import LBFGS

class WassersteinRiskOptimizer(RiskOptimizer):
    def __init__(self, loss, loss_grad, params={}, **kwargs):
        # TODO: rewrite these manipulations with '|' when in 3.9
        merged_params = params.copy()
        merged_params.update(kwargs)
        super().__init__(loss, loss_grad, params=merged_params)

        self.params = self.default_params()
        self.params.update(params)
        self.params.update(kwargs)

        if self.params['max_iter'] is not None:
            key = self.params['algorithm'] + '_nb_iterations'
            self.params[key] = self.params['max_iter']

        self.oracle = OracleSmoothedWDRO(loss, loss_grad, self.params['rho'],
                self.params['kappa'],
                superquantile_smoothing_parameter=self.params['mu'],
                norm_smoothing_parameter=self.params['mu_norm'])

        if self.params['algorithm'] == 'gradient':
            self.algorithm = GradientMethod(self.oracle, self.params)
        elif self.params['algorithm'] == 'l-bfgs':
            self.algorithm = LBFGS(self.oracle, self.params)
        else:
            raise NotImplementedError(f"algorithm: {self.params['algorithm']}")

    def default_params(self):
        params = {
            # General Parameters
            'algorithm': 'gradient',
            'w_start': None,
            'alpha': None,
            'alpha_start': 100.0,
            'max_iter': None,
            
            # WDRO Parameters
            'rho': 1,
            'kappa': 2,

            # Smoothing Parameters
            'mu': 1000.0,
            'mu_norm': 0.01,

            # Gradient Parameters
            'gradient_stepsize_decrease': lambda k: 1.0,
            'gradient_nb_iterations': 100,

            # LBFGS Parameters
            'l-bfgs_nb_iterations': 1000,
        }
        return params
