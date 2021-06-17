from .wdr_risk_optimization import WassersteinRiskOptimizer
from .estimators import DRLogisticRegression

# This uses cooperative multiple inheritance: calls of the
# form super().method(...) in the parent DRLogisticRegression will actually call
# methods from WassertsteinRiskOptimizer instead of those of RiskOptimizer
#
# References:
# https://stackoverflow.com/a/5446718
# https://rhettinger.wordpress.com/2011/05/26/super-considered-super/
# https://docs.python.org/3/library/functions.html#super

class WDRLogisticRegression(DRLogisticRegression, WassersteinRiskOptimizer):

    def __init__(self, rho=0.5, kappa=0.2, mu=1.0, mu_norm=0.01, **kwargs):

        DRLogisticRegression.__init__(self, p = 1 - 0.5*rho/kappa, mu=mu, **kwargs) 

        params = {
            'algorithm': 'l-bfgs',
            'rho': rho,
            'kappa': kappa,
            'mu': mu,
            'mu_norm': mu_norm,
        }

        WassersteinRiskOptimizer.__init__(self, self.logistic_loss, self.logistic_grad, **params)

