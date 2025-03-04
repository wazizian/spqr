"""Core module
.. moduleauthor:: Yassine LAGUEL
"""

from .oracle import OracleSubgradient, OracleSmoothGradient
from .risk_optimization import RiskOptimizer
from .estimators import DRLinearRegression, DRLogisticRegression

from ._version import __version__

__all__ = ['__version__', 'OracleSubgradient', 'OracleSmoothGradient',  'RiskOptimizer',
           'DRLinearRegression', 'DRLogisticRegression']
