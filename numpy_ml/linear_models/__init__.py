"""A module containing assorted linear models."""
"""一个包含各种线性模型的模块。"""

from .ridge import RidgeRegression
from .glm import GeneralizedLinearModel
from .logistic import LogisticRegression
from .bayesian_regression import (
    BayesianLinearRegressionKnownVariance,
    BayesianLinearRegressionUnknownVariance,
)
from .naive_bayes import GaussianNBClassifier
from .linear_regression import LinearRegression
