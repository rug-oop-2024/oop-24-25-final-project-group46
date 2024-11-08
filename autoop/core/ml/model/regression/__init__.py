"""Store regression models for base __init__."""
from autoop.core.ml.model.regression.dtr import DecisionTreeRegression
from autoop.core.ml.model.regression.mlr import MultipleLinearRegression
from autoop.core.ml.model.regression.svm import SupportVectorRegression

__all__ = [
    "DecisionTreeRegression",
    "MultipleLinearRegression",
    "SupportVectorRegression",
]
