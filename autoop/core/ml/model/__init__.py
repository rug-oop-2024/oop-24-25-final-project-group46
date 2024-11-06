from autoop.core.ml.model.classification.decision_tree_class import (
    DecisionTreeClassification,
)
from autoop.core.ml.model.classification.knn import KNN
from autoop.core.ml.model.classification.svc import SupportVectorClassification
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.decision_tree_regr import DecisionTreeRegression
from autoop.core.ml.model.regression.mlr import MultipleLinearRegression
from autoop.core.ml.model.regression.svm import SVR

REGRESSION_MODELS = [
    DecisionTreeRegression,
    MultipleLinearRegression,
    SVR,
]  # add your models as str here

CLASSIFICATION_MODELS = [
    DecisionTreeClassification,
    KNN,
    SupportVectorClassification,
]  # add your models as str here


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    raise NotImplementedError("To be implemented.")
