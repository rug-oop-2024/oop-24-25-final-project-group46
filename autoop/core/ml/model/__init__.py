"""Create a constructor for the base model."""
from autoop.core.ml.model.classification import DecisionTreeClassification
from autoop.core.ml.model.classification import KNN
from autoop.core.ml.model.classification import RandomForestClassification

from autoop.core.ml.model.base_model import Model
from autoop.core.ml.model.regression import SupportVectorRegression
from autoop.core.ml.model.regression import DecisionTreeRegression
from autoop.core.ml.model.regression import MultipleLinearRegression


REGRESSION_MODELS = [
    "DecisionTreeRegression",
    "MultipleLinearRegression",
    "SupportVectorRegression",
]  # add your models as str here

CLASSIFICATION_MODELS = [
    "DecisionTreeClassification",
    "KNN",
    "RandomForestClassification",
]  # add your models as str here


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name == "DecisionTreeRegression":
        DecisionTreeRegression
    elif model_name == "MultipleLinearRegression":
        MultipleLinearRegression
    elif model_name == "SVR":
        SupportVectorRegression
    elif model_name == "DecisionTreeClassification":
        DecisionTreeClassification
    elif model_name == "KNN":
        KNN
    elif model_name == "RandomForestClassification":
        RandomForestClassification
    raise NotImplementedError("This is not a valid model, to be implemented.")
