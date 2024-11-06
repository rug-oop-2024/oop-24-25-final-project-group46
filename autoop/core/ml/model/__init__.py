from autoop.core.ml.model.classification import DecisionTreeClassification, KNN, SupportVectorClassification
from autoop.core.ml.model.base_model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression, SVR, DecisionTreeRegression

REGRESSION_MODELS = [
    "DecisionTreeRegression",
    "MultipleLinearRegression",
    "SVR",
]  # add your models as str here

CLASSIFICATION_MODELS = [
    "DecisionTreeClassification",
    "KNN",
    "SupportVectorClassification",
]  # add your models as str here


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name == "DecisionTreeRegression":
        DecisionTreeRegression
    elif model_name == "MultipleLinearRegression":
        MultipleLinearRegression
    elif model_name == "SVR":
        SVR
    elif model_name == "DecisionTreeClassification":
        DecisionTreeClassification
    elif model_name == "KNN":
        KNN
    elif model_name == "SupportVectorClassification":
        SupportVectorClassification
    raise NotImplementedError("This is not a valid model, to be implemented.")
