"""Create a constructor for the classification models."""
from autoop.core.ml.model.classification.dtc import DecisionTreeClassification
from autoop.core.ml.model.classification.knn import KNN
from autoop.core.ml.model.classification.rf import RandomForestClassification

__all__ = [
    "DecisionTreeClassification",
    "KNN",
    "RandomForestClassification",
]
