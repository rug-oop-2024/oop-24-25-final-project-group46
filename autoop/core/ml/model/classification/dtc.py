from sklearn.tree import DecisionTreeClassifier

from autoop.core.ml.model.base_model import Model

import numpy as np


class DecisionTreeClassification(Model):
    """A wrapper for Decision Tree Classification."""

    def __init__(self, parameters: dict = None, type: str = None) -> None:
        """Create a constructor for the Decision Tree Classifier model."""
        self._parameters = parameters
        self._type = type
        self.model = DecisionTreeClassifier(**self._parameters, self._type)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the Decision Tree Classifier to the data."""
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> list:
        """Predict class labels for new observations."""
        return self.model.predict(observations).tolist()
