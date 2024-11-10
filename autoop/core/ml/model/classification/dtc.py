from sklearn.tree import DecisionTreeClassifier

from autoop.core.ml.model.base_model import Model

import numpy as np

from typing import Optional


class DecisionTreeClassification(Model):
    """A wrapper for Decision Tree Classification."""
    def __init__(self, parameters: Optional[dict] = None, **kwargs) -> None:
        """Create a constructor for the Decision Tree Classifier model."""
        if parameters is None:
            parameters = {}
        super().__init__(self, name="DecisionTreeClassification", type="classification", **kwargs)
        self._model = DecisionTreeClassifier(**self._parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the Decision Tree Classifier to the data."""
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict class labels for new observations."""
        return self._model.predict(observations)
