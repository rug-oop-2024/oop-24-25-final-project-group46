from sklearn.tree import DecisionTreeClassifier

from autoop.core.ml.model.base_model import Model

import numpy as np

from typing import Optional


class DecisionTreeClassification(Model):
    """A wrapper for Decision Tree Classification."""
    def __init__(self, parameters: Optional[dict] = None, **kwargs) -> None:
        """Create a constructor for the Decision Tree Classifier model."""
        super().__init__(
            name="DecisionTreeClassification",
            type="classification",
            parameters=parameters,
            **kwargs
        )
        self._type = "classification"
        self.model = DecisionTreeClassifier(**self._parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the Decision Tree Classifier to the data."""
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict class labels for new observations."""
        return self.model.predict(observations)