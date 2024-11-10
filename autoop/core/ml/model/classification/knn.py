import numpy as np
from autoop.core.ml.model.base_model import Model
from typing import Optional
from sklearn.neighbors import KNeighborsClassifier


class KNN(Model):
    """A class for finding the k nearest neighbours."""

    def __init__(self, parameters: Optional[dict] = None, **kwargs) -> None:
        """Initialize the KNN model with given parameters and k value. """
        super().__init__(name = "KNN", type = "classification", parameters = parameters, **kwargs)
        self.model = KNeighborsClassifier(**self.parameters)
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Store the training data and their labels."""
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict the labels for a set of observations."""
        return self.model.predict(observations)