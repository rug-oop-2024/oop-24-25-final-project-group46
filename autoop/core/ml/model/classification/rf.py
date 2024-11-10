from sklearn.ensemble import RandomForestClassifier

from autoop.core.ml.model.base_model import Model

import numpy as np

from typing import Optional


class RandomForestClassification(Model):
    """A wrapper for Support Vector Regression using scikit-learn's SVR."""

    def __init__(self, parameters: Optional[dict] = None, **kwargs) -> None:
        """Create a constructor for the SVR model."""
        super().__init__(
            name="RandomForestClassification",
            type="classification",
            parameters=parameters,
            **kwargs
        )
        self._type = "classification"
        self.model = RandomForestClassifier(**self._parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the Support Vector Regressor to the data."""
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict target values using the trained model."""
        return self.model.predict(observations)