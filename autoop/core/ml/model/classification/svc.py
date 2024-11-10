from sklearn.svm import SVC

from autoop.core.ml.model.base_model import Model

import numpy as np

from typing import Optional


class SupportVectorClassification(Model):
    """A wrapper for Support Vector Regression using scikit-learn's SVR."""

    def __init__(self, parameters: Optional[dict] = None, **kwargs) -> None:
        """Create a constructor for the SVR model."""
        super().__init__(name="SupportVectorClassification", type="classification", parameters=parameters, **kwargs)
        self.model = SVC(**self._parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the Support Vector Regressor to the data."""
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> list:
        """Predict target values using the trained model."""
        return self.model.predict(observations).tolist()