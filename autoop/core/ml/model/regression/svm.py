import numpy as np

from sklearn.svm import SVR

from autoop.core.ml.model.base_model import Model

from typing import Optional

class SupportVectorRegression(Model):
    """A wrapper for Support Vector Regression using scikit-learn's SVR."""

    def __init__(self, parameters: Optional[dict] = None, **kwargs) -> None:
        """Create a constructor for the Support vector regression model."""
        self._type = "regression"
        super().__init__(name = "SupportVectorRegression", type = self._type, parameters = parameters, **kwargs)
        self.model = SVR(**self._parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the Support Vector Regressor to the data."""
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict target values using the trained model."""
        return self.model.predict(observations)