import numpy as np

from sklearn.tree import DecisionTreeRegressor

from autoop.core.ml.model.base_model import Model

from typing import Optional

class DecisionTreeRegression(Model):
    """A wrapper for Decision Tree Regression."""

    def __init__(self, parameters: Optional[dict] = None, **kwargs) -> None:
        """Create a constructor for the Decision Tree Regression model."""
        self._type = "regression"
        super().__init__(name = "DecisionTreeRegression", type = self._type, parameters = parameters, **kwargs)
        self.model = DecisionTreeRegressor(**self._parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the Decision Tree Regressor to the data."""
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict target values using the trained model."""
        return self.model.predict(observations)