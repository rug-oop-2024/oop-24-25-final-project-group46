import numpy as np

from sklearn.tree import DecisionTreeRegressor

from autoop.core.ml.model.base_model import Model

from typing import Optional

class DecisionTreeRegression(Model):
    """A wrapper for Decision Tree Regression."""

    def __init__(self, parameters: Optional[dict] = None, **kwargs) -> None:
        """Initialize the Multiple Linear Regression model."""
        super().__init__(self, name = "dtr", type = "regression", parameters = parameters, **kwargs)
        parameters = parameters if not None else {}
        self.model = DecisionTreeRegressor(**self.parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the Decision Tree Regressor to the data."""
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> list:
        """Predict target values using the trained model."""
        return self.model.predict(observations)