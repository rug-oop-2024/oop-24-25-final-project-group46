import numpy as np

from sklearn.tree import DecisionTreeRegressor

from autoop.core.ml.model.base_model import Model


class DecisionTreeRegression(Model):
    """A wrapper for Decision Tree Regression."""

    def __init__(self, parameters: dict = None, type: str = None) -> None:
        """Create a constructor for the Decision Tree model."""
        super().__init__(parameters if parameters is not None else {})
        self.type = "classification"
        self.model = DecisionTreeRegressor(**self._parameters)


    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the Decision Tree Regressor to the data."""
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> list:
        """Predict target values using the trained model."""
        return self.model.predict(observations)