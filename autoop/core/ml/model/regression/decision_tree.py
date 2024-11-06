from autoop.core.ml.model import Model

from sklearn.tree import DecisionTreeRegressor

class DecisionTreeRegression(Model):
    """A wrapper for Decision Tree Regression."""

    def __init__(self, parameters: dict = None) -> None:
        """Create a constructor for the Decision Tree model."""
        super().__init__(parameters if parameters is not None else {})
        self.model = DecisionTreeRegressor(**self._parameters)

    def fit(self, observations, ground_truth) -> None:
        """Fit the Decision Tree Regressor to the data."""
        self.model.fit(observations, ground_truth)

    def predict(self, observations) -> list:
        """Predict target values using the trained model."""
        return self.model.predict(observations).tolist()
