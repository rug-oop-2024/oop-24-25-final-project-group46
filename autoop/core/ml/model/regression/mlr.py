import numpy as np
from autoop.core.ml.model.base_model import Model

class MultipleLinearRegression(Model):
    """A class for Multiple Linear Regression model."""

    def __init__(self) -> None:
        """Initialize the Multiple Linear Regression model."""
        super().__init__(parameters={})

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the model by finding optimal parameters using the observations and their labels."""
        # Add a column of ones to the observations for the intercept term.
        ones_column = np.ones((observations.shape[0], 1))
        x_added_column = np.column_stack([observations, ones_column])

        # Transpose the observation matrix.
        x_transposed = x_added_column.T
        # Calculate (X^T * X)^(-1) * X^T * y
        weights_and_intercept = np.linalg.inv(x_transposed @ x_added_column) @ x_transposed @ ground_truth

        # Store the parameters (weights and intercept) for use in predictions.
        self._parameters["weights_and_intercept"] = weights_and_intercept

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict target values using the trained model."""
        # Add a column of ones for the intercept term in predictions.
        ones_column = np.ones((observations.shape[0], 1))
        x_added_column = np.column_stack([observations, ones_column])

        # Retrieve the stored weights and intercept.
        weights_and_intercept = self.parameters.get("weights_and_intercept")

        # Return predictions by calculating X * weights_and_intercept.
        return x_added_column @ weights_and_intercept

