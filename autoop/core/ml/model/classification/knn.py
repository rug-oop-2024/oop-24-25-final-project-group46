from collections import Counter
import numpy as np
from autoop.core.ml.model.base_model import Model

class KNN(Model):
    """A class for finding the k nearest neighbours."""

    def __init__(self, parameters: dict, num_k: int) -> None:
        """Initialize the KNN model with given parameters and k value. """
        super().__init__(parameters if parameters is not None else {})
        self.k = num_k

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Store the training data and their labels."""
        # Update parameters to store training data
        self.parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray) -> list:
        """Predict the labels for a set of observations."""
        predictions = []
        training_observations = self.parameters.get("observations")
        ground_truth = self.parameters.get("ground_truth")

        for single_observation in observations:
            # Calculate distances to all training points.
            distances = np.linalg.norm(training_observations - single_observation, axis=1)
            # Find k nearest neighbors.
            k_indices = np.argsort(distances)[:self.k]
            # Retrieve labels of nearest neighbors.
            k_nearest_labels = ground_truth[k_indices]
            # Make prediction based on the most common label.
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions

