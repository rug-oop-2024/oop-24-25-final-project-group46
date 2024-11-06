from sklearn.tree import DecisionTreeClassifier

from autoop.core.ml.model import Model


class DecisionTreeClassification(Model):
    """A wrapper for Decision Tree Classification."""

    def __init__(self, parameters: dict = None) -> None:
        """Create a constructor for the Decision Tree Classifier model."""
        super().__init__(parameters if parameters else {})
        self.model = DecisionTreeClassifier(**self._parameters)

    def fit(self, observations, ground_truth) -> None:
        """Fit the Decision Tree Classifier to the data."""
        self.model.fit(observations, ground_truth)

    def predict(self, observations) -> list:
        """Predict class labels for new observations."""
        return self.model.predict(observations).tolist()
