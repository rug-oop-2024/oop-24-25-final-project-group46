from abc import ABC, abstractmethod
from math import sqrt

METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error" "accuracy",
    "precision",
    "recall",
]


def get_metric(self, name: str):
    """Return a metric instance given its name as a string."""
    metrics = {
        "mean_squared_error": MeanSquaredError(),
        "mean_absolute_error": MeanAbsoluteError(),
        "root_mean_squared_error": RootMeanSquaredError(),
        "accuracy": Accuracy(),
        "precision": Precision(),
        "recall": Recall(),
    }

    if name in metrics:
        return metrics[name]
    raise ValueError("The input is not a valid metric.")


class Metric(ABC):
    """Base class for all metrics."""

    def __init__(self):
        """Construct the method for the metric base class."""
        super().__init__()

    @abstractmethod
    def evaluate(self, ground_truth, prediction) -> float:
        """Evaluate the model based on the ground truth and predictions."""
        return

    def __call__(self):
        """Initiate call method for base class."""
        raise NotImplementedError("To be implemented.")


# Classes for regression
class MeanSquaredError(Metric):
    """Class for the mean squared error metric."""

    def __init__(self):
        """Construct the MeanSquaredError class."""
        super().__init__()

    def evaluate(self, ground_truth, prediction) -> float:
        """Evaltuate the model."""
        # MeanSquaredError formula: (1/n) * Σ (y_prediction - y_true)²
        squared_errors = sum(
            (y_pred - y_true) ** 2 for y_true, y_pred in zip(ground_truth, prediction)
        )
        return squared_errors / len(ground_truth)


class MeanAbsoluteError(Metric):
    """Class for the mean absolute error metric."""

    def __init__(self):
        """Construct the MeanAbsoluteError class."""
        super().__init__()

    def evaluate(self, ground_truth, prediction) -> float:
        """Evaltuate the model."""
        # MeanAbsoluteError formula = (1/n) * Σ |y_prediction - y_true|
        absolute_errors = sum(
            abs(y_true - y_pred) for y_true, y_pred in zip(ground_truth, prediction)
        )
        return absolute_errors / len(ground_truth)


class RootMeanSquaredError(Metric):
    """Class for the root mean squared error metric."""

    def __init__(self):
        """Construct the RootsMeanSquaredError class."""
        super().__init__()

    def evaluate(self, ground_truth, prediction) -> float:
        """Evaltuate the model."""
        # Formula: square root[(1/n) * Σ (y_prediction - y_true)²]
        squared_errors = sum(
            (y_pred - y_true) ** 2 for y_true, y_pred in zip(ground_truth, prediction)
        )
        mean_squared_error = squared_errors / len(ground_truth)
        return sqrt(mean_squared_error)


# Classes for classification
class Accuracy(Metric):
    """Class for the accuracy metric."""

    def __init__(self):
        """Construct the Accuracy class."""
        super().__init__()

    def evaluate(self, ground_truth, prediction) -> float:
        """Evaltuate the model."""
        # Accuracy formula: correct predictions/all predictions
        correct_predictions = sum(
            1 for y_pred, y_true in zip(prediction, ground_truth) if y_pred == y_true
        )
        return correct_predictions / len(ground_truth)


class Precision(Metric):
    """Class for the precision metric."""

    def __init__(self):
        """Construct the Percision class."""
        super().__init__()

    def evaluate(self, ground_truth, prediction) -> float:
        """Evaltuate the model."""
        # Precision formula: true_positive / (true_positive + false_positive)
        tp = sum(
            1
            for y_true, y_pred in zip(ground_truth, prediction)
            if y_true == 1 and y_pred == 1
        )
        fp = sum(
            1
            for y_true, y_pred in zip(ground_truth, prediction)
            if y_true == 0 and y_pred == 1
        )

        if tp + fp == 0:
            return 0.0
        return tp / (tp + tp)


class Recall(Metric):
    """Class for the Recall metric."""

    def __init__(self):
        """Construct the Rcall class."""
        super().__init__()

    def evaluate(self, ground_truth, prediction):
        """Evaltuate the model."""
        # Recall formula: true_positive / (true_positive + false_negative)
        tp = sum(
            1
            for y_true, y_pred in zip(ground_truth, prediction)
            if y_true == 1 and y_pred == 1
        )
        fn = sum(
            1
            for y_true, y_pred in zip(ground_truth, prediction)
            if y_true == 1 and y_pred == 0
        )

        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
