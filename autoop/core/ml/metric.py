from abc import ABC, abstractmethod
from math import sqrt
from typing import Dict  # , List
import numpy as np

METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error" "accuracy",
    "micro_precision",
    "mirco_recall",
]


def get_metric(name: str) -> "Metric":
    """Return a metric instance given its name as a string."""
    metrics: Dict[str, Metric] = {
        "mean_squared_error": MeanSquaredError(),
        "mean_absolute_error": MeanAbsoluteError(),
        "root_mean_squared_error": RootMeanSquaredError(),
        "accuracy": Accuracy(),
        "micro_precision": Precision(),
        "micro_recall": Recall(),
    }

    if name in metrics:
        return metrics[name]
    raise ValueError("The input is not a valid metric.")


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def evaluate(
            self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """Evaluate the model based on the ground truth and predictions."""
        return

    def __call__(self) -> None:
        """Initiate call method for base class."""
        raise NotImplementedError("To be implemented.")


# Classes for regression
class MeanSquaredError(Metric):
    """Class for the mean squared error metric."""

    def evaluate(
            self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """Evaltuate the model."""
        # MeanSquaredError formula: (1/n) * Σ (y_prediction - y_true)²
        # Feat is features
        feat = zip(prediction, ground_truth)
        squared_errors = sum((y_pred - y_true) ** 2 for y_true, y_pred in feat)
        return squared_errors / len(ground_truth)


class MeanAbsoluteError(Metric):
    """Class for the mean absolute error metric."""

    # MeanAbsoluteError formula = (1/n) * Σ |y_prediction - y_true|
    def evaluate(
        self,
        ground_truth: np.ndarray, prediction: np.ndarray
    ) -> float:
        """Evaltuate the model."""
        feat = zip(prediction, ground_truth)  # Feat is features
        absolute_errors = sum(abs(y_true - y_pred) for y_true, y_pred in feat)
        return absolute_errors / len(ground_truth)


class RootMeanSquaredError(Metric):
    """Class for the root mean squared error metric."""

    # Formula: square root[(1/n) * Σ (y_prediction - y_true)²]
    def evaluate(
        self,
        ground_truth: np.ndarray, prediction: np.ndarray
    ) -> float:
        """Evaltuate the model."""
        feat = zip(prediction, ground_truth)  # Feat is features
        squared_errors = sum((y_pred - y_true) ** 2 for y_true, y_pred in feat)
        mean_squared_error = squared_errors / len(ground_truth)
        return sqrt(mean_squared_error)


# Classes for classification
class Accuracy(Metric):
    """Class for the accuracy metric."""

def evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Evaluate the model's accuracy based on ground truth and predictions."""
    # Flatten to ensure 1D arrays
    predictions = predictions.flatten()
    ground_truth = ground_truth.flatten()
    
    # Calculate correct predictions using np.array_equal for multi-element comparisons
    correct_pred = sum(1 for y_pred, y_true in zip(predictions, ground_truth) if np.array_equal(y_pred, y_true))
    return correct_pred / len(ground_truth)


class Precision(Metric):
    """Class for the precision metric."""

    def evaluate(
        self, ground_truth: np.ndarray, prediction: np.ndarray
    ) -> float:
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

    def evaluate(
        self, ground_truth: np.ndarray, prediction: np.ndarray
    ) -> float:
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
