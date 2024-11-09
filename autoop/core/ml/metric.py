from abc import ABC, abstractmethod
from math import sqrt
from typing import Dict  # , List
import numpy as np

METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "accuracy",
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
    def evaluate(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """Evaluate the model based on the ground truth and predictions."""
        return

    def __call__(self) -> None:
        """Initiate call method for base class."""
        return f"{__class__.__name__}"


# Classes for regression
class MeanSquaredError(Metric):
    """Class for the mean squared error metric."""

    def evaluate(self, ground_truth: np.ndarray, predictions: np.ndarray) -> float:
        """Evaltuate the model."""
        squared_errors = np.sum((ground_truth - predictions)**2)
        return squared_errors / len(ground_truth)


class MeanAbsoluteError(Metric):
    """Class for the mean absolute error metric."""

    def evaluate(self,ground_truth: np.ndarray, predictions: np.ndarray) -> float:
        """Evaltuate the model."""
        return np.mean(np.abs(ground_truth - predictions))

        
class RootMeanSquaredError(Metric):
    """Class for the root mean squared error metric."""

    def evaluate(self,ground_truth: np.ndarray, predictions: np.ndarray) -> float:
        """Evaltuate the model."""
        squared_errors = np.sum((ground_truth - predictions)**2)
        denominator = np.sum((ground_truth - np.mean(ground_truth))**2)
        return (1 - squared_errors) / denominator


# Classes for classification
class Accuracy(Metric):
    """Class for the accuracy metric."""

    def evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Evaluate the model's accuracy based on ground truth and predictions."""
        true_positives = np.sum(ground_truth == predictions)
        return true_positives/len(ground_truth)


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
