from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from math import sqrt

METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error"
    "accuracy",
    "precision",
    "recall"
]

def get_metric(name: str):
    """Factory function to get a metric by name."""
    # Return a metric instance given its str name.
    if name == "mean_squared_error":
        output = MeanSquaredError()
    elif name == "mean_absolute_error":
        output = MeanAbsoluteError()
    elif name == "root_mean_squared_error":
        output = RootMeanSquaredError()
    elif name == "accuracy":
        output = Accuracy()
    elif name == "precision":
        output = Precision()
    elif name == "recall":
        output = Recall()
    else: 
        raise ValueError("The input is not a valid metric.")

    return output

class Metric(ABC):
    """Base class for all metrics."""

    def __init__(self):
        """Constructor method for the metric base class."""
        super().__init__()

    @abstractmethod
    def evaluate(ground_truth, Y) -> float:
        """An abstract method for evaluating the machine learning model."""
        return


    # remember: metrics take ground truth and prediction as input and return a real number

    def __call__(self):
        raise NotImplementedError("To be implemented.")

# add here concrete implementations of the Metric class






# Classes for regression
class MeanSquaredError(Metric):
    """Class for the mean squared error metric."""

    def __init__(self):
        super().__init__()
    
    def evaluate(self, ground_truth, Y) -> float:
        # MSE formula: (1/n) * Σ (y_pred - y_true)²
        squared_errors = sum((y_pred - y_true) ** 2 for y_true, y_pred in zip(ground_truth, Y))
        return squared_errors / len(ground_truth)
class MeanAbsoluteError(Metric):
    """Class for the mean absolute error metric."""
    def __init__(self):
        super().__init__()
    
    def evaluate(self, ground_truth, Y) -> float:
        # MAE formula = (1/n) * Σ |y_pred - y_true|        
        absolute_errors = sum(abs(y_true - y_pred) for y_true, y_pred in zip(ground_truth, Y))
        return absolute_errors / len(ground_truth)

class RootMeanSquaredError(Metric):
    """Class for the root mean squared error metric."""
    def __init__(self):
        super().__init__()

    def evaluate(self, ground_truth, Y) -> float:
        # RMSE formula: sqrt[(1/n) * Σ (y_pred - y_true)²]
        squared_errors = sum((y_pred - y_true) ** 2 for y_true, y_pred in zip(ground_truth, Y))
        mean_squared_error = squared_errors / len(ground_truth)
        return sqrt(mean_squared_error)





# Classes for classification
class Accuracy(Metric):
    """Class for the accuracy metric."""
    def __init__(self):
        super().__init__()

    def evaluate(self, ground_truth, Y) -> float:
        # Accuracy formula: correct predictions/all predictions
        correct_predictions = sum(1 for y_pred, y_true in zip(Y, ground_truth) if y_pred == y_true)
        return correct_predictions / len(ground_truth)

class Precision(Metric):
    """Class for the precision metric."""
    def __init__(self):
        super().__init__()
    
    def evaluate(self, ground_truth, Y) -> float:
        # Precision formula: TP / (TP + FP)
        TP = sum(1 for y_true, y_pred in zip(ground_truth, Y) if y_true == 1 and y_pred == 1)
        FP = sum(1 for y_true, y_pred in zip(ground_truth, Y) if y_true == 0 and y_pred == 1)
        
        if TP + FP == 0:
            return 0.0
        return TP / (TP + FP)

class Recall(Metric):
    """Class for the recall metric."""
    def __init__(self):
        super().__init__()
    
    def evaluate(self, ground_truth, Y):

        # Recall formula: TP / (TP + FN)
        TP = sum(1 for y_true, y_pred in zip(ground_truth, Y) if y_true == 1 and y_pred == 1)
        FN = sum(1 for y_true, y_pred in zip(ground_truth, Y) if y_true == 1 and y_pred == 0)
        
        if TP + FN == 0:
            return 0.0
        return TP / (TP + FN)
    