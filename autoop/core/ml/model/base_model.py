
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
# from typing import Literal


class Model(Artifact, ABC):
    """Define a base model class for training and prediction."""

    def __init__(self, parameters: dict) -> None:
        """Initialize the model with an empty parameters dictionary."""
        ABC().__init__(self)
        Artifact().__init__(self)
        self._parameters = parameters


    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Use observations and ground truths to modify the internal state."""
        return

    @abstractmethod
    def predict(self, observations: np.ndarray) -> list:
        """Return a prediction using observations."""
        return

    @property
    def parameters(self) -> dict:
        """Provide a getter for the parameters variable."""
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, value: dict) -> None:
        if self._validate_dict(value):
            self._parameters = value
        else:
            raise ValueError("Invalid type, parameter type has to be a dict.")

    def _validate_dict(self, parameters: dict) -> bool:
        return isinstance(parameters, dict)

