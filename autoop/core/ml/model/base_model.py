from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Optional
import pickle


class Model(Artifact, ABC):
    """Define a base model class for training and prediction."""

    def __init__(self, name:str, type: Literal["regression", "classification"], parameters: Optional[dict] = None, model: object = None, **kwargs) -> None:
        """Initialize the model with an empty parameters dictionary."""
        self._parameters = parameters if parameters is not None else {}
        self._model = model if model is not None else {}
        self._type = type
        Artifact.__init__(
            self,
            name=name,
            asset_path=f"models/{name}.pkl",
            data=b"",
            version="1.0",
            **kwargs,
        )
 


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
    
    @property
    def type(self) -> str:
        """Provide a getter for the parameters variable."""
        return self._type
    
    @parameters.setter
    def type(self, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Invalid type, type has to be a string.")
        self._type = value
    
    def to_artifact(self, name: str) -> "Artifact":
        """Define a method to convert the model to an artifact."""
        model = pickle.dumps(self)
        artifact = Artifact(
            name=name,
            data=model,
            type="model",
        )
        return artifact