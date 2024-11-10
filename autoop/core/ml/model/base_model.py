
from abc import abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Optional
import pickle
from pydantic import BaseModel, Field


class Model(Artifact, BaseModel):
    """Define a base model class for training and prediction."""

    parameters : Optional[dict] = Field()
    model: object = None

    def __init__(self, name:str, *args, **kwargs) -> None:
        """Initialize the model with an empty parameters dictionary."""
        BaseModel().__init__(self, **kwargs)
        Artifact().__init__(
            self,
            name=name,
            asset_path=f"autoop/core/ml/model/{name}.py",
            *args,
            **kwargs
        )
 


    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Use observations and ground truths to modify the internal state."""
        return

    @abstractmethod
    def predict(self, observations: np.ndarray) -> list:
        """Return a prediction using observations."""
        return

    # @property
    # def parameters(self) -> dict:
    #     """Provide a getter for the parameters variable."""
    #     return deepcopy(self._parameters)

    # @parameters.setter
    # def parameters(self, value: dict) -> None:
    #     if self._validate_dict(value):
    #         self._parameters = value
    #     else:
    #         raise ValueError("Invalid type, parameter type has to be a dict.")

    # def _validate_dict(self, parameters: dict) -> bool:
    #     return isinstance(parameters, dict)
    
    
    # def to_artifact(self, name: str) -> "Artifact":
    #     """Define a method to convert the model to an artifact."""
    #     model = pickle.dumps(self)
    #     artifact = Artifact(
    #         name=name,
    #         data=model,
    #         type=self.type,
    #         asset_path = self.asset_path
    #     )
    #     return artifact


