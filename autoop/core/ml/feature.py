
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    name: str = Field(default=None)
    type: str = Field(default=None)
    
    def __str__(self):
        raise NotImplementedError("To be implemented.")