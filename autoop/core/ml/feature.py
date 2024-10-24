
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset
# zelf toegevoegd
from autoop.functional.feature import detect_feature_types

class Feature(BaseModel):
    name: str = Field(default=None)
    type: str = Field(default=None)

    def __str__(self):
        return f"Feature (name={self.name}, type={self.type})"
    
    

    
    