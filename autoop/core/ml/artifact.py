import base64
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    name: str = Field(default=None)
    asset_path: str = Field(default=None)
    data: bytes = Field(default=None)
    version: str = Field(default=None)
    metadata: dict = Field(default=None)
    type: str = Field(default=None)
    tags: list = Field(default=None)

    @abstractmethod
    def read(self):
        """Create an abstract method for reading the data."""
        return

    @abstractmethod
    def save(self, data):
        """Create an abstract method for saving the data."""
        return

    pass
