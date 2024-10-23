import base64
from abc import abstractmethod

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
    
    def get_asset_id(self) -> str:
        """Generates an id of an asset."""
        if not self.asset_path or not self.version: 
            raise ValueError("asset_path and version have to be set to generate an id.")
        encoded_path = base64.b64encode(self.asset_path()).decode()
        return f"{encoded_path}:{self.version}"
    

    pass
