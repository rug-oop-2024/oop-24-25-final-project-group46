import base64

import json 

from autoop.core.storage import default_storage_instance
from pydantic import BaseModel, Field


class Artifact(BaseModel):
    """Create a base class for storing information about objects."""
    type: str = Field(None)
    name: str = Field(None)
    asset_path: str = Field(None)
    data: bytes = Field(None)
    version: str = Field("1.0.0")
    tags: str = Field(default="")
    metadata: dict[str, str] = Field(default_factory=dict) 
    id : str = None

    def __init__(self, **data) -> None:
        """Create a constructor for the Artifact class."""
        super().__init__(**data)
        self.id = self.get_asset_id()



    def read(self) -> bytes:
        """Create a method for reading the data."""
        return self.data

    def save(self, data: bytes = None) -> None:
        """Save the artifact's data and metadata."""
        self.data = data
        return self.data

    def get_asset_id(self) -> str:
        """Generate an id based on asset_path and version."""
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        correct_version = self.version.replace('.', '_').replace(';', '_'). replace(',', '_').replace('=', '_')
        return f"{encoded_path}_{correct_version}"
