import base64

from abc import ABC


class Artifact(ABC):

    def __init__(
            self,
            name: str,
            asset_path: str,
            data: bytes,
            version: str,
            metadata: dict,
            type: str, 
            tags: list
        ) -> None: 
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.metadata = metadata
        self.type = type
        self.tags = tags

    def read(self) -> bytes:
        """Create a method for reading the data."""
        return self.data
    
    def save(self, data: bytes) -> bytes:
        """Create a method for saving the data."""
        self.data = data
        return data
    
    def get_asset_id(self) -> str:
        """Generates an id of an asset."""
        if not self.asset_path or not self.version: 
            raise ValueError("asset_path and version have to be set to generate an id.")
        encoded_path = base64.b64encode(self.asset_path()).decode()
        return f"{encoded_path}:{self.version}"
