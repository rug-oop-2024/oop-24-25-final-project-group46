import base64

from abc import ABC


class Artifact(ABC):
    """Create a base class for storing information about objects."""
    def __init__(
            self,
            name: str,
            asset_path: str,
            data: bytes,
            version: str,
            metadata: dict = None,
            type: str = None,
            tags: list = None
            ) -> None:
        """Create a constructor for the Artifact class."""
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.metadata = metadata if not None else {}
        self.type = type
        self.tags = tags if not None else []

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
            raise ValueError("asset_path and version have to be set to have id.")
        encoded_path = base64.b64encode(self.asset_path()).decode()
        return f"{encoded_path}:{self.version}"
