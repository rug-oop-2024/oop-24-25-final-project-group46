import base64

from abc import ABC
import json

from autoop.core.storage import default_storage_instance


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
            tags: str = None
    ) -> None:
        """Create a constructor for the Artifact class."""
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.metadata = metadata if not None else {}
        self._type = type
        self.tags = tags if not None else []
        self.id = self.get_asset_id()
        self._storage = default_storage_instance

    def read(self) -> bytes:
        """Create a method for reading the data."""
        return self.data

    def save(self, data: bytes = None) -> None:
        """Save the artifact's data and metadata."""
        if data is not None:
            self.data = data  

        # Save the main data
        print(f"Saving main data to {self.asset_path}")
        self._storage.save(self.data, self.asset_path)

        # Save the metadata as a JSON file in the same path
        metadata_path = f"{self.asset_path}_metadata.json"

        metadata_content = {
            "name": self.name,
            "version": self.version,
            "tags": self.tags,
            "metadata": self.metadata,
            "type": self._type,
            "id": self.id,
            "asset_path": self.asset_path
        }

        print(f"Saving metadata to {metadata_path}")
        self._storage.save(json.dumps(
            metadata_content).encode(),
            metadata_path
        )
        return self.data

    def get_asset_id(self) -> str:
        """Generate an id based on asset_path and version."""
        if not self.asset_path or not self.version:
            raise ValueError(
                "Both asset_path and version are required to generate an id."
            )
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"
